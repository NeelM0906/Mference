@preconcurrency import AVFoundation
import AppKit
import CoreAudio
import Foundation
import Synchronization

// Capture for the live app (Notes spec 6.3): the microphone through
// AVAudioEngine and the call through a Core Audio process tap
// (`CATapDescription` + a private aggregate device), both converted to
// 16 kHz mono. Only `MferenceAppCore/Speech` captures audio (CONTRACTS.md).
// Never used in tests: they use `FakeAudioSourceFactory`.

/// The real sources and device lists.
public struct LiveAudioSourceFactory: AudioSourceFactory {
    public init() {}

    public func microphone(deviceID: String?) throws -> any AudioSource {
        MicrophoneSource(deviceUID: deviceID)
    }

    public func call(_ source: CallAudioSource) throws -> any AudioSource {
        switch source {
        case .off: throw SpeechError.failed("call audio is off")
        case .allApps: ProcessTapSource(target: .allApps)
        case .app(let bundleID, _): ProcessTapSource(target: .app(bundleID))
        }
    }

    public func inputDevices() async -> [AudioInputDevice] { CoreAudioDevices.inputDevices() }
    public func runningAudioApps() async -> [RunningAudioApp] { await CoreAudioDevices.runningAudioApps() }
}

// MARK: Microphone

/// The microphone through AVAudioEngine's input tap. A device or format
/// change (headphones plugged in) restarts the tap with the new format.
final class MicrophoneSource: AudioSource, @unchecked Sendable {
    let track = TrackKind.microphone
    private let deviceUID: String?
    private let lock = NSLock()
    private var engine: AVAudioEngine?
    private var continuation: AsyncThrowingStream<AudioChunk, Error>.Continuation?
    private var observer: NSObjectProtocol?

    init(deviceUID: String?) { self.deviceUID = deviceUID }

    func start() async throws -> AsyncThrowingStream<AudioChunk, Error> {
        guard AVCaptureDevice.authorizationStatus(for: .audio) == .authorized else {
            throw SpeechError.permissionDenied(.microphone)
        }
        let (stream, continuation) = AsyncThrowingStream.makeStream(of: AudioChunk.self)
        let engine = AVAudioEngine()
        lock.withLock {
            self.engine = engine
            self.continuation = continuation
        }
        do {
            try configure(engine)
        } catch {
            await stop()
            throw error as? SpeechError ?? .failed(error.localizedDescription)
        }
        observer = NotificationCenter.default.addObserver(forName: .AVAudioEngineConfigurationChange, object: engine,
                                                          queue: nil) { [weak self] _ in self?.restart() }
        continuation.onTermination = { [weak self] _ in Task { await self?.stop() } }
        return stream
    }

    private func configure(_ engine: AVAudioEngine) throws {
        let input = engine.inputNode
        if let deviceUID, let device = CoreAudioDevices.deviceID(forUID: deviceUID), let unit = input.audioUnit {
            var id = device
            let status = AudioUnitSetProperty(unit, kAudioOutputUnitProperty_CurrentDevice, kAudioUnitScope_Global, 0,
                                              &id, UInt32(MemoryLayout<AudioDeviceID>.size))
            if status != noErr { throw SpeechError.failed("cannot use the chosen microphone (\(status))") }
        }
        let format = input.outputFormat(forBus: 0)
        guard format.sampleRate > 0, format.channelCount > 0 else {
            throw SpeechError.failed("the microphone has no input format")
        }
        let resampler = try AudioResampler(inputFormat: format)
        input.installTap(onBus: 0, bufferSize: 4_096, format: format) { [weak self] buffer, _ in
            guard let self, let samples = try? resampler.convert(buffer), !samples.isEmpty else { return }
            self.lock.withLock { self.continuation }?.yield(AudioChunk(samples: samples))
        }
        engine.prepare()
        try engine.start()
    }

    private func restart() {
        guard let engine = lock.withLock({ self.engine }) else { return }
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
        do {
            try configure(engine)
        } catch {
            lock.withLock { self.continuation }?.finish(throwing: error)
        }
    }

    func stop() async {
        let (engine, continuation) = lock.withLock { () -> (AVAudioEngine?, AsyncThrowingStream<AudioChunk, Error>.Continuation?) in
            defer {
                self.engine = nil
                self.continuation = nil
            }
            return (self.engine, self.continuation)
        }
        if let observer { NotificationCenter.default.removeObserver(observer) }
        observer = nil
        engine?.inputNode.removeTap(onBus: 0)
        engine?.stop()
        continuation?.finish()
    }
}

// MARK: Call audio

/// Other apps' output through a Core Audio process tap: every app but this
/// one (the safe default, spec 12), or one app with its helper processes
/// (browsers and call apps play through helpers).
final class ProcessTapSource: AudioSource, @unchecked Sendable {
    enum Target: Sendable { case allApps, app(String) }

    let track = TrackKind.call
    private let target: Target
    private let lock = NSLock()
    private var tap = ProcessTap()
    private var continuation: AsyncThrowingStream<AudioChunk, Error>.Continuation?

    init(target: Target) { self.target = target }

    func start() async throws -> AsyncThrowingStream<AudioChunk, Error> {
        let (stream, continuation) = AsyncThrowingStream.makeStream(of: AudioChunk.self)
        lock.withLock { self.continuation = continuation }
        do {
            let processes: [AudioObjectID]
            switch target {
            case .allApps: processes = []
            case .app(let bundleID):
                processes = CoreAudioDevices.processObjects(bundleID: bundleID)
                if processes.isEmpty { throw SpeechError.failed("\(bundleID) is not playing audio") }
            }
            var resampler: AudioResampler?
            try tap.start(processes: processes) { [weak self] buffer in
                guard let self else { return }
                if resampler == nil { resampler = try? AudioResampler(inputFormat: buffer.format) }
                guard let samples = try? resampler?.convert(buffer), !samples.isEmpty else { return }
                self.lock.withLock { self.continuation }?.yield(AudioChunk(samples: samples))
            }
        } catch {
            tap.stop()
            lock.withLock { self.continuation = nil }
            throw error as? SpeechError ?? .permissionDenied(.systemAudio)
        }
        continuation.onTermination = { [weak self] _ in Task { await self?.stop() } }
        return stream
    }

    func stop() async {
        tap.stop()
        let continuation = lock.withLock { () -> AsyncThrowingStream<AudioChunk, Error>.Continuation? in
            defer { self.continuation = nil }
            return self.continuation
        }
        continuation?.finish()
    }
}

/// One process tap and its private aggregate device. The IO block runs on
/// its own serial queue.
struct ProcessTap {
    private(set) var tapID = AudioObjectID(kAudioObjectUnknown)
    private(set) var aggregateID = AudioObjectID(kAudioObjectUnknown)
    private var procID: AudioDeviceIOProcID?
    private let queue = DispatchQueue(label: "ai.mference.speech.call-tap", qos: .userInitiated)

    /// `processes` empty taps every process but this app.
    mutating func start(processes: [AudioObjectID], onBuffer: @escaping (AVAudioPCMBuffer) -> Void) throws {
        let description: CATapDescription
        if processes.isEmpty {
            let own = CoreAudioDevices.processObject(pid: ProcessInfo.processInfo.processIdentifier)
            description = CATapDescription(stereoGlobalTapButExcludeProcesses: own.map { [$0] } ?? [])
        } else {
            description = CATapDescription(stereoMixdownOfProcesses: processes)
        }
        description.uuid = UUID()
        description.isPrivate = true
        description.muteBehavior = .unmuted
        description.name = "Mference call audio"

        var tap = AudioObjectID(kAudioObjectUnknown)
        try check(AudioHardwareCreateProcessTap(description, &tap), "create the process tap")
        tapID = tap

        guard let outputUID = CoreAudioDevices.defaultOutputUID() else {
            throw SpeechError.failed("no output device")
        }
        let aggregate: [String: Any] = [
            kAudioAggregateDeviceNameKey: "Mference call audio",
            kAudioAggregateDeviceUIDKey: UUID().uuidString,
            kAudioAggregateDeviceMainSubDeviceKey: outputUID,
            kAudioAggregateDeviceIsPrivateKey: true,
            kAudioAggregateDeviceIsStackedKey: false,
            kAudioAggregateDeviceTapAutoStartKey: true,
            kAudioAggregateDeviceSubDeviceListKey: [[kAudioSubDeviceUIDKey: outputUID]],
            kAudioAggregateDeviceTapListKey: [[kAudioSubTapDriftCompensationKey: true,
                                               kAudioSubTapUIDKey: description.uuid.uuidString]],
        ]
        var device = AudioObjectID(kAudioObjectUnknown)
        try check(AudioHardwareCreateAggregateDevice(aggregate as CFDictionary, &device), "create the aggregate device")
        aggregateID = device

        var streamFormat = AudioStreamBasicDescription()
        var size = UInt32(MemoryLayout<AudioStreamBasicDescription>.size)
        var address = AudioObjectPropertyAddress(mSelector: kAudioTapPropertyFormat, mScope: kAudioObjectPropertyScopeGlobal,
                                                 mElement: kAudioObjectPropertyElementMain)
        try check(AudioObjectGetPropertyData(tap, &address, 0, nil, &size, &streamFormat), "read the tap format")
        guard let format = AVAudioFormat(streamDescription: &streamFormat) else {
            throw SpeechError.failed("the tap's format is not readable")
        }
        var proc: AudioDeviceIOProcID?
        try check(AudioDeviceCreateIOProcIDWithBlock(&proc, device, queue) { _, input, _, _, _ in
            guard let buffer = AVAudioPCMBuffer(pcmFormat: format, bufferListNoCopy: input, deallocator: nil) else { return }
            onBuffer(buffer)
        }, "create the IO proc")
        procID = proc
        try check(AudioDeviceStart(device, proc), "start the tap")
    }

    mutating func stop() {
        if aggregateID != kAudioObjectUnknown {
            if let procID {
                AudioDeviceStop(aggregateID, procID)
                AudioDeviceDestroyIOProcID(aggregateID, procID)
            }
            AudioHardwareDestroyAggregateDevice(aggregateID)
        }
        if tapID != kAudioObjectUnknown { AudioHardwareDestroyProcessTap(tapID) }
        procID = nil
        aggregateID = AudioObjectID(kAudioObjectUnknown)
        tapID = AudioObjectID(kAudioObjectUnknown)
    }

    private func check(_ status: OSStatus, _ step: String) throws {
        guard status == noErr else {
            // A refused tap reads as call audio being off (spec 4.3).
            throw SpeechError.permissionDenied(.systemAudio)
        }
    }
}

// MARK: Devices and processes

enum CoreAudioDevices {
    static func property<T>(_ object: AudioObjectID, _ selector: AudioObjectPropertySelector,
                            scope: AudioObjectPropertyScope = kAudioObjectPropertyScopeGlobal, default value: T) -> T {
        var address = AudioObjectPropertyAddress(mSelector: selector, mScope: scope, mElement: kAudioObjectPropertyElementMain)
        var result = value
        var size = UInt32(MemoryLayout<T>.size)
        let status = withUnsafeMutablePointer(to: &result) { AudioObjectGetPropertyData(object, &address, 0, nil, &size, $0) }
        return status == noErr ? result : value
    }

    static func objects(_ object: AudioObjectID, _ selector: AudioObjectPropertySelector) -> [AudioObjectID] {
        var address = AudioObjectPropertyAddress(mSelector: selector, mScope: kAudioObjectPropertyScopeGlobal,
                                                 mElement: kAudioObjectPropertyElementMain)
        var size: UInt32 = 0
        guard AudioObjectGetPropertyDataSize(object, &address, 0, nil, &size) == noErr, size > 0 else { return [] }
        var ids = [AudioObjectID](repeating: 0, count: Int(size) / MemoryLayout<AudioObjectID>.size)
        guard AudioObjectGetPropertyData(object, &address, 0, nil, &size, &ids) == noErr else { return [] }
        return ids
    }

    static func string(_ object: AudioObjectID, _ selector: AudioObjectPropertySelector) -> String? {
        var address = AudioObjectPropertyAddress(mSelector: selector, mScope: kAudioObjectPropertyScopeGlobal,
                                                 mElement: kAudioObjectPropertyElementMain)
        var value: Unmanaged<CFString>?
        var size = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
        guard AudioObjectGetPropertyData(object, &address, 0, nil, &size, &value) == noErr else { return nil }
        return value?.takeRetainedValue() as String?
    }

    static func inputChannels(_ device: AudioObjectID) -> Int {
        var address = AudioObjectPropertyAddress(mSelector: kAudioDevicePropertyStreamConfiguration,
                                                 mScope: kAudioDevicePropertyScopeInput, mElement: kAudioObjectPropertyElementMain)
        var size: UInt32 = 0
        guard AudioObjectGetPropertyDataSize(device, &address, 0, nil, &size) == noErr, size > 0 else { return 0 }
        let raw = UnsafeMutableRawPointer.allocate(byteCount: Int(size), alignment: MemoryLayout<AudioBufferList>.alignment)
        defer { raw.deallocate() }
        guard AudioObjectGetPropertyData(device, &address, 0, nil, &size, raw) == noErr else { return 0 }
        let list = UnsafeMutableAudioBufferListPointer(raw.assumingMemoryBound(to: AudioBufferList.self))
        return list.reduce(0) { $0 + Int($1.mNumberChannels) }
    }

    static func inputDevices() -> [AudioInputDevice] {
        let system = AudioObjectID(kAudioObjectSystemObject)
        let defaultInput: AudioObjectID = property(system, kAudioHardwarePropertyDefaultInputDevice, default: 0)
        return objects(system, kAudioHardwarePropertyDevices).compactMap { device in
            guard inputChannels(device) > 0, let uid = string(device, kAudioDevicePropertyDeviceUID),
                  let name = string(device, kAudioObjectPropertyName) else { return nil }
            return AudioInputDevice(id: uid, name: name, isDefault: device == defaultInput)
        }
    }

    static func deviceID(forUID uid: String) -> AudioDeviceID? {
        objects(AudioObjectID(kAudioObjectSystemObject), kAudioHardwarePropertyDevices)
            .first { string($0, kAudioDevicePropertyDeviceUID) == uid }
    }

    static func defaultOutputUID() -> String? {
        let device: AudioObjectID = property(AudioObjectID(kAudioObjectSystemObject),
                                             kAudioHardwarePropertyDefaultSystemOutputDevice, default: 0)
        return device == 0 ? nil : string(device, kAudioDevicePropertyDeviceUID)
    }

    static func processObject(pid: pid_t) -> AudioObjectID? {
        var address = AudioObjectPropertyAddress(mSelector: kAudioHardwarePropertyTranslatePIDToProcessObject,
                                                 mScope: kAudioObjectPropertyScopeGlobal, mElement: kAudioObjectPropertyElementMain)
        var pid = pid
        var object = AudioObjectID(kAudioObjectUnknown)
        var size = UInt32(MemoryLayout<AudioObjectID>.size)
        let status = AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &address,
                                                UInt32(MemoryLayout<pid_t>.size), &pid, &size, &object)
        return status == noErr && object != kAudioObjectUnknown ? object : nil
    }

    /// Audio process objects of an app and its helpers (`<bundle-id>.*`).
    static func processObjects(bundleID: String) -> [AudioObjectID] {
        objects(AudioObjectID(kAudioObjectSystemObject), kAudioHardwarePropertyProcessObjectList).filter { process in
            guard let id = string(process, kAudioProcessPropertyBundleID) else { return false }
            return id == bundleID || id.hasPrefix(bundleID + ".")
        }
    }

    /// Apps with an audio process (playing or able to), by their main bundle id.
    @MainActor
    static func runningAudioAppsOnMain() -> [RunningAudioApp] {
        let own = Bundle.main.bundleIdentifier
        let audioBundles = Set(objects(AudioObjectID(kAudioObjectSystemObject), kAudioHardwarePropertyProcessObjectList)
            .compactMap { string($0, kAudioProcessPropertyBundleID) })
        var apps: [String: RunningAudioApp] = [:]
        for app in NSWorkspace.shared.runningApplications where app.activationPolicy == .regular {
            guard let id = app.bundleIdentifier, id != own,
                  audioBundles.contains(where: { $0 == id || $0.hasPrefix(id + ".") }) else { continue }
            apps[id] = RunningAudioApp(bundleID: id, name: app.localizedName ?? id)
        }
        return apps.values.sorted { $0.name.localizedCaseInsensitiveCompare($1.name) == .orderedAscending }
    }

    static func runningAudioApps() async -> [RunningAudioApp] {
        await MainActor.run { runningAudioAppsOnMain() }
    }
}

// MARK: Sleep and wake

/// `NSWorkspace` sleep and wake notifications.
public struct WorkspacePowerEvents: PowerEventSource {
    public init() {}

    public func events() -> AsyncStream<PowerEvent> {
        let (stream, continuation) = AsyncStream.makeStream(of: PowerEvent.self)
        let center = NSWorkspace.shared.notificationCenter
        let sleep = center.addObserver(forName: NSWorkspace.willSleepNotification, object: nil, queue: nil) { _ in
            continuation.yield(.willSleep)
        }
        let wake = center.addObserver(forName: NSWorkspace.didWakeNotification, object: nil, queue: nil) { _ in
            continuation.yield(.didWake)
        }
        let observers = UncheckedBox([sleep, wake])
        continuation.onTermination = { _ in
            for observer in observers.value { center.removeObserver(observer) }
        }
        return stream
    }
}

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
    /// Told once per tap when it first hears other apps (call audio works).
    let onCallAudioHeard: @Sendable () -> Void

    init(onCallAudioHeard: @escaping @Sendable () -> Void = {}) {
        self.onCallAudioHeard = onCallAudioHeard
    }

    public func microphone(deviceID: String?) throws -> any AudioSource {
        MicrophoneSource(deviceUID: deviceID)
    }

    public func call(_ source: CallAudioSource) throws -> any AudioSource {
        switch source {
        case .off: throw SpeechError.failed("call audio is off")
        case .allApps: ProcessTapSource(target: .allApps, onHeard: onCallAudioHeard)
        case .app(let bundleID, _): ProcessTapSource(target: .app(bundleID), onHeard: onCallAudioHeard)
        }
    }

    public func inputDevices() async -> [AudioInputDevice] { CoreAudioDevices.inputDevices() }
    public func runningAudioApps() async -> [RunningAudioApp] { await CoreAudioDevices.runningAudioApps() }
    public func isPlayingOutput(_ source: CallAudioSource) async -> Bool? { CoreAudioDevices.isPlayingOutput(source) }
}

// MARK: Microphone

/// The microphone through AVAudioEngine's input tap. A device or format
/// change (headphones plugged in) restarts the tap with the new format.
///
/// Locking: the tap block reads only `continuation`, under `lock`. Starting,
/// restarting (the configuration-change notification) and stopping the
/// engine all run on `control`, one at a time, so a device change cannot
/// restart capture after `stop()`.
final class MicrophoneSource: AudioSource, @unchecked Sendable {
    let track = TrackKind.microphone
    private let deviceUID: String?
    private let lock = NSLock()
    private var continuation: AsyncThrowingStream<AudioChunk, Error>.Continuation?
    /// Only touched on `control`; nil once stopped.
    private var engine: AVAudioEngine?
    private var observer: NSObjectProtocol?
    private let control = DispatchQueue(label: "ai.mference.speech.microphone-control")

    init(deviceUID: String?) { self.deviceUID = deviceUID }

    func start() async throws -> AsyncThrowingStream<AudioChunk, Error> {
        guard AVCaptureDevice.authorizationStatus(for: .audio) == .authorized else {
            throw SpeechError.permissionDenied(.microphone)
        }
        let (stream, continuation) = AsyncThrowingStream.makeStream(of: AudioChunk.self)
        lock.withLock { self.continuation = continuation }
        do {
            try await onControl {
                let engine = AVAudioEngine()
                self.engine = engine
                try self.configure(engine)
                self.observer = NotificationCenter.default.addObserver(
                    forName: .AVAudioEngineConfigurationChange, object: engine, queue: nil
                ) { [weak source = self, control = self.control] _ in
                    guard let source else { return }
                    control.async { source.restart() }
                }
            }
        } catch {
            await stop()
            throw error as? SpeechError ?? .failed(error.localizedDescription)
        }
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

    /// On `control`. Does nothing once stopped.
    private func restart() {
        guard let engine else { return }
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
        do {
            try configure(engine)
        } catch {
            lock.withLock { self.continuation }?.finish(throwing: error)
        }
    }

    func stop() async {
        try? await onControl {
            if let observer = self.observer { NotificationCenter.default.removeObserver(observer) }
            self.observer = nil
            self.engine?.inputNode.removeTap(onBus: 0)
            self.engine?.stop()
            self.engine = nil
        }
        let continuation = lock.withLock { () -> AsyncThrowingStream<AudioChunk, Error>.Continuation? in
            defer { self.continuation = nil }
            return self.continuation
        }
        continuation?.finish()
    }

    private func onControl<T: Sendable>(_ body: @escaping @Sendable () throws -> T) async throws -> T {
        try await withCheckedThrowingContinuation { continuation in
            control.async {
                do { continuation.resume(returning: try body()) } catch { continuation.resume(throwing: error) }
            }
        }
    }
}

// MARK: Call audio

/// Other apps' output through a Core Audio process tap: every app but this
/// one (the safe default, spec 12), or one app with its helper processes
/// (browsers and call apps play through helpers). Core Audio errors end the
/// stream as failures (`SpeechError.failed`); nothing here decides or
/// records a permission. A denied tap does not fail, it delivers digital
/// silence, and silence is never read as a denial (an in-person meeting is
/// silent too): `onHeard` runs once, when the tap first hears a non-zero
/// sample. When the default output device changes or goes away (Bluetooth
/// headphones), the tap is rebuilt on the new one; if that fails the stream
/// ends with the error.
///
/// Locking: the IO block touches only `io`, which is never held while a
/// tap stops, so stopping (which waits for an IO cycle in flight) cannot
/// deadlock with it. Starting, rebuilding and stopping the tap all run on
/// `control`, one at a time, so a device change cannot race `stop()`.
final class ProcessTapSource: AudioSource, @unchecked Sendable {
    enum Target: Sendable { case allApps, app(String) }

    let track = TrackKind.call
    private let target: Target
    private let onHeard: @Sendable () -> Void
    private struct IOState {
        var continuation: AsyncThrowingStream<AudioChunk, Error>.Continuation?
        var heard = false
    }
    private let io = Mutex(IOState())
    /// Only touched on `control`.
    private var tap = ProcessTap()
    private var stopped = false
    private var listener: AudioObjectPropertyListenerBlock?
    private let control = DispatchQueue(label: "ai.mference.speech.call-control")

    init(target: Target, onHeard: @escaping @Sendable () -> Void = {}) {
        self.target = target
        self.onHeard = onHeard
    }

    func start() async throws -> AsyncThrowingStream<AudioChunk, Error> {
        let (stream, continuation) = AsyncThrowingStream.makeStream(of: AudioChunk.self)
        io.withLock { $0.continuation = continuation }
        do {
            try await onControl {
                self.stopped = false
                try self.startTap()
                self.listenForOutputChanges()
            }
        } catch {
            io.withLock { $0.continuation = nil }
            throw error as? SpeechError ?? .failed(error.localizedDescription)
        }
        continuation.onTermination = { [weak self] _ in Task { await self?.stop() } }
        return stream
    }

    /// Runs `body` on `control` and waits for it.
    private func onControl<T: Sendable>(_ body: @escaping @Sendable () throws -> T) async throws -> T {
        try await withCheckedThrowingContinuation { continuation in
            control.async {
                do { continuation.resume(returning: try body()) } catch { continuation.resume(throwing: error) }
            }
        }
    }

    private func processes() throws -> [AudioObjectID] {
        switch target {
        case .allApps: return []
        case .app(let bundleID):
            let processes = CoreAudioDevices.processObjects(bundleID: bundleID)
            if processes.isEmpty { throw SpeechError.failed("\(bundleID) has no audio process") }
            return processes
        }
    }

    /// On `control`. A tap that fails partway is torn down before the error.
    private func startTap() throws {
        let processes = try processes()
        var resampler: AudioResampler?
        do {
            try tap.start(processes: processes) { [weak self] buffer in
                guard let self else { return }
                if resampler == nil { resampler = try? AudioResampler(inputFormat: buffer.format) }
                self.noteSound(buffer)
                guard let samples = try? resampler?.convert(buffer), !samples.isEmpty else { return }
                self.io.withLock { $0.continuation }?.yield(AudioChunk(samples: samples))
            }
        } catch {
            tap.stop()
            throw error
        }
    }

    /// On the tap's IO queue: the first non-zero sample tells `onHeard`, once.
    private func noteSound(_ buffer: AVAudioPCMBuffer) {
        guard !io.withLock({ $0.heard }), Self.hasSound(buffer) else { return }
        let first = io.withLock { state -> Bool in
            defer { state.heard = true }
            return !state.heard
        }
        if first { control.async { [onHeard] in onHeard() } }
    }

    /// Whether any sample in any channel is non-zero (a denied tap delivers
    /// exact zeros).
    static func hasSound(_ buffer: AVAudioPCMBuffer) -> Bool {
        guard let channels = buffer.floatChannelData else { return false }
        let frames = Int(buffer.frameLength)
        for channel in 0..<Int(buffer.format.channelCount) {
            if UnsafeBufferPointer(start: channels[channel], count: frames).contains(where: { $0 != 0 }) { return true }
        }
        return false
    }

    private static var outputDeviceAddress: AudioObjectPropertyAddress {
        AudioObjectPropertyAddress(mSelector: kAudioHardwarePropertyDefaultOutputDevice,
                                   mScope: kAudioObjectPropertyScopeGlobal,
                                   mElement: kAudioObjectPropertyElementMain)
    }

    /// On `control`: the default output device (not the alert device).
    private func listenForOutputChanges() {
        let block: AudioObjectPropertyListenerBlock = { [weak self] _, _ in self?.outputChanged() }
        var address = Self.outputDeviceAddress
        AudioObjectAddPropertyListenerBlock(AudioObjectID(kAudioObjectSystemObject), &address, control, block)
        listener = block
    }

    /// On `control`: the default output changed or went away, so the tap
    /// is rebuilt on the new one. After `stop()` it does nothing.
    private func outputChanged() {
        guard !stopped else { return }
        tap.stop()
        do {
            try startTap()
        } catch {
            finish(throwing: error as? SpeechError ?? .failed("the output device changed"))
        }
    }

    private func finish(throwing error: Error?) {
        let continuation = io.withLock { state -> AsyncThrowingStream<AudioChunk, Error>.Continuation? in
            defer { state.continuation = nil }
            return state.continuation
        }
        guard let continuation else { return }
        if let error { continuation.finish(throwing: error) } else { continuation.finish() }
    }

    func stop() async {
        let listener = try? await onControl { () -> UncheckedBox<AudioObjectPropertyListenerBlock?> in
            self.stopped = true
            self.tap.stop()
            defer { self.listener = nil }
            return UncheckedBox(self.listener)
        }
        // Removed off `control`, where a late callback may be waiting; it
        // finds `stopped` and returns.
        if let block = listener?.value {
            var address = Self.outputDeviceAddress
            AudioObjectRemovePropertyListenerBlock(AudioObjectID(kAudioObjectSystemObject), &address, control, block)
        }
        finish(throwing: nil)
    }
}

/// One process tap and its private aggregate device. The IO block runs on
/// its own serial queue; `stop()` waits for a cycle in flight, so the
/// caller must not hold a lock that the IO block takes.
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
            throw SpeechError.failed("call audio: no output device")
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
            throw SpeechError.failed("call audio: the tap's format is not readable")
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

    /// A Core Audio error is a failure, never a permission answer.
    private func check(_ status: OSStatus, _ step: String) throws {
        guard status == noErr else { throw SpeechError.failed("call audio: cannot \(step) (\(status))") }
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
                                             kAudioHardwarePropertyDefaultOutputDevice, default: 0)
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

    /// Whether the audio processes `source` covers are playing sound now
    /// (`kAudioProcessPropertyIsRunningOutput`): every process but this
    /// one for `.allApps`, the app and its helpers for `.app`. An app with no
    /// audio process is not playing; nil when no process could be read.
    static func isPlayingOutput(_ source: CallAudioSource) -> Bool? {
        let processes: [AudioObjectID]
        switch source {
        case .off: return nil
        case .allApps:
            let own = getpid()
            processes = objects(AudioObjectID(kAudioObjectSystemObject), kAudioHardwarePropertyProcessObjectList)
                .filter { property($0, kAudioProcessPropertyPID, default: own) != own }
        case .app(let bundleID, _):
            processes = processObjects(bundleID: bundleID)
            if processes.isEmpty { return false }
        }
        var readable = false
        for process in processes {
            var address = AudioObjectPropertyAddress(mSelector: kAudioProcessPropertyIsRunningOutput,
                                                     mScope: kAudioObjectPropertyScopeGlobal,
                                                     mElement: kAudioObjectPropertyElementMain)
            var running: UInt32 = 0
            var size = UInt32(MemoryLayout<UInt32>.size)
            guard AudioObjectGetPropertyData(process, &address, 0, nil, &size, &running) == noErr else { continue }
            if running != 0 { return true }
            readable = true
        }
        return readable ? false : nil
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

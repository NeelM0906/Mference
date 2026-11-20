@preconcurrency import AVFoundation
import Foundation
@testable import MferenceAppCore
import Synchronization
import Testing

/// The live `SpeechService` over scripted stages (Notes spec 6.2, 6.3, 9):
/// fixture PCM for the microphone and call tracks, tones as speech and
/// zeros as silence, scripted text and speaker turns. No model, microphone,
/// tap, TCC prompt or network.
@Suite(.serialized) struct SpeechServiceTests {
    // MARK: Fixtures

    /// Recognises a phrase by its loudest sample: each fixture voice is a
    /// tone at its own amplitude.
    final class AmplitudeTranscriber: SpeechRecognizing {
        let voices: [(amplitude: Float, text: String)]
        let delay: Duration?
        init(_ voices: [(Float, String)], delay: Duration? = nil) {
            self.voices = voices.map { (amplitude: $0.0, text: $0.1) }
            self.delay = delay
        }
        func transcribe(_ samples: [Float]) async throws -> RecognizedPhrase {
            if let delay { try await Task.sleep(for: delay) }
            let peak = samples.map(abs).max() ?? 0
            let voice = voices.min { abs($0.amplitude - peak) < abs($1.amplitude - peak) }
            return RecognizedPhrase(text: voice?.text ?? "", languageCode: "en")
        }
    }

    /// PCM from (seconds, amplitude) parts, as 100 ms capture chunks.
    static func chunks(_ parts: [(TimeInterval, Float)]) -> [AudioChunk] {
        let samples = parts.flatMap { FakeAudioSource.tone(seconds: $0.0, amplitude: $0.1) }
        return stride(from: 0, to: samples.count, by: 1_600).map {
            AudioChunk(samples: Array(samples[$0..<min(samples.count, $0 + 1_600)]))
        }
    }

    /// Collects one single-consumer stream and waits for what tests expect.
    actor EventLog<Event: Sendable> {
        private(set) var events: [Event] = []
        private var finished = false
        private var task: Task<Void, Never>?

        init(_ stream: AsyncStream<Event>) {
            task = nil
            Task { await self.consume(stream) }
        }

        private func consume(_ stream: AsyncStream<Event>) async {
            for await event in stream { events.append(event) }
            finished = true
        }

        var isFinished: Bool { finished }

        /// Waits until `condition` holds for the events so far (5 s at most).
        func wait(_ condition: ([Event]) -> Bool) async -> Bool {
            let deadline = ContinuousClock.now + .seconds(5)
            while !condition(events) {
                if ContinuousClock.now > deadline { return false }
                try? await Task.sleep(for: .milliseconds(5))
            }
            return true
        }

        func waitUntilFinished() async -> Bool {
            let deadline = ContinuousClock.now + .seconds(5)
            while !finished {
                if ContinuousClock.now > deadline { return false }
                try? await Task.sleep(for: .milliseconds(5))
            }
            return true
        }
    }

    final class FailureLog: Sendable {
        private let log = Mutex<[SpeechError]>([])
        var errors: [SpeechError] { log.withLock { $0 } }
        func append(_ error: SpeechError) { log.withLock { $0.append(error) } }
    }

    struct Rig {
        let service: SpeechService
        let loader: ScriptedModelLoader
        let power: FakePowerEvents
        let root: URL
        let recordings: RecordingStore
        let callFailures: FailureLog

        var installRoot: URL { root.appending(path: "Speech") }
    }

    /// A service with `installed` packages on disk (one tiny file each and
    /// the installer's marker).
    static func rig(installed: [SpeechPackageID] = [.speech], sources: any AudioSourceFactory,
                    recognizer: any SpeechRecognizing = ScriptedTranscriber(),
                    diarizer: any SpeakerDiarizing = ScriptedDiarizer(live: [], final: []),
                    idleUnload: Duration? = nil, callAudioAllowed: Bool = true) throws -> Rig {
        let root = FileManager.default.temporaryDirectory.appending(path: "mference-speech-\(UUID().uuidString)")
        let speechRoot = root.appending(path: "Speech")
        var packages: [SpeechPackageID: SpeechModelPackage] = [:]
        for id in SpeechPackageID.allCases {
            let file = SpeechModelFile(repoPath: "m.bin", installPath: "\(id.rawValue)/rev/m.bin", size: 4,
                                       sha256: String(repeating: "0", count: 64))
            packages[id] = SpeechModelPackage(id: id, sources: [
                SpeechModelSource(repo: "test/\(id.rawValue)", revision: "rev", license: "MIT", files: [file])])
            guard installed.contains(id) else { continue }
            let url = speechRoot.appending(path: file.installPath)
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try Data([1, 2, 3, 4]).write(to: url)
            let marker = speechRoot.appending(path: ".installed/\(id.rawValue).json")
            try FileManager.default.createDirectory(at: marker.deletingLastPathComponent(), withIntermediateDirectories: true)
            try Data("{}".utf8).write(to: marker)
        }
        let installer = SpeechModelInstaller(root: speechRoot, packages: packages, allowedHosts: [])
        let loader = ScriptedModelLoader(recognizer: recognizer, voiceActivity: ScriptedVoiceActivity(), diarizer: diarizer)
        let recordings = RecordingStore(root: root.appending(path: "Recordings"), availableBytes: { _ in 100 << 30 })
        let power = FakePowerEvents()
        let failures = FailureLog()
        let service = SpeechService(.init(installer: installer, loader: loader, sources: sources,
                                          recordings: recordings, power: power, idleUnload: idleUnload,
                                          onCallAudioFailure: { error in failures.append(error) },
                                          callAudioAllowed: { callAudioAllowed }))
        return Rig(service: service, loader: loader, power: power, root: root, recordings: recordings,
                   callFailures: failures)
    }

    static func finals(_ events: [MeetingEvent]) -> [TranscriptSegment] {
        events.compactMap { if case .segment(let s) = $0, s.isFinal { s } else { nil } }
    }

    static func result(_ events: [MeetingEvent]) -> TranscriptResult? {
        events.compactMap { if case .ended(let r) = $0 { r } else { nil } }.first
    }

    // MARK: Dictation

    @Test func dictationInsertsCleanPhrasesInOrder() async throws {
        let mic = FakeAudioSource(track: .microphone, chunks: Self.chunks([(0.3, 0), (1, 0.2), (1, 0), (0.8, 0.4), (1, 0)]))
        let rig = try Self.rig(sources: FakeAudioSourceFactory(microphone: mic),
                               recognizer: AmplitudeTranscriber([(0.2, "Um, I I think so."), (0.4, "Uh.")]))
        defer { try? FileManager.default.removeItem(at: rig.root) }
        #expect(rig.service.status.speech == .ready)
        let session = try await rig.service.startDictation(DictationOptions(language: .fixed("en")))
        #expect(rig.service.status.liveSession?.kind == .dictation)
        let log = EventLog(session.events)
        #expect(await log.wait { $0.contains { if case .phraseDropped = $0 { true } else { false } } })
        await session.stop()
        #expect(await log.waitUntilFinished())
        let events = await log.events
        let started = events.compactMap { if case .phraseStarted(let id) = $0 { id } else { nil } }
        let finished = events.compactMap { if case .phraseFinished(let id, let text, let language) = $0 { (id, text, language) } else { nil } }
        let dropped = events.compactMap { if case .phraseDropped(let id) = $0 { id } else { nil } }
        #expect(started.count == 2)
        #expect(finished.map(\.0) == [started[0]] && dropped == [started[1]])
        #expect(finished.first?.1 == "I think so.", "fillers and the repeated word are gone")
        #expect(finished.first?.2 == "en")
        #expect(events.last == .ended)
        // Each phrase is announced before its text.
        #expect(events.firstIndex(of: .phraseStarted(started[0]))! < events.firstIndex { if case .phraseFinished = $0 { true } else { false } }!)
        #expect(events.contains { if case .level(let level) = $0 { level > 0 } else { false } })
        #expect(rig.service.status.liveSession == nil)
        #expect(mic.isStopped)
    }

    @Test func oneLiveSessionAtATime() async throws {
        let mic = FakeAudioSource(track: .microphone, chunks: [])
        let rig = try Self.rig(sources: FakeAudioSourceFactory(microphone: mic))
        defer { try? FileManager.default.removeItem(at: rig.root) }
        let dictation = try await rig.service.startDictation(DictationOptions())
        await #expect(throws: SpeechError.busy(.dictation)) {
            _ = try await rig.service.startMeeting(MeetingOptions(noteID: UUID(), callAudio: .off))
        }
        await #expect(throws: SpeechError.busy(.dictation)) { _ = try await rig.service.startDictation(DictationOptions()) }
        await dictation.stop()
        await dictation.stop()  // idempotent
        let meeting = try await rig.service.startMeeting(MeetingOptions(noteID: UUID(), callAudio: .off))
        #expect(rig.service.status.liveSession?.kind == .meeting)
        await meeting.stop()
        let log = EventLog(meeting.events)
        #expect(await log.waitUntilFinished())
        #expect(rig.service.status.liveSession == nil)
    }

    @Test func nothingStartsBeforeTheModelIsInstalled() async throws {
        let rig = try Self.rig(installed: [], sources: FakeAudioSourceFactory(microphone: FakeAudioSource(track: .microphone, chunks: [])))
        defer { try? FileManager.default.removeItem(at: rig.root) }
        #expect(rig.service.status.speech == .notInstalled)
        await #expect(throws: SpeechError.notInstalled(.speech)) { _ = try await rig.service.startDictation(DictationOptions()) }
        await #expect(throws: SpeechError.notInstalled(.speech)) {
            _ = try await rig.service.startMeeting(MeetingOptions(noteID: UUID()))
        }
        await #expect(throws: SpeechError.notInstalled(.speech)) {
            _ = try await rig.service.transcribeFile(URL(fileURLWithPath: "/tmp/x.m4a"), options: .init(noteID: UUID()))
        }
        #expect(rig.loader.speechLoads == 0)
    }

    // MARK: Meetings

    @Test func meetingSegmentsFromTwoTracksInTimeOrderWithoutEchoes() async throws {
        // Mic: "Can you hear me?" at 0.5 s, then the call's words echoed back at 2 s.
        let mic = FakeAudioSource(track: .microphone, chunks: Self.chunks([(0.5, 0), (1, 0.2), (0.5, 0), (2.2, 0.3), (0.8, 0)]))
        // Call: "Ship it on Friday." at 2 s.
        let call = FakeAudioSource(track: .call, chunks: Self.chunks([(2, 0), (2.2, 0.4), (0.8, 0)]))
        let rig = try Self.rig(sources: FakeAudioSourceFactory(microphone: mic, call: call),
                               recognizer: AmplitudeTranscriber([(0.2, "Can you hear me?"), (0.3, "ship it on friday"),
                                                                 (0.4, "Ship it on Friday.")]))
        defer { try? FileManager.default.removeItem(at: rig.root) }
        let note = UUID()
        let meeting = try await rig.service.startMeeting(MeetingOptions(noteID: note, callAudio: .allApps, speakerLabels: false))
        let log = EventLog(meeting.events)
        #expect(await log.wait { events in
            Self.finals(events).count >= 2
                && events.filter { if case .segmentRemoved = $0 { true } else { false } }.count >= 1
        })
        await meeting.stop()
        #expect(await log.waitUntilFinished())
        let events = await log.events
        guard case .started(callAudio: true) = events.first else {
            Issue.record("first event is .started(callAudio: true)")
            return
        }
        let result = try #require(Self.result(events))
        #expect(result.segments.map(\.text) == ["Can you hear me?", "Ship it on Friday."])
        #expect(result.segments.map(\.track) == [.microphone, .call])
        #expect(result.segments.map(\.speakerID) == [SpeakerInfo.youID, SpeakerInfo.othersID])
        #expect(result.speakers.map(\.name) == ["You", "Others"])
        #expect(result.segments.map(\.start) == result.segments.map(\.start).sorted())
        #expect(abs(result.duration - 5) < 0.01)
        #expect(Set(result.tracks.map(\.fileName)) == ["microphone-1.m4a", "call-1.m4a"])
        #expect(result.audioDirectory == rig.recordings.directory(for: note))
        #expect(events.contains { if case .finishing(.transcribingLastPhrases) = $0 { true } else { false } })
        // Each live segment was announced before its text.
        for segment in result.segments {
            #expect(events.contains { if case .segment(let s) = $0 { s.id == segment.id && !s.isFinal } else { false } })
        }
    }

    @Test func deniedCallAudioRecordsTheMicrophoneOnly() async throws {
        let mic = FakeAudioSource(track: .microphone, chunks: Self.chunks([(1, 0.2), (1, 0)]))
        let rig = try Self.rig(sources: FakeAudioSourceFactory(microphone: mic, call: nil),
                               recognizer: AmplitudeTranscriber([(0.2, "Just me.")]))
        defer { try? FileManager.default.removeItem(at: rig.root) }
        let meeting = try await rig.service.startMeeting(MeetingOptions(noteID: UUID(), callAudio: .allApps))
        let log = EventLog(meeting.events)
        #expect(await log.wait { Self.finals($0).count == 1 })
        await meeting.stop()
        #expect(await log.waitUntilFinished())
        let events = await log.events
        guard events.count > 2, case .started(callAudio: false) = events[0],
              case .callAudioUnavailable(.permissionDenied(.systemAudio)) = events[1] else {
            Issue.record("expected mic-only start, got \(events.prefix(3))")
            return
        }
        #expect(rig.callFailures.errors == [.permissionDenied(.systemAudio)])
        let result = try #require(Self.result(events))
        #expect(result.segments.map(\.text) == ["Just me."])
        #expect(result.tracks.map(\.kind) == [.microphone])
    }

    @Test func callAudioIsNotTappedWithoutPermission() async throws {
        // L6: the record popover asks first; the service never taps unasked.
        let mic = FakeAudioSource(track: .microphone, chunks: Self.chunks([(1, 0)]))
        let call = FakeAudioSource(track: .call, chunks: Self.chunks([(1, 0.4)]))
        let rig = try Self.rig(sources: FakeAudioSourceFactory(microphone: mic, call: call), callAudioAllowed: false)
        defer { try? FileManager.default.removeItem(at: rig.root) }
        let meeting = try await rig.service.startMeeting(MeetingOptions(noteID: UUID(), callAudio: .allApps))
        let log = EventLog(meeting.events)
        await meeting.stop()
        #expect(await log.waitUntilFinished())
        let events = await log.events
        #expect(events.prefix(2) == [.started(callAudio: false), .callAudioUnavailable(.permissionDenied(.systemAudio))])
        #expect(call.startCount == 0)
        #expect(rig.callFailures.errors.isEmpty, "not asked is not denied")
    }

    @Test func quitClosesTheMeetingWithoutTheSpeakerPassOrConversion() async throws {
        // Review #226 HIGH: shutdown must fit the quit deadline.
        let mic = FakeAudioSource(track: .microphone, chunks: Self.chunks([(1, 0.2), (1, 0)]))
        let call = FakeAudioSource(track: .call, chunks: Self.chunks([(1, 0.4), (1, 0)]))
        let diarizer = ScriptedDiarizer(live: [SpeakerTurn(speaker: "0", start: 0, end: 1.2)],
                                        final: [SpeakerTurn(speaker: "F", start: 0, end: 2)])
        let rig = try Self.rig(installed: [.speech, .speakerLabels],
                               sources: FakeAudioSourceFactory(microphone: mic, call: call),
                               recognizer: AmplitudeTranscriber([(0.2, "Me."), (0.4, "Them.")]), diarizer: diarizer)
        defer { try? FileManager.default.removeItem(at: rig.root) }
        let note = UUID()
        let meeting = try await rig.service.startMeeting(MeetingOptions(noteID: note, callAudio: .allApps))
        let log = EventLog(meeting.events)
        #expect(await log.wait { Self.finals($0).count == 2 })
        await rig.service.shutdown()
        #expect(await log.waitUntilFinished())
        let events = await log.events
        let result = try #require(Self.result(events))
        #expect(diarizer.finalPassFiles.isEmpty, "no speaker pass at quit")
        #expect(result.speakerMapping.isEmpty && result.segments.contains { $0.speakerID == "live-0" })
        #expect(Set(result.tracks.map(\.fileName)) == ["microphone-1.m4a", "call-1.m4a"])
        let folder = rig.recordings.directory(for: note)
        #expect(FileManager.default.fileExists(atPath: folder.appending(path: "call-1.caf").path), "converted later")
        #expect(await rig.service.recoverInterruptedRecordings().isEmpty)
        #expect(FileManager.default.fileExists(atPath: folder.appending(path: "call-1.m4a").path))
    }

    @Test func renamesCarryOverAfterTheFinalSpeakerPass() async throws {
        let mic = FakeAudioSource(track: .microphone, chunks: Self.chunks([(6, 0)]))
        // Two call voices: 0.5-2 s and 3-4.5 s.
        let call = FakeAudioSource(track: .call, chunks: Self.chunks([(0.5, 0), (1.5, 0.3), (1, 0), (1.5, 0.5), (1.5, 0)]))
        let diarizer = ScriptedDiarizer(
            live: [SpeakerTurn(speaker: "0", start: 0.4, end: 2.1), SpeakerTurn(speaker: "1", start: 2.9, end: 4.6)],
            final: [SpeakerTurn(speaker: "B", start: 0, end: 2.5), SpeakerTurn(speaker: "A", start: 2.5, end: 6)])
        let rig = try Self.rig(installed: [.speech, .speakerLabels],
                               sources: FakeAudioSourceFactory(microphone: mic, call: call),
                               recognizer: AmplitudeTranscriber([(0.3, "First voice."), (0.5, "Second voice.")]),
                               diarizer: diarizer)
        defer { try? FileManager.default.removeItem(at: rig.root) }
        let meeting = try await rig.service.startMeeting(MeetingOptions(noteID: UUID(), callAudio: .allApps, speakerLabels: true))
        let log = EventLog(meeting.events)
        #expect(await log.wait { Self.finals($0).count >= 2 })
        await meeting.stop()
        #expect(await log.waitUntilFinished())
        let events = await log.events
        let live = Self.finals(events)
        #expect(Set(live.map(\.speakerID)) == ["live-0", "live-1"])
        #expect(events.contains { if case .speakersUpdated(let s) = $0 { s.count == 3 } else { false } })
        let result = try #require(Self.result(events))
        #expect(result.speakerMapping == ["live-0": "speaker-B", "live-1": "speaker-A"])
        #expect(result.segments.map(\.speakerID) == ["speaker-B", "speaker-A"])
        #expect(Set(result.segments.map(\.id)) == Set(live.map(\.id)), "segment ids never change")
        #expect(result.speakers.map(\.name) == ["Speaker 2", "Speaker 3"])
        #expect(diarizer.finalPassFiles.count == 1)
        #expect(diarizer.finalPassFiles.first?.map(\.lastPathComponent) == ["call-1.caf"])
        #expect(events.contains { if case .finishing(.labellingSpeakers(let p)) = $0 { p == 1 } else { false } })
    }

    @Test func sleepClosesTheFilesAndWakeResumesInNewOnes() async throws {
        let mic = FakeAudioSource(track: .microphone, chunks: Self.chunks([(1, 0.2), (1, 0)]))
        let rig = try Self.rig(sources: FakeAudioSourceFactory(microphone: mic),
                               recognizer: AmplitudeTranscriber([(0.2, "Before and after.")]))
        defer { try? FileManager.default.removeItem(at: rig.root) }
        let note = UUID()
        let meeting = try await rig.service.startMeeting(MeetingOptions(noteID: note, callAudio: .off))
        let log = EventLog(meeting.events)
        #expect(await log.wait { Self.finals($0).count == 1 })
        #expect(await log.wait { _ in rig.power.listenerCount == 1 })
        rig.power.send(.willSleep)
        rig.power.send(.didWake)
        #expect(await log.wait { events in events.contains { if case .gap = $0 { true } else { false } } })
        #expect(await log.wait { Self.finals($0).count == 2 })
        await meeting.stop()
        #expect(await log.waitUntilFinished())
        let result = try #require(Self.result(await log.events))
        #expect(result.gaps.count == 1)
        #expect(abs((result.gaps.first?.at ?? 0) - 2) < 0.01)
        #expect(result.tracks.map(\.fileName) == ["microphone-1.m4a", "microphone-2.m4a"])
        #expect(result.tracks.map(\.startOffset) == [0, 2])
        #expect(result.segments.count == 2 && result.segments[1].start >= 2)
        #expect(mic.startCount == 2)
        #expect(abs(result.duration - 4) < 0.01)
    }

    @Test func pauseDropsAudioAndMarksStatus() async throws {
        let mic = FakeAudioSource(track: .microphone, chunks: Self.chunks([(1, 0)]))
        let rig = try Self.rig(sources: FakeAudioSourceFactory(microphone: mic))
        defer { try? FileManager.default.removeItem(at: rig.root) }
        let meeting = try await rig.service.startMeeting(MeetingOptions(noteID: UUID(), callAudio: .off))
        let log = EventLog(meeting.events)
        #expect(await log.wait { _ in true })
        try await Task.sleep(for: .milliseconds(50))
        let before = await meeting.markMoment()
        await meeting.pause()
        #expect(rig.service.status.liveSession?.isPaused == true)
        for chunk in Self.chunks([(2, 0.3)]) { mic.push(chunk) }
        try await Task.sleep(for: .milliseconds(50))
        #expect(await meeting.markMoment() == before, "paused audio is not recorded")
        await meeting.resume()
        #expect(rig.service.status.liveSession?.isPaused == false)
        await meeting.stop()
        #expect(await log.waitUntilFinished())
        let events = await log.events
        #expect(events.contains(.paused) && events.contains(.resumed))
    }

    // MARK: Files

    static func writeWAV(_ url: URL, parts: [(TimeInterval, Float)]) throws {
        let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16_000, channels: 1, interleaved: false)!
        let file = try AVAudioFile(forWriting: url, settings: [AVFormatIDKey: kAudioFormatLinearPCM,
                                                               AVSampleRateKey: 16_000, AVNumberOfChannelsKey: 1,
                                                               AVLinearPCMBitDepthKey: 16])
        let samples = parts.flatMap { FakeAudioSource.tone(seconds: $0.0, amplitude: $0.1) }
        let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count))!
        buffer.frameLength = AVAudioFrameCount(samples.count)
        for (index, sample) in samples.enumerated() { buffer.floatChannelData![0][index] = sample }
        try file.write(from: buffer)
    }

    @Test func aFileBecomesATranscriptWithItsAudioKept() async throws {
        let rig = try Self.rig(sources: FakeAudioSourceFactory(microphone: FakeAudioSource(track: .microphone, chunks: [])),
                               recognizer: AmplitudeTranscriber([(0.2, "First part."), (0.4, "Second part.")]))
        defer { try? FileManager.default.removeItem(at: rig.root) }
        let input = rig.root.appending(path: "interview.wav")
        try Self.writeWAV(input, parts: [(0.5, 0), (2, 0.2), (1, 0), (3, 0.4), (1, 0)])
        let note = UUID()
        let job = try await rig.service.transcribeFile(input, options: FileTranscriptionOptions(noteID: note))
        let log = EventLog(job.events)
        #expect(await log.waitUntilFinished())
        let events = await log.events
        guard case .queued(position: 0) = events.first else {
            Issue.record("first event is .queued(0): \(events.prefix(2))")
            return
        }
        #expect(events.contains { if case .started(let duration) = $0 { abs(duration - 7.5) < 0.05 } else { false } })
        let fractions = events.compactMap { if case .progress(let f, _, _) = $0 { f } else { nil } }
        #expect(fractions == fractions.sorted() && fractions.last == 1)
        guard case .finished(let result) = events.last else {
            Issue.record("last event is .finished: \(String(describing: events.last))")
            return
        }
        #expect(result.segments.map(\.text) == ["First part.", "Second part."])
        #expect(result.segments.allSatisfy { $0.track == .file })
        let copy = try #require(result.tracks.first?.fileName)
        #expect(result.tracks.count == 1 && copy.hasPrefix("import-") && copy.hasSuffix("-interview.wav"))
        #expect(FileManager.default.fileExists(atPath: rig.recordings.directory(for: note).appending(path: copy).path))
        #expect(rig.service.status.fileQueue.first?.phase == .done)
    }

    @Test func filesWaitWhileDictationRuns() async throws {
        let rig = try Self.rig(sources: FakeAudioSourceFactory(microphone: FakeAudioSource(track: .microphone, chunks: [])),
                               recognizer: AmplitudeTranscriber([(0.2, "Later.")]))
        defer { try? FileManager.default.removeItem(at: rig.root) }
        let input = rig.root.appending(path: "memo.wav")
        try Self.writeWAV(input, parts: [(1, 0.2), (1, 0)])
        let dictation = try await rig.service.startDictation(DictationOptions())
        let job = try await rig.service.transcribeFile(input, options: FileTranscriptionOptions(noteID: UUID(), keepAudio: false))
        let log = EventLog(job.events)
        #expect(await log.wait { $0.contains(.paused) })
        #expect(rig.service.status.fileQueue.first?.phase == .paused)
        try await Task.sleep(for: .milliseconds(100))
        #expect(!(await log.isFinished), "nothing transcribes while the microphone is live")
        await dictation.stop()
        #expect(await log.waitUntilFinished())
        guard case .finished(let result) = await log.events.last else {
            Issue.record("the job finishes once dictation ends")
            return
        }
        #expect(result.segments.map(\.text) == ["Later."])
        #expect(result.tracks.isEmpty && result.audioDirectory == nil, "Delete after transcribing")
    }

    @Test func cancelStopsAQueuedOrRunningFile() async throws {
        let rig = try Self.rig(sources: FakeAudioSourceFactory(microphone: FakeAudioSource(track: .microphone, chunks: [])),
                               recognizer: AmplitudeTranscriber([(0.2, "Slow.")], delay: .milliseconds(200)))
        defer { try? FileManager.default.removeItem(at: rig.root) }
        let input = rig.root.appending(path: "long.wav")
        try Self.writeWAV(input, parts: Array(repeating: [(1, 0.2), (1, 0)], count: 10).flatMap { $0 })
        let first = try await rig.service.transcribeFile(input, options: FileTranscriptionOptions(noteID: UUID()))
        let second = try await rig.service.transcribeFile(input, options: FileTranscriptionOptions(noteID: UUID()))
        let firstLog = EventLog(first.events)
        let secondLog = EventLog(second.events)
        #expect(await secondLog.wait { $0.contains(.queued(position: 1)) })
        await rig.service.cancel(second.id)
        #expect(await secondLog.waitUntilFinished())
        #expect(await secondLog.events.last == .cancelled)
        #expect(await firstLog.wait { $0.contains { if case .started = $0 { true } else { false } } })
        await rig.service.cancel(first.id)
        #expect(await firstLog.waitUntilFinished())
        #expect(await firstLog.events.last == .cancelled)
        #expect(rig.service.status.fileQueue.isEmpty)
    }

    @Test func twoImportsWithTheSameNameKeepTheirOwnFiles() async throws {
        // Review #226: same names must not overwrite a queued input or kept audio.
        let rig = try Self.rig(sources: FakeAudioSourceFactory(microphone: FakeAudioSource(track: .microphone, chunks: [])),
                               recognizer: AmplitudeTranscriber([(0.2, "One."), (0.4, "Two.")]))
        defer { try? FileManager.default.removeItem(at: rig.root) }
        let note = UUID()
        let first = rig.root.appending(path: "a/memo.wav")
        let second = rig.root.appending(path: "b/memo.wav")
        for url in [first, second] {
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        }
        try Self.writeWAV(first, parts: [(1, 0.2), (1, 0)])
        try Self.writeWAV(second, parts: [(1, 0.4), (1, 0)])
        let one = try await rig.service.transcribeFile(first, options: FileTranscriptionOptions(noteID: note))
        let two = try await rig.service.transcribeFile(second, options: FileTranscriptionOptions(noteID: note))
        let logOne = EventLog(one.events)
        let logTwo = EventLog(two.events)
        #expect(await logOne.waitUntilFinished())
        #expect(await logTwo.waitUntilFinished())
        guard case .finished(let a) = await logOne.events.last, case .finished(let b) = await logTwo.events.last else {
            Issue.record("both files finish")
            return
        }
        #expect(a.segments.map(\.text) == ["One."] && b.segments.map(\.text) == ["Two."])
        #expect(a.tracks.first?.fileName != b.tracks.first?.fileName)
        #expect(one.fileName == "memo.wav" && two.fileName == "memo.wav")
    }

    @Test func timeLeftComesFromMeasuredSpeed() {
        var meter = SpeedMeter()
        meter.record(audio: 2, took: 0.1)
        #expect(meter.remaining(for: 100) == nil, "too little measured yet")
        meter.record(audio: 8, took: 0.4)
        #expect(meter.remaining(for: 100).map { abs($0 - 5) < 1e-9 } == true, "0.5 s per 10 s of audio")
    }

    // MARK: Models and quit

    @Test func modelsUnloadAfterIdleAndLoadAgain() async throws {
        let mic = FakeAudioSource(track: .microphone, chunks: [])
        let rig = try Self.rig(sources: FakeAudioSourceFactory(microphone: mic), idleUnload: .milliseconds(50))
        defer { try? FileManager.default.removeItem(at: rig.root) }
        let first = try await rig.service.startDictation(DictationOptions())
        await first.stop()
        try await Task.sleep(for: .milliseconds(300))
        let second = try await rig.service.startDictation(DictationOptions())
        await second.stop()
        #expect(rig.loader.speechLoads == 2)
        #expect(rig.service.status.speech == .ready)
    }

    @Test func shutdownEndsTheMeetingAndCancelsFiles() async throws {
        let mic = FakeAudioSource(track: .microphone, chunks: Self.chunks([(1, 0.2), (1, 0)]))
        let rig = try Self.rig(sources: FakeAudioSourceFactory(microphone: mic),
                               recognizer: AmplitudeTranscriber([(0.2, "Bye.")]))
        defer { try? FileManager.default.removeItem(at: rig.root) }
        let note = UUID()
        let meeting = try await rig.service.startMeeting(MeetingOptions(noteID: note, callAudio: .off))
        let log = EventLog(meeting.events)
        let input = rig.root.appending(path: "queued.wav")
        try Self.writeWAV(input, parts: [(1, 0.2)])
        let fileNote = UUID()
        let job = try await rig.service.transcribeFile(input, options: FileTranscriptionOptions(noteID: fileNote))
        let jobLog = EventLog(job.events)
        await rig.service.shutdown()
        #expect(await log.waitUntilFinished())
        #expect(Self.result(await log.events) != nil)
        #expect(await jobLog.waitUntilFinished())
        #expect(await jobLog.events.last == .cancelled)
        #expect(mic.isStopped)
        // The file's copy stays for the next launch (Codex review on #251),
        // and the queue finds it under the name the service gave it.
        let copy = try #require(FileTranscriptionQueue.inputCopy(of: "queued.wav",
                                                                 in: rig.recordings.directory(for: fileNote)))
        #expect(copy.lastPathComponent.hasPrefix("import-"))
        // Closed cleanly: nothing for the next launch to recover.
        #expect(await rig.service.recoverInterruptedRecordings().isEmpty)
    }

    @Test func cancelDuringTheLastPhraseIsACancel() async throws {
        // Codex review on #251: recognition runs off the job's task, so a
        // Cancel while the last phrase is recognized must not end in success.
        let rig = try Self.rig(sources: FakeAudioSourceFactory(microphone: FakeAudioSource(track: .microphone, chunks: [])),
                               recognizer: AmplitudeTranscriber([(0.2, "Last.")], delay: .milliseconds(500)))
        defer { try? FileManager.default.removeItem(at: rig.root) }
        let input = rig.root.appending(path: "speech.wav")
        try Self.writeWAV(input, parts: [(1, 0.2)]) // speech to the end: the phrase ends at the flush
        let job = try await rig.service.transcribeFile(input, options: FileTranscriptionOptions(noteID: UUID()))
        let log = EventLog(job.events)
        #expect(await log.wait { $0.contains { if case .started = $0 { true } else { false } } })
        try await Task.sleep(for: .milliseconds(150))
        await rig.service.cancel(job.id)
        #expect(await log.waitUntilFinished())
        #expect(await log.events.last == .cancelled)
    }

    @Test func recordingsAreNotDeletedWhileAMeetingRecordsIntoThem() async throws {
        // Codex review on #251: deleting the note for good mid-meeting.
        let mic = FakeAudioSource(track: .microphone, chunks: Self.chunks([(1, 0.2), (1, 0)]))
        let rig = try Self.rig(sources: FakeAudioSourceFactory(microphone: mic),
                               recognizer: AmplitudeTranscriber([(0.2, "Hi.")]))
        defer { try? FileManager.default.removeItem(at: rig.root) }
        let note = UUID()
        let folder = rig.recordings.directory(for: note)
        let meeting = try await rig.service.startMeeting(MeetingOptions(noteID: note, callAudio: .off))
        let log = EventLog(meeting.events)
        await #expect(throws: SpeechError.busy(.meeting)) { try await rig.service.deleteRecordings(for: note) }
        #expect(FileManager.default.fileExists(atPath: folder.path))
        await meeting.stop()
        #expect(await log.waitUntilFinished())
        // Once the meeting has let go of the folder, it deletes.
        var deleted = false
        for _ in 0..<200 where !deleted {
            deleted = (try? await rig.service.deleteRecordings(for: note)) != nil
            if !deleted { try await Task.sleep(for: .milliseconds(10)) }
        }
        #expect(deleted)
        #expect(!FileManager.default.fileExists(atPath: folder.path))
    }

    @Test func quitWaitsBrieflyForAPhraseStillBeingRecognized() async throws {
        // Codex review on #251: quitting mid-phrase must not drop the phrase.
        let mic = FakeAudioSource(track: .microphone, chunks: Self.chunks([(1, 0.2), (1, 0)]))
        let rig = try Self.rig(sources: FakeAudioSourceFactory(microphone: mic),
                               recognizer: AmplitudeTranscriber([(0.2, "Last words.")], delay: .milliseconds(500)))
        defer { try? FileManager.default.removeItem(at: rig.root) }
        let meeting = try await rig.service.startMeeting(MeetingOptions(noteID: UUID(), callAudio: .off))
        let log = EventLog(meeting.events)
        #expect(await log.wait { events in
            events.contains { if case .segment(let segment) = $0 { !segment.isFinal } else { false } }
        })
        await rig.service.shutdown()
        #expect(await log.waitUntilFinished())
        let result = try #require(Self.result(await log.events))
        #expect(result.segments.map(\.text) == ["Last words."])
    }

    @Test func removeRefusesWhileRecordingAndThenDeletes() async throws {
        let rig = try Self.rig(sources: FakeAudioSourceFactory(microphone: FakeAudioSource(track: .microphone, chunks: [])))
        defer { try? FileManager.default.removeItem(at: rig.root) }
        let dictation = try await rig.service.startDictation(DictationOptions())
        await #expect(throws: SpeechError.busy(.dictation)) { try await rig.service.remove(.speech) }
        await dictation.stop()
        try await rig.service.remove(.speech)
        #expect(rig.service.status.speech == .notInstalled)
        await #expect(throws: SpeechError.notInstalled(.speech)) { _ = try await rig.service.startDictation(DictationOptions()) }
    }
}

// MARK: No call audio heard

extension SpeechServiceTests {
    static func hints(_ events: [MeetingEvent]) -> Int { events.filter { $0 == .noCallAudioHeard }.count }

    static func callFinals(_ events: [MeetingEvent]) -> Int { finals(events).filter { $0.track == .call }.count }

    /// A meeting whose call track plays `call`; returns its events once the
    /// call has `callFinals` finished phrases (and the hint, if `hint`).
    func silentCallMeeting(_ call: [(TimeInterval, Float)], playingOutput: Bool?, callFinals: Int = 0,
                           hint: Bool) async throws -> [MeetingEvent] {
        let mic = FakeAudioSource(track: .microphone, chunks: Self.chunks([(1, 0)]))
        let source = FakeAudioSource(track: .call, chunks: Self.chunks(call))
        let rig = try Self.rig(sources: FakeAudioSourceFactory(microphone: mic, call: source, playingOutput: playingOutput),
                               recognizer: AmplitudeTranscriber([(0.4, "Hello?")]))
        defer { try? FileManager.default.removeItem(at: rig.root) }
        let meeting = try await rig.service.startMeeting(MeetingOptions(noteID: UUID(), callAudio: .allApps))
        let log = EventLog(meeting.events)
        #expect(await log.wait { Self.callFinals($0) >= callFinals && (!hint || Self.hints($0) > 0) })
        await meeting.stop()
        #expect(await log.waitUntilFinished())
        return await log.events
    }

    @Test func silentCallAudioHintsOnceAfterThirtySeconds() async throws {
        // A refused tap: exact zeros for 45 s while other apps play.
        let events = try await silentCallMeeting([(45, 0)], playingOutput: true, hint: true)
        #expect(Self.hints(events) == 1, "once per meeting, not again at each recheck")
        #expect(events.first == .started(callAudio: true))
        #expect(!events.contains { if case .callAudioUnavailable = $0 { true } else { false } }, "only a hint")
        #expect(Self.result(events)?.tracks.map(\.kind).contains(.call) == true, "the call track still records")
    }

    @Test func silentCallAudioHintsWhenPlaybackCantBeRead() async throws {
        // Core Audio can't say whether other apps play: silence alone is enough.
        let events = try await silentCallMeeting([(31, 0)], playingOutput: nil, hint: true)
        #expect(Self.hints(events) == 1)
    }

    @Test func noHintBeforeThirtySeconds() async throws {
        // 25 s of silence, then the other side speaks; the phrase proves the
        // silence was all read.
        let events = try await silentCallMeeting([(25, 0), (1, 0.4), (1, 0)], playingOutput: true, callFinals: 1,
                                                 hint: false)
        #expect(Self.hints(events) == 0)
    }

    @Test func noHintOnceTheCallWasHeard() async throws {
        // The other side spoke at the start, then went quiet for 40 s.
        let events = try await silentCallMeeting([(1, 0.4), (40, 0), (1, 0.4), (1, 0)], playingOutput: true,
                                                 callFinals: 2, hint: false)
        #expect(Self.hints(events) == 0)
    }

    @Test func noHintWhileNothingElsePlays() async throws {
        // An in-person meeting: silence on the tap and no app playing sound
        // through 40 s (asked at 30, 35 and 40 s), then a phrase.
        let events = try await silentCallMeeting([(41, 0), (1, 0.4), (1, 0)], playingOutput: false, callFinals: 1,
                                                 hint: false)
        #expect(Self.hints(events) == 0)
    }
}

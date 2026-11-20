@preconcurrency import AVFoundation
import Foundation

/// Where recordings live and how they survive crashes and sleep (spec 6.3,
/// 7): `Recordings/<note-id>/`, one file per track per capture run
/// (`microphone-1.caf`, `call-1.caf`, then `-2` after a wake), plus
/// `recording.json`, which says the meeting is still open. Tracks are
/// written as crash-safe CAF while recording and converted to AAC `.m4a`
/// when the meeting ends (or at the next launch after a crash).
public final class RecordingStore: Sendable {
    /// Recording refuses to start with less free space (spec 6.3).
    public static let minimumFreeBytes: Int64 = 1 << 30
    static let manifestName = "recording.json"

    public let root: URL
    private let availableBytes: @Sendable (URL) -> UInt64?

    public static var defaultRoot: URL {
        AppStorageLocation.applicationSupport.appending(path: "Recordings", directoryHint: .isDirectory)
    }

    public init(root: URL = RecordingStore.defaultRoot,
                availableBytes: @escaping @Sendable (URL) -> UInt64? = DownloadManager.availableCapacity) {
        self.root = root
        self.availableBytes = availableBytes
    }

    public func directory(for noteID: UUID) -> URL {
        root.appending(path: noteID.uuidString, directoryHint: .isDirectory)
    }

    /// Throws `.lowDiskSpace` under 1 GB free.
    public func checkFreeSpace() throws {
        guard let available = availableBytes(root) else { return }
        if Int64(available) < Self.minimumFreeBytes { throw SpeechError.lowDiskSpace(availableBytes: Int64(available)) }
    }

    /// Bytes under `Recordings/`.
    public func totalSize() -> Int64 {
        guard let files = FileManager.default.enumerator(at: root, includingPropertiesForKeys: [.fileSizeKey]) else { return 0 }
        var total: Int64 = 0
        for case let url as URL in files {
            total += Int64((try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0)
        }
        return total
    }

    public func delete(noteID: UUID) throws {
        let folder = directory(for: noteID)
        guard FileManager.default.fileExists(atPath: folder.path) else { return }
        try FileManager.default.removeItem(at: folder)
    }

    // MARK: Manifest

    struct Manifest: Codable, Hashable {
        enum State: String, Codable {
            case recording
            /// Capture stopped cleanly at quit; the tracks convert to AAC at the next launch.
            case converting
            case finished
            case recovered
        }
        var noteID: UUID
        var startedAt: Date
        var state: State
        var keepAudio: Bool
        var tracks: [RecordedTrack]
        var updatedAt: Date
    }

    func manifestURL(_ noteID: UUID) -> URL { directory(for: noteID).appending(path: Self.manifestName) }

    func readManifest(_ noteID: UUID) -> Manifest? {
        guard let data = try? Data(contentsOf: manifestURL(noteID)) else { return nil }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return try? decoder.decode(Manifest.self, from: data)
    }

    func writeManifest(_ manifest: Manifest) throws {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.sortedKeys]
        try FileManager.default.createDirectory(at: directory(for: manifest.noteID), withIntermediateDirectories: true)
        try encoder.encode(manifest).write(to: manifestURL(manifest.noteID), options: .atomic)
    }

    // MARK: Recording

    /// Opens a meeting's folder and marks it as recording. Throws
    /// `.lowDiskSpace` first.
    func begin(noteID: UUID, startedAt: Date, keepAudio: Bool) throws -> RecordingSession {
        try checkFreeSpace()
        let manifest = Manifest(noteID: noteID, startedAt: startedAt, state: .recording, keepAudio: keepAudio,
                                tracks: [], updatedAt: startedAt)
        try writeManifest(manifest)
        return RecordingSession(store: self, manifest: manifest)
    }

    // MARK: Crash recovery

    /// Run once at launch. Meetings whose app died while recording: their
    /// CAF files are repaired and converted (or deleted when the audio was
    /// not to be kept), the folder is marked recovered, and they are
    /// returned. Meetings closed at quit (`converting`) are converted and
    /// marked finished without being reported: their result was delivered.
    /// The manifest is saved after each track, so dying mid-conversion
    /// never leaves it naming a deleted file.
    public func recoverInterrupted() async -> [InterruptedRecording] {
        guard let folders = try? FileManager.default.contentsOfDirectory(at: root, includingPropertiesForKeys: nil) else {
            return []
        }
        var recovered: [InterruptedRecording] = []
        for folder in folders {
            guard let noteID = UUID(uuidString: folder.lastPathComponent),
                  var manifest = readManifest(noteID), manifest.state == .recording || manifest.state == .converting
            else { continue }
            let crashed = manifest.state == .recording
            var endedAt = manifest.updatedAt
            for (index, track) in manifest.tracks.enumerated() {
                let url = folder.appending(path: track.fileName)
                guard FileManager.default.fileExists(atPath: url.path) else { continue }
                if let modified = try? url.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate {
                    endedAt = max(endedAt, modified)
                }
                guard url.pathExtension == "caf" else { continue }
                if crashed, let duration = try? TrackWriter.repair(url) { manifest.tracks[index].duration = duration }
                if manifest.keepAudio {
                    // A failed conversion keeps the CAF, which plays too.
                    if let m4a = try? await AudioFileConverter.convertToM4A(url) {
                        manifest.tracks[index].fileName = m4a.lastPathComponent
                    }
                } else {
                    try? FileManager.default.removeItem(at: url)
                }
                try? writeManifest(manifest)
            }
            let existing = manifest.tracks.filter {
                FileManager.default.fileExists(atPath: folder.appending(path: $0.fileName).path)
            }
            manifest.tracks = manifest.keepAudio ? existing : []
            manifest.state = crashed ? .recovered : .finished
            if crashed { manifest.updatedAt = endedAt }
            try? writeManifest(manifest)
            if crashed { recovered.append(InterruptedRecording(noteID: noteID, tracks: manifest.tracks, endedAt: endedAt)) }
        }
        return recovered.sorted { $0.endedAt < $1.endedAt }
    }
}

/// One meeting's files while it records. Used by the meeting pipeline from
/// one task at a time.
final class RecordingSession: @unchecked Sendable {
    private let store: RecordingStore
    private var manifest: RecordingStore.Manifest
    private var writers: [TrackKind: (writer: TrackWriter, index: Int)] = [:]
    private var runs: [TrackKind: Int] = [:]
    private var lastFlush = Date.distantPast
    /// Audio and the manifest reach the disk at least this often.
    static let flushInterval: TimeInterval = 3

    init(store: RecordingStore, manifest: RecordingStore.Manifest) {
        self.store = store
        self.manifest = manifest
    }

    var noteID: UUID { manifest.noteID }
    var directory: URL { store.directory(for: manifest.noteID) }
    var tracks: [RecordedTrack] { manifest.tracks }

    /// Opens a new file for `kind` that starts at `startOffset` (recording time).
    func openTrack(_ kind: TrackKind, at startOffset: TimeInterval) throws {
        try closeTrack(kind)
        let run = (runs[kind] ?? 0) + 1
        runs[kind] = run
        let name = "\(kind.rawValue)-\(run).caf"
        let writer = try TrackWriter(url: directory.appending(path: name))
        manifest.tracks.append(RecordedTrack(fileName: name, kind: kind, startOffset: startOffset, duration: 0))
        writers[kind] = (writer, manifest.tracks.count - 1)
        try saveManifest()
    }

    func write(_ samples: [Float], to kind: TrackKind, now: Date = .now) throws {
        guard let open = writers[kind] else { return }
        try open.writer.write(samples)
        manifest.tracks[open.index].duration = open.writer.duration
        if now.timeIntervalSince(lastFlush) >= Self.flushInterval { try flush(now: now) }
    }

    /// Audio to disk, then the manifest (so a crash loses at most a few seconds).
    func flush(now: Date = .now) throws {
        for open in writers.values { try open.writer.flush() }
        lastFlush = now
        manifest.updatedAt = now
        try saveManifest()
    }

    func closeTrack(_ kind: TrackKind) throws {
        guard let open = writers.removeValue(forKey: kind) else { return }
        try open.writer.close()
        manifest.tracks[open.index].duration = open.writer.duration
        try saveManifest()
    }

    /// Closes every open file (sleep, stop).
    func closeAll() throws {
        for kind in Array(writers.keys) { try closeTrack(kind) }
    }

    /// The closed CAF files of one kind, placed in recording time, for the final pass.
    func files(of kind: TrackKind) -> [(url: URL, startOffset: TimeInterval)] {
        manifest.tracks.filter { $0.kind == kind }.map { (directory.appending(path: $0.fileName), $0.startOffset) }
    }

    /// Ends the recording: converts every track to `.m4a` when the audio is
    /// kept, or deletes it. The manifest is saved after each track. A track
    /// that fails to convert stays as its CAF (it plays too) and the failure
    /// is returned. When `stopEarly` says so (the app is quitting, spec 6.1)
    /// the remaining tracks, and one being converted, are left as CAF and the
    /// folder is marked `converting`: `recoverInterrupted()` converts them at
    /// the next launch, and the returned tracks already name the `.m4a`
    /// files they will be.
    func finish(keepAudio: Bool, stopEarly: @escaping @Sendable () async -> Bool = { false }) async
        -> (tracks: [RecordedTrack], failure: SpeechError?) {
        var failure: SpeechError?
        do {
            try closeAll()
        } catch {
            failure = .failed("closing the recording: \(error.localizedDescription)")
        }
        manifest.keepAudio = keepAudio
        guard keepAudio else {
            for track in manifest.tracks { try? FileManager.default.removeItem(at: directory.appending(path: track.fileName)) }
            manifest.tracks = []
            manifest.state = .finished
            manifest.updatedAt = .now
            try? saveManifest()
            return ([], failure)
        }
        var deferred = false
        for (index, track) in manifest.tracks.enumerated() where (track.fileName as NSString).pathExtension == "caf" {
            if !deferred, await stopEarly() { deferred = true }
            if deferred { continue }
            do {
                let url = directory.appending(path: track.fileName)
                manifest.tracks[index].fileName = try await AudioFileConverter.convertToM4A(url, stopEarly: stopEarly)
                    .lastPathComponent
            } catch is AudioFileConverter.Stopped {
                deferred = true
            } catch {
                failure = failure ?? .failed("converting \(track.fileName): \(error.localizedDescription)")
            }
            try? saveManifest()
        }
        manifest.state = deferred ? .converting : .finished
        manifest.updatedAt = .now
        try? saveManifest()
        guard deferred else { return (manifest.tracks, failure) }
        return (manifest.tracks.map { track in
            var named = track
            if (track.fileName as NSString).pathExtension == "caf" {
                named.fileName = (track.fileName as NSString).deletingPathExtension + ".m4a"
            }
            return named
        }, failure)
    }

    private func saveManifest() throws { try store.writeManifest(manifest) }
}

/// AAC conversion of a finished track, read and written in bounded chunks.
enum AudioFileConverter {
    static let chunkFrames: AVAudioFrameCount = 16_000
    /// How many chunks are converted between two `stopEarly` checks.
    static let chunksPerCheck = 8

    /// `stopEarly` said stop (the app is quitting): the CAF is kept and the
    /// partial `.m4a` removed.
    struct Stopped: Error {}

    /// Writes `<name>.m4a` next to `caf`, then deletes `caf`. Returns the new
    /// file. Stops between chunks when the calling task is cancelled
    /// (`CancellationError`) or `stopEarly` says so (`Stopped`); either way
    /// the CAF stays and no partial `.m4a` is left.
    static func convertToM4A(_ caf: URL, stopEarly: @escaping @Sendable () async -> Bool = { false }) async throws -> URL {
        let output = caf.deletingPathExtension().appendingPathExtension("m4a")
        let conversion = Task.detached {
            try? FileManager.default.removeItem(at: output)
            do {
                let input = try AVAudioFile(forReading: caf)
                let format = input.processingFormat
                let settings: [String: Any] = [AVFormatIDKey: kAudioFormatMPEG4AAC,
                                               AVSampleRateKey: format.sampleRate,
                                               AVNumberOfChannelsKey: format.channelCount,
                                               AVEncoderAudioQualityKey: AVAudioQuality.high.rawValue]
                do {
                    let writer = try AVAudioFile(forWriting: output, settings: settings,
                                                 commonFormat: format.commonFormat, interleaved: format.isInterleaved)
                    guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: chunkFrames) else {
                        throw SpeechError.failed("cannot allocate a conversion buffer")
                    }
                    var chunks = 0
                    while input.framePosition < input.length {
                        try Task.checkCancellation()
                        if chunks % chunksPerCheck == 0, await stopEarly() { throw Stopped() }
                        chunks += 1
                        try input.read(into: buffer, frameCount: chunkFrames)
                        if buffer.frameLength == 0 { break }
                        try writer.write(from: buffer)
                    }
                }
            } catch {
                try? FileManager.default.removeItem(at: output)
                throw error
            }
            try FileManager.default.removeItem(at: caf)
        }
        try await withTaskCancellationHandler {
            try await conversion.value
        } onCancel: {
            conversion.cancel()
        }
        return output
    }
}

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
        enum State: String, Codable { case recording, finished, recovered }
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

    /// Meetings whose app quit while recording: their CAF files are
    /// repaired and converted, and the folder is marked recovered. Audio of
    /// a meeting set to "Delete after transcribing" is deleted instead.
    public func recoverInterrupted() async -> [InterruptedRecording] {
        guard let folders = try? FileManager.default.contentsOfDirectory(at: root, includingPropertiesForKeys: nil) else {
            return []
        }
        var recovered: [InterruptedRecording] = []
        for folder in folders {
            guard let noteID = UUID(uuidString: folder.lastPathComponent),
                  var manifest = readManifest(noteID), manifest.state == .recording else { continue }
            var tracks: [RecordedTrack] = []
            var endedAt = manifest.updatedAt
            for var track in manifest.tracks {
                let url = folder.appending(path: track.fileName)
                guard FileManager.default.fileExists(atPath: url.path) else { continue }
                if let modified = try? url.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate {
                    endedAt = max(endedAt, modified)
                }
                if url.pathExtension == "caf" {
                    guard let duration = try? TrackWriter.repair(url) else { continue }
                    track.duration = duration
                    if manifest.keepAudio, let m4a = try? await AudioFileConverter.convertToM4A(url) {
                        track.fileName = m4a.lastPathComponent
                    }
                }
                tracks.append(track)
            }
            if !manifest.keepAudio {
                for track in tracks { try? FileManager.default.removeItem(at: folder.appending(path: track.fileName)) }
                tracks = []
            }
            manifest.tracks = tracks
            manifest.state = .recovered
            manifest.updatedAt = endedAt
            try? writeManifest(manifest)
            recovered.append(InterruptedRecording(noteID: noteID, tracks: tracks, endedAt: endedAt))
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
    /// kept, or deletes it, and marks the folder finished.
    func finish(keepAudio: Bool) async throws -> [RecordedTrack] {
        try closeAll()
        var kept: [RecordedTrack] = []
        for var track in manifest.tracks {
            let url = directory.appending(path: track.fileName)
            if keepAudio {
                if url.pathExtension == "caf" {
                    track.fileName = try await AudioFileConverter.convertToM4A(url).lastPathComponent
                }
                kept.append(track)
            } else {
                try? FileManager.default.removeItem(at: url)
            }
        }
        manifest.tracks = kept
        manifest.state = .finished
        manifest.keepAudio = keepAudio
        manifest.updatedAt = .now
        try saveManifest()
        return kept
    }

    private func saveManifest() throws { try store.writeManifest(manifest) }
}

/// AAC conversion of a finished track, read and written in bounded chunks.
enum AudioFileConverter {
    static let chunkFrames: AVAudioFrameCount = 16_000

    /// Writes `<name>.m4a` next to `caf`, then deletes `caf`. Returns the new file.
    static func convertToM4A(_ caf: URL) async throws -> URL {
        let output = caf.deletingPathExtension().appendingPathExtension("m4a")
        try await Task.detached {
            try? FileManager.default.removeItem(at: output)
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
                while input.framePosition < input.length {
                    try Task.checkCancellation()
                    try input.read(into: buffer, frameCount: chunkFrames)
                    if buffer.frameLength == 0 { break }
                    try writer.write(from: buffer)
                }
            }
            try FileManager.default.removeItem(at: caf)
        }.value
        return output
    }
}

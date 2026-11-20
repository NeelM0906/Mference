import Foundation

/// Writes one track as 16-bit 16 kHz mono PCM in a CAF file, crash-safe
/// (spec 6.3): the data chunk's size stays "unknown" (-1, which CAF allows
/// for the last chunk) while recording, so the file is readable up to its
/// last flushed byte even if the app dies; `close()` writes the real size,
/// and `repair(_:)` does the same for a file a crash left open.
/// `flush()` pushes buffered bytes to disk; the session calls it every few
/// seconds. Not thread-safe: one writer per track, used from one task.
final class TrackWriter {
    static let sampleRate = SpeechAudio.sampleRate
    /// 'caff' header (8) + 'desc' chunk (12 + 32) + 'data' header (12) + edit count (4).
    static let headerLength: UInt64 = 8 + 44 + 12 + 4
    /// Offset of the data chunk's 64-bit size.
    static let dataSizeOffset: UInt64 = 8 + 44 + 4

    let url: URL
    private let handle: FileHandle
    private var pending = Data()
    private(set) var sampleCount = 0
    private(set) var isClosed = false

    /// Buffered bytes written at most this often without a flush.
    static let bufferLimit = 64 << 10

    init(url: URL) throws {
        let manager = FileManager.default
        try manager.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        guard manager.createFile(atPath: url.path, contents: Self.header(dataSize: -1)) else {
            throw SpeechError.failed("cannot create \(url.lastPathComponent)")
        }
        self.url = url
        handle = try FileHandle(forWritingTo: url)
        try handle.seekToEnd()
    }

    deinit { try? handle.close() }

    var duration: TimeInterval { SpeechAudio.seconds(sampleCount) }

    /// Appends 16 kHz mono samples (clipped to -1...1).
    func write(_ samples: [Float]) throws {
        guard !isClosed else { return }
        pending.reserveCapacity(pending.count + samples.count * 2)
        for sample in samples {
            let value = Int16(max(-1, min(1, sample)) * Float(Int16.max))
            withUnsafeBytes(of: value.littleEndian) { pending.append(contentsOf: $0) }
        }
        sampleCount += samples.count
        if pending.count >= Self.bufferLimit { try writePending() }
    }

    /// Pushes everything written so far to disk.
    func flush() throws {
        guard !isClosed else { return }
        try writePending()
        try handle.synchronize()
    }

    /// Writes the data chunk's size and closes the file.
    func close() throws {
        guard !isClosed else { return }
        try writePending()
        try handle.seek(toOffset: Self.dataSizeOffset)
        let size = Int64(4 + sampleCount * 2)
        try handle.write(contentsOf: withUnsafeBytes(of: size.bigEndian) { Data($0) })
        try handle.synchronize()
        try handle.close()
        isClosed = true
    }

    private func writePending() throws {
        guard !pending.isEmpty else { return }
        try handle.write(contentsOf: pending)
        pending.removeAll(keepingCapacity: true)
    }

    // MARK: Files a crash left open

    /// Sets the data size of a CAF file this writer made from its length
    /// (dropping a half-written sample) and returns its duration. Throws if
    /// the file is not one of ours.
    @discardableResult
    static func repair(_ url: URL) throws -> TimeInterval {
        do {
            return try repairFile(url)
        } catch let error as SpeechError {
            throw error
        } catch {
            throw SpeechError.failed("\(url.lastPathComponent): \(error.localizedDescription)")
        }
    }

    private static func repairFile(_ url: URL) throws -> TimeInterval {
        let handle = try FileHandle(forUpdating: url)
        defer { try? handle.close() }
        let length = try handle.seekToEnd()
        guard length >= headerLength else { throw SpeechError.failed("\(url.lastPathComponent) is too short") }
        try handle.seek(toOffset: 0)
        guard let magic = try handle.read(upToCount: 4), magic == Data("caff".utf8) else {
            throw SpeechError.failed("\(url.lastPathComponent) is not a CAF file")
        }
        var audioBytes = length - headerLength
        if audioBytes % 2 == 1 {
            audioBytes -= 1
            try handle.truncate(atOffset: headerLength + audioBytes)
        }
        try handle.seek(toOffset: dataSizeOffset)
        try handle.write(contentsOf: withUnsafeBytes(of: Int64(4 + audioBytes).bigEndian) { Data($0) })
        try handle.synchronize()
        return SpeechAudio.seconds(Int(audioBytes / 2))
    }

    /// Duration of a CAF file this writer made, from its length.
    static func duration(of url: URL) -> TimeInterval {
        let length = DownloadManager.fileSize(url)
        return length > headerLength ? SpeechAudio.seconds(Int((length - headerLength) / 2)) : 0
    }

    static func header(dataSize: Int64) -> Data {
        var data = Data()
        func append<T: FixedWidthInteger>(_ value: T) { withUnsafeBytes(of: value.bigEndian) { data.append(contentsOf: $0) } }
        data.append(contentsOf: Array("caff".utf8))
        append(UInt16(1))  // version
        append(UInt16(0))  // flags
        data.append(contentsOf: Array("desc".utf8))
        append(Int64(32))
        append(sampleRate.bitPattern)
        data.append(contentsOf: Array("lpcm".utf8))
        append(UInt32(2))  // kCAFLinearPCMFormatFlagIsLittleEndian; integer samples
        append(UInt32(2))  // bytes per packet
        append(UInt32(1))  // frames per packet
        append(UInt32(1))  // channels
        append(UInt32(16)) // bits per channel
        data.append(contentsOf: Array("data".utf8))
        append(dataSize)
        append(UInt32(0))  // edit count
        return data
    }
}

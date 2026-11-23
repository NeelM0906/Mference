import CryptoKit
import Darwin
import Foundation

/// A local copy of a pinned GGUF file, used as the source of the same install
/// the Hub path makes. It must be the pinned bytes: the size and the full
/// SHA-256 are checked before anything is planned from it, and the header is
/// then read through bounded, growing prefix reads, exactly as the remote
/// loader reads it.
enum LocalGGUFSource {
    /// Bytes per read while hashing.
    static let hashChunkBytes = 8 << 20

    /// What the file was when it was hashed: the same inode, size and
    /// modification time mean the same bytes. Every later reader of the file
    /// (the header parse and each copied range) checks it, so bytes that
    /// changed after the hash are never installed.
    struct Identity: Sendable, Equatable {
        let device: UInt64
        let inode: UInt64
        let size: UInt64
        let modifiedSeconds: Int64
        let modifiedNanoseconds: Int64

        static func of(fd: Int32, path: String) throws -> Identity {
            var st = stat()
            guard fstat(fd, &st) == 0 else {
                throw RepackError.configurationInvalid(detail: "fstat failed for \(path): errno \(errno)")
            }
            return Identity(device: UInt64(st.st_dev), inode: UInt64(st.st_ino), size: UInt64(st.st_size),
                            modifiedSeconds: Int64(st.st_mtimespec.tv_sec),
                            modifiedNanoseconds: Int64(st.st_mtimespec.tv_nsec))
        }

        func require(fd: Int32, path: String) throws {
            guard try Identity.of(fd: fd, path: path) == self else {
                throw RepackError.sourceFingerprintRejected(path: path, sha256: "file changed after verification")
            }
        }
    }

    /// Hashes the file and returns its identity, taken before the hash and
    /// required unchanged after it.
    @discardableResult
    static func verify(path: String, against shard: GGUFSourcePin.Shard, audit: RepackAudit) throws -> Identity {
        let fd = try Posix.openReadNoFollow(path)
        defer { close(fd) }
        let identity = try Identity.of(fd: fd, path: path)
        let size = try Posix.fileSize(fd: fd, path: path)
        guard size == shard.size else {
            throw RepackError.sourceFingerprintRejected(
                path: path, sha256: "size \(size), pinned \(shard.size)")
        }
        let chunk = UnsafeMutableRawBufferPointer.allocate(byteCount: hashChunkBytes, alignment: 16_384)
        defer { chunk.deallocate() }
        var hasher = SHA256()
        var offset: UInt64 = 0
        while offset < size {
            try Task.checkCancellation()
            let count = Int(min(UInt64(chunk.count), size - offset))
            try Posix.preadAll(fd: fd, path: path, buf: chunk.baseAddress!, count: count, offset: offset)
            hasher.update(bufferPointer: UnsafeRawBufferPointer(start: chunk.baseAddress, count: count))
            audit.recordRead(bytes: count)
            offset += UInt64(count)
        }
        let digest = hasher.finalize().map { String(format: "%02x", $0) }.joined()
        guard digest == shard.sha256.lowercased() else {
            throw RepackError.sourceFingerprintRejected(path: path, sha256: digest)
        }
        try identity.require(fd: fd, path: path)
        return identity
    }

    /// Parses the header from growing prefixes of the file. Returns the header
    /// and the prefix bytes read.
    static func readHeader(path: String, size: UInt64,
                           identity: Identity? = nil) throws -> (GGUF.Header, UInt64) {
        let fd = try Posix.openReadNoFollow(path)
        defer { close(fd) }
        try identity?.require(fd: fd, path: path)
        var want = min(RemoteGGUFSnapshotLoader.initialHeaderRead, Int(clamping: size))
        while true {
            var prefix = Data(count: want)
            try prefix.withUnsafeMutableBytes { raw in
                try Posix.preadAll(fd: fd, path: path, buf: raw.baseAddress!, count: want, offset: 0)
            }
            switch try GGUF.parseHeader(path: path, fileSize: size, prefix: prefix) {
            case .complete(let header):
                try identity?.require(fd: fd, path: path)
                return (header, UInt64(want))
            case .needBytes(let n):
                guard n > want else {
                    throw RepackError.ggufInvalid(path: path, detail: "header does not fit in the file")
                }
                want = n
            }
        }
    }
}

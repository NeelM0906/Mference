import CryptoKit
import Foundation

/// A pinned multi-shard GGUF upload: one commit and the exact LFS SHA-256 and
/// size of every shard, in split order.
public struct GGUFSourcePin: Sendable, Equatable {
    public struct Shard: Sendable, Equatable {
        public let path: String
        public let sha256: String
        public let size: UInt64

        public init(path: String, sha256: String, size: UInt64) {
            self.path = path
            self.sha256 = sha256
            self.size = size
        }
    }

    public let commit: String
    public let shards: [Shard]

    public init(commit: String, shards: [Shard]) {
        self.commit = commit
        self.shards = shards
    }

    /// SHA-256 over the canonical pin text; the GGUF analogue of the
    /// safetensors index hash recorded as `sourceSnapshotHash`.
    public var fingerprint: String {
        var text = "gguf-pin-v1\n\(commit)\n"
        for s in shards { text += "\(s.path) \(s.sha256) \(s.size)\n" }
        return SHA256.hash(data: Data(text.utf8)).map { String(format: "%02x", $0) }.joined()
    }

    public var totalBytes: UInt64 { shards.reduce(0) { $0 + $1.size } }
}

struct RemoteGGUFSnapshot {
    let pin: GGUFSourcePin
    /// One header per shard, in split order.
    let headers: [GGUF.Header]
    let remoteFiles: [String: RemoteFileInfo]
    /// Header bytes fetched to establish the plan.
    let metadataSourceBytes: UInt64
}

/// Resolves and parses a pinned GGUF upload. Fails closed: every shard's
/// resolve response must report the pinned commit and the pinned LFS SHA-256
/// and size before any of its bytes are read.
enum RemoteGGUFSnapshotLoader {
    static let initialHeaderRead = 1 << 20

    static func load(remote: HuggingFaceRemoteSource,
                     pin: GGUFSourcePin,
                     audit: RepackAudit? = nil) async throws -> RemoteGGUFSnapshot {
        guard !pin.shards.isEmpty else { throw RepackError.configurationInvalid(detail: "GGUF pin has no shards") }
        let pinned = remote.pinned(commit: pin.commit)
        var files: [String: RemoteFileInfo] = [:]
        var headers: [GGUF.Header] = []
        var fetched: UInt64 = 0
        for shard in pin.shards {
            let info = try await pinned.resolveFileInfo(filename: shard.path, audit: audit)
            try validate(info: info, against: shard, commit: pin.commit)
            files[shard.path] = info
            let (header, bytes) = try await readHeader(remote: pinned, info: info, path: shard.path, audit: audit)
            headers.append(header)
            fetched += bytes
        }
        try validateSplit(headers: headers, pin: pin)
        return RemoteGGUFSnapshot(pin: pin, headers: headers, remoteFiles: files, metadataSourceBytes: fetched)
    }

    static func validate(info: RemoteFileInfo, against shard: GGUFSourcePin.Shard, commit: String) throws {
        guard info.resolvedCommit == commit else {
            throw RepackError.remoteProtocolInvalid(
                detail: "\(shard.path) resolved to commit \(info.resolvedCommit), pinned \(commit)")
        }
        let etag = (info.etag ?? "").trimmingCharacters(in: CharacterSet(charactersIn: "\"W/")).lowercased()
        guard etag == shard.sha256.lowercased(), info.size == shard.size else {
            throw RepackError.sourceFingerprintRejected(path: shard.path, sha256: etag.isEmpty ? "<no etag>" : etag)
        }
        guard info.acceptsRanges else {
            throw RepackError.remoteProtocolInvalid(detail: "\(shard.path) does not advertise byte ranges")
        }
    }

    /// Reads the header with growing range reads, appending only the bytes
    /// not yet fetched. Returns the header and the bytes fetched.
    private static func readHeader(remote: HuggingFaceRemoteSource, info: RemoteFileInfo, path: String,
                                   audit: RepackAudit?) async throws -> (GGUF.Header, UInt64) {
        var prefix = Data()
        var want = min(initialHeaderRead, Int(clamping: info.size))
        while true {
            if want > prefix.count {
                let part = try await remote.downloadRangeToTempFile(filename: path, info: info,
                                                                    offset: UInt64(prefix.count),
                                                                    length: want - prefix.count, audit: audit)
                defer { try? FileManager.default.removeItem(atPath: part.path) }
                let data = try Data(contentsOf: URL(fileURLWithPath: part.path))
                guard data.count == want - prefix.count else {
                    throw RepackError.remoteBodyTruncated(path: path, expected: UInt64(want - prefix.count),
                                                          actual: UInt64(data.count))
                }
                prefix.append(data)
            }
            switch try GGUF.parseHeader(path: path, fileSize: info.size, prefix: prefix) {
            case .complete(let h):
                return (h, UInt64(prefix.count))
            case .needBytes(let n):
                guard n > prefix.count else {
                    throw RepackError.ggufInvalid(path: path, detail: "header does not fit in the file")
                }
                want = n
            }
        }
    }

    /// Split GGUF bookkeeping: `split.no` is 0-based and must match the pin
    /// order, every shard must agree on `split.count`, and the tensor total
    /// must equal `split.tensors.count`.
    static func validateSplit(headers: [GGUF.Header], pin: GGUFSourcePin) throws {
        var total = 0
        for (i, h) in headers.enumerated() {
            let path = pin.shards[i].path
            guard h.metadata["split.no"]?.uint64 == UInt64(i),
                  h.metadata["split.count"]?.uint64 == UInt64(headers.count) else {
                throw RepackError.ggufInvalid(path: path, detail: "split.no/split.count do not match the pin order")
            }
            total += h.tensors.count
        }
        guard headers[0].metadata["split.tensors.count"]?.uint64 == UInt64(total) else {
            throw RepackError.ggufInvalid(path: pin.shards[0].path,
                                          detail: "split.tensors.count != \(total) tensors across shards")
        }
        var names = Set<String>()
        for (i, h) in headers.enumerated() {
            for t in h.tensors where !names.insert(t.name).inserted {
                throw RepackError.ggufInvalid(path: pin.shards[i].path, detail: "tensor \(t.name) in two shards")
            }
        }
    }
}

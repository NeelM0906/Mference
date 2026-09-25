import Foundation

public enum PrefillError: Error, CustomStringConvertible, Equatable {
    public static let chunkedRequiresChunkedRunnerReason =
        "chunked prefill requires a ChunkedPrefillRunner-backed runtime"

    case chunkedUnsupported(String)
    case chunkedRunnerDirty(String)
    case prefillCursorMismatch(String)
    case unsupportedPrefillSeed(String)

    public var description: String {
        switch self {
        case .chunkedUnsupported(let reason),
             .chunkedRunnerDirty(let reason),
             .prefillCursorMismatch(let reason),
             .unsupportedPrefillSeed(let reason):
            return reason
        }
    }
}

struct PrefillChunkCommitState: Sendable, Equatable {
    private(set) var isDirty = false
    private(set) var inFlightStartPosition: Int?
    private(set) var inFlightTokenCount: Int?

    var inFlightEndPosition: Int? {
        guard let start = inFlightStartPosition,
              let count = inFlightTokenCount else { return nil }
        return start + count
    }

    init() {}

    mutating func markDirty(startPosition: Int, tokenCount: Int) {
        precondition(startPosition >= 0, "prefill dirty startPosition must be non-negative")
        precondition(tokenCount > 0, "prefill dirty tokenCount must be positive")
        isDirty = true
        inFlightStartPosition = startPosition
        inFlightTokenCount = tokenCount
    }

    mutating func markCommitted() {
        isDirty = false
        inFlightStartPosition = nil
        inFlightTokenCount = nil
    }

    mutating func reset() {
        markCommitted()
    }

    func requireClean(operation: String) throws {
        guard !isDirty else {
            let range: String
            if let start = inFlightStartPosition, let end = inFlightEndPosition {
                range = " for in-flight tokens [\(start), \(end))"
            } else {
                range = ""
            }
            throw PrefillError.chunkedRunnerDirty(
                "\(operation) rejected because a previous forward operation wrote sequence state\(range) but did not commit; call reset() before reusing the runner")
        }
    }
}

struct PrefillChunkSpan: Sendable, Equatable {
    let tokenOffset: Int
    let tokenCount: Int
    let startPosition: Int
    let completedCount: Int

    init(tokenOffset: Int,
                tokenCount: Int,
                startPosition: Int,
                completedCount: Int) {
        self.tokenOffset = tokenOffset
        self.tokenCount = tokenCount
        self.startPosition = startPosition
        self.completedCount = completedCount
    }

}

enum PrefillChunkPlanner {
    static func spans(tokenCount: Int,
                             startPosition: Int,
                             config: PrefillRuntimeConfig) -> [PrefillChunkSpan] {
        spans(tokenCount: tokenCount,
              startPosition: startPosition,
              chunkTokens: config.chunkTokens)
    }

    static func spans(tokenCount: Int,
                             startPosition: Int,
                             chunkTokens: Int) -> [PrefillChunkSpan] {
        precondition(tokenCount >= 0, "prefill tokenCount must be non-negative")
        precondition(startPosition >= 0, "prefill startPosition must be non-negative")
        let chunk = max(1, min(chunkTokens, PrefillRuntimeConfig.maxChunkTokens))
        guard tokenCount > 0 else { return [] }

        var spans: [PrefillChunkSpan] = []
        spans.reserveCapacity((tokenCount + chunk - 1) / chunk)
        var offset = 0
        while offset < tokenCount {
            let count = min(chunk, tokenCount - offset)
            let completed = offset + count
            spans.append(PrefillChunkSpan(tokenOffset: offset,
                                          tokenCount: count,
                                          startPosition: startPosition + offset,
                                          completedCount: completed))
            offset = completed
        }
        return spans
    }
}

public enum PrefillKVStorageMode: String, Sendable, Equatable {
    case fp16
    case bf16
}

public enum PrefillExecutedMode: String, Sendable, Equatable, Codable {
    case off
    case chunked
    case sequential
    case mixed
    case unreported
    case unsupported
}

/// Work actually completed by one prefill call, excluding cached prompt tokens
/// and subsequent decode. Record a batch only after its execution succeeds.
/// Token-ordered recurrence within a layer-major GPU batch is still batched.
public struct PrefillExecutionReport: Sendable, Equatable, Encodable {
    public private(set) var batchedChunkSizes: [Int] = []
    public private(set) var replayedTokens: Int = 0
    public private(set) var replayReasons: [String: Int] = [:]

    public init() {}

    public var batchedTokens: Int { batchedChunkSizes.reduce(0, +) }
    public var computedTokens: Int { batchedTokens + replayedTokens }
    public var executedMode: PrefillExecutedMode {
        if batchedTokens > 0 { return replayedTokens > 0 ? .mixed : .chunked }
        return replayedTokens > 0 ? .sequential : .off
    }

    mutating func append(_ other: PrefillExecutionReport) {
        batchedChunkSizes += other.batchedChunkSizes
        replayedTokens += other.replayedTokens
        for (reason, count) in other.replayReasons { replayReasons[reason, default: 0] += count }
    }

    mutating func recordBatch(_ tokenCount: Int) {
        precondition(tokenCount > 0)
        batchedChunkSizes.append(tokenCount)
    }

    mutating func recordReplay(_ tokenCount: Int, reason: String) {
        precondition(tokenCount > 0 && !reason.isEmpty)
        replayedTokens += tokenCount
        replayReasons[reason, default: 0] += tokenCount
    }

    private enum CodingKeys: String, CodingKey {
        case executedMode, batchedTokens, replayedTokens, batchedChunkSizes, replayReasons
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(executedMode, forKey: .executedMode)
        try c.encode(batchedTokens, forKey: .batchedTokens)
        try c.encode(replayedTokens, forKey: .replayedTokens)
        try c.encode(batchedChunkSizes, forKey: .batchedChunkSizes)
        try c.encode(replayReasons, forKey: .replayReasons)
    }
}

public enum PrefillChunkCompleteness: String, Sendable, Equatable {
    case complete
    case unreported
    case unsupported
}

public struct PrefillExecutionDiagnostics: Sendable, Equatable {
    public let requestedMode: PrefillRuntimeConfig.Mode
    public let executedMode: PrefillExecutedMode
    public let kvStorageMode: PrefillKVStorageMode?
    public let chunkCompleteness: PrefillChunkCompleteness
    public let unsupportedReason: String?

    public init(config: PrefillRuntimeConfig,
                executedMode: PrefillExecutedMode,
                kvStorageMode: PrefillKVStorageMode? = nil,
                chunkCompleteness: PrefillChunkCompleteness? = nil,
                unsupportedReason: String? = nil) {
        self.requestedMode = config.mode
        self.executedMode = executedMode
        self.kvStorageMode = kvStorageMode
        self.chunkCompleteness = chunkCompleteness
            ?? (executedMode == .unreported ? .unreported
                : executedMode == .unsupported ? .unsupported : .complete)
        self.unsupportedReason = unsupportedReason
    }

    public static func unsupported(config: PrefillRuntimeConfig,
                                   kvStorageMode: PrefillKVStorageMode? = nil,
                                   reason: String) -> PrefillExecutionDiagnostics {
        PrefillExecutionDiagnostics(config: config,
                                    executedMode: .unsupported,
                                    kvStorageMode: kvStorageMode,
                                    chunkCompleteness: .unsupported,
                                    unsupportedReason: reason)
    }
}

public struct PrefillRuntimeConfig: Sendable, Equatable {
    public enum Mode: String, Sendable, Equatable {
        case off
        case chunked
    }

    /// Largest supported prefill chunk. Chunked prefill re-reads each
    /// layer's routed experts once per chunk, so expert I/O scales with
    /// prompt_tokens / chunk_tokens: measured on Qwen 3.6 (M5, 2,940-token
    /// long-synthesis case), chunk 128 reads 228 GB and chunk 4096 reads
    /// 18 GB -- one sequential sweep of the expert files, the floor.
    /// Scratch and the FP16 KV ring scale with the CONFIGURED chunk, not
    /// this cap, so installations that keep the 128 default see no change.
    public static let maxChunkTokens = 4096

    public let mode: Mode
    public let chunkTokens: Int

    private init(mode: Mode, chunkTokens: Int) {
        self.mode = mode
        self.chunkTokens = chunkTokens
    }

    public var enabled: Bool { mode == .chunked }

    public static var off: PrefillRuntimeConfig {
        PrefillRuntimeConfig(mode: .off, chunkTokens: 128)
    }

    public static var defaultChunked: PrefillRuntimeConfig {
        production(chunkTokens: 128)
    }

    public static func production(chunkTokens: Int) -> PrefillRuntimeConfig {
        precondition(RuntimeConfiguration.allowedPrefillChunkTokens.contains(chunkTokens),
                     "unsupported prefill chunk size")
        return PrefillRuntimeConfig(mode: .chunked, chunkTokens: chunkTokens)
    }
}

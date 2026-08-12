import Mference

public enum AppContextLengthOption: Int, CaseIterable, Identifiable, Sendable {
    case fourK = 4_096
    case eightK = 8_192
    case sixteenK = 16_384
    case thirtyTwoK = 32_768
    case sixtyFourK = 65_536
    case oneTwentyEightK = 128_000

    public var id: Int { rawValue }
    public var tokens: Int { rawValue }

    public var shortLabel: String {
        switch self {
        case .oneTwentyEightK: "128K"
        default: "\(tokens / 1_024)K"
        }
    }

    public var fp16KVBytes: UInt64 {
        // Sized against the Gemma 4 baseline; Qwen 3.6 and DeepSeek V4 have
        // different (smaller-per-token) attention state, so the menu labels
        // overstate their cost. The option is a UI estimate, not a budget.
        let architecture = ArchConfig.gemma4_26B_A4B
        let fullLayers = architecture.fullAttentionLayerMask.reduce(0) {
            $0 + ($1 == 0 ? 0 : 1)
        }
        let slidingLayers = architecture.numLayers - fullLayers
        let fp16Bytes = 2
        let keyAndValue = 2
        let slidingRows = min(
            tokens,
            architecture.slidingWindow + PrefillRuntimeConfig.defaultChunked.chunkTokens)
        let slidingBytesPerRow = architecture.numKVHeads
            * architecture.headDim * keyAndValue * fp16Bytes
        let fullBytesPerRow = architecture.numFullKVHeads
            * architecture.fullHeadDim * keyAndValue * fp16Bytes
        return UInt64(slidingLayers * slidingRows * slidingBytesPerRow)
            + UInt64(fullLayers * tokens * fullBytesPerRow)
    }

    public var menuLabel: String {
        switch self {
        case .fourK: "4K, Default"
        case .eightK: "8K, +85 MB"
        case .sixteenK: "16K, +250 MB"
        case .thirtyTwoK: "32K, +590 MB"
        case .sixtyFourK: "64K, +1.26 GB"
        case .oneTwentyEightK: "128K, +2.54 GB"
        }
    }
}

import Foundation

/// One physical tensor in a safetensors shard. Coordinates are absolute file
/// offsets so the writer can map and copy a single tensor without re-parsing
/// the shard header.
struct SourceTensor: Sendable, Hashable {
    enum Dtype: UInt8, Sendable, Hashable {
        case u32  = 0
        case bf16 = 1
        case fp16 = 2
        case fp32 = 3
        /// Integer lookup tables (DeepSeek V4's `tid2eid`). Copied through
        /// as raw bytes; the runtime reads them dtype-aware on the CPU.
        case i64  = 4
        /// Integer lookup tables, including Maple FlashHead's token map.
        case i32  = 5

        var elementBytes: Int {
            switch self {
            case .u32: 4; case .bf16: 2; case .fp16: 2; case .fp32: 4
            case .i64: 8; case .i32: 4
            }
        }
    }

    let name: String
    let shardPath: String
    let dtype: Dtype
    let shape: [UInt64]
    let absoluteOffset: UInt64
    let sizeBytes: UInt64
}

/// Bit-width + group-size override resolved from `config.json ->
/// quantization`. DeepSeek V4's conversion mixes group sizes on 2-bit
/// routed experts (`gate_proj` is group 32 on most layers, 64 on the last).
struct QuantSpec: Sendable, Hashable {
    let bits: Int
    let groupSize: Int
}

import Metal

extension Native {
    /// Element type of an activation or state buffer.
    public enum DType: String, Codable, Sendable {
        case f32, f16, bf16, i32, u32
        public var byteSize: Int {
            switch self {
            case .f32, .i32, .u32: return 4
            case .f16, .bf16: return 2
            }
        }
    }

    /// A 2-D view of GPU memory: `rows` rows of `cols` elements of `dtype`,
    /// row `r` starting at `offset + r * rowStride` bytes in `buffer`.
    /// Activations are F32 unless a contract says otherwise
    /// (`docs/NATIVE_ENGINE.md`, "Precision").
    public struct TensorRef {
        public var buffer: MTLBuffer
        public var offset: Int
        public var dtype: DType
        public var rows: Int
        public var cols: Int
        public var rowStride: Int

        public init(buffer: MTLBuffer, offset: Int = 0, dtype: DType,
                    rows: Int, cols: Int, rowStride: Int? = nil) {
            self.buffer = buffer
            self.offset = offset
            self.dtype = dtype
            self.rows = rows
            self.cols = cols
            self.rowStride = rowStride ?? cols * dtype.byteSize
        }
    }

    /// A GPU-resident token-id slot (`u32` ids), so a step can consume the
    /// token the previous step sampled without a CPU round trip (pipelining).
    public struct TokenSlot {
        public var buffer: MTLBuffer
        public var offset: Int
        public var count: Int
        public init(buffer: MTLBuffer, offset: Int, count: Int) {
            self.buffer = buffer
            self.offset = offset
            self.count = count
        }
    }
}

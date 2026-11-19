import Foundation
import Metal

extension Native {
    /// Shared helpers for the reference ops' Metal side.
    enum ReferenceMetal {
        static let module = "native_reference"

        /// `RefLayout` in native_reference.metal.
        struct Layout {
            var type: UInt32
            var blockElems: UInt32
            var blockBytes: UInt32
            var blocksPerRow: UInt32
            var rowsPerTile: UInt32
            var blocksPerTile: UInt32
            var planes: UInt32
            var fieldCount: UInt32
            var fieldBytes: (UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32)

            init(_ tensor: QuantTensor) throws {
                guard let format = QuantBlockFormat.of(tensor.type),
                      ReferenceDequant.supportedTypes.contains(tensor.type) else {
                    throw NativeError.unsupported("reference dequant of \(tensor.type)")
                }
                guard format.elements <= 256, format.bytes <= 256, format.fields.count <= 8 else {
                    throw NativeError.unsupported("block of \(tensor.type) too large for the reference")
                }
                guard tensor.cols % format.elements == 0 else {
                    throw NativeError.invalidShape("\(tensor.name): \(tensor.cols) inputs not whole blocks")
                }
                let blocksPerRow = tensor.cols / format.elements
                guard tensor.layout.fits(rows: tensor.rows, blocksPerRow: blocksPerRow) else {
                    throw NativeError.invalidShape("\(tensor.name): layout \(tensor.layout) does not fit")
                }
                guard tensor.byteCount == tensor.rows * blocksPerRow * format.bytes else {
                    throw NativeError.invalidShape("\(tensor.name): byteCount \(tensor.byteCount)")
                }
                var fb = [UInt32](repeating: 0, count: 8)
                for (i, f) in format.fields.enumerated() { fb[i] = UInt32(f.bytes) }
                type = tensor.type.rawValue
                blockElems = UInt32(format.elements)
                blockBytes = UInt32(format.bytes)
                self.blocksPerRow = UInt32(blocksPerRow)
                rowsPerTile = UInt32(tensor.layout.rowsPerTile)
                blocksPerTile = UInt32(tensor.layout.blocksPerTile)
                planes = tensor.layout.planes ? 1 : 0
                fieldCount = UInt32(format.fields.count)
                fieldBytes = (fb[0], fb[1], fb[2], fb[3], fb[4], fb[5], fb[6], fb[7])
            }
        }

        static func setBytes<T>(_ encoder: MTLComputeCommandEncoder, _ value: T, index: Int) {
            var v = value
            encoder.setBytes(&v, length: MemoryLayout<T>.stride, index: index)
        }

        /// Dispatch `width x height` threads in 2-D threadgroups.
        static func dispatch(_ encoder: MTLComputeCommandEncoder, _ pipeline: MTLComputePipelineState,
                             width: Int, height: Int) {
            guard width > 0, height > 0 else { return }
            encoder.setComputePipelineState(pipeline)
            let w = min(pipeline.threadExecutionWidth, width)
            let h = max(1, min(pipeline.maxTotalThreadsPerThreadgroup / w, height, 4))
            encoder.dispatchThreads(MTLSize(width: width, height: height, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: w, height: h, depth: 1))
        }

        static func checkF32(_ t: TensorRef, rows: Int, cols: Int, _ what: String) throws {
            guard t.dtype == .f32, t.rows == rows, t.cols == cols, t.rowStride >= cols * 4,
                  t.offset % 4 == 0, t.rowStride % 4 == 0 else {
                throw NativeError.invalidShape("\(what): expected F32 \(rows)x\(cols), got \(t.dtype) \(t.rows)x\(t.cols)")
            }
        }
    }

    /// The reference `QuantGEMM`: one thread per output value, each block
    /// gathered through the tensor's `TileLayout` and dequantized exactly as
    /// ggml's `dequantize_row_<type>`, F32 accumulation in input order. The
    /// same kernel serves every entry point and every layout; the entry points
    /// differ only in the row counts they accept.
    public final class ReferenceQuantGEMM: QuantGEMM {
        let context: MetalContext
        private var gemm: MTLComputePipelineState?
        private var getRows: MTLComputePipelineState?

        public init(context: MetalContext) {
            self.context = context
        }

        public func supports(type: GGMLType, layout: TileLayout, family: GPUFamilyClass) -> Bool {
            ReferenceDequant.supportedTypes.contains(type)
        }

        public func prepare(types: Set<GGMLType>) throws {
            for t in types where !ReferenceDequant.supportedTypes.contains(t) {
                throw NativeError.unsupported("reference QuantGEMM: \(t)")
            }
            gemm = try context.modulePipeline(module: ReferenceMetal.module, name: "ref_quant_gemm")
            getRows = try context.modulePipeline(module: ReferenceMetal.module, name: "ref_get_rows")
        }

        public func encodeGEMV(_ encoder: MTLComputeCommandEncoder, weight: QuantTensor, input: TensorRef,
                               output: TensorRef, mode: GEMMOutputMode) throws {
            guard input.rows == 1 else { throw NativeError.invalidShape("GEMV needs 1 row, got \(input.rows)") }
            try encode(encoder, weight: weight, input: input, output: output, mode: mode)
        }

        public func encodeVerify(_ encoder: MTLComputeCommandEncoder, weight: QuantTensor, input: TensorRef,
                                 output: TensorRef, mode: GEMMOutputMode) throws {
            guard (2...16).contains(input.rows) else {
                throw NativeError.invalidShape("verify needs 2...16 rows, got \(input.rows)")
            }
            try encode(encoder, weight: weight, input: input, output: output, mode: mode)
        }

        public func encodePrefill(_ encoder: MTLComputeCommandEncoder, weight: QuantTensor, input: TensorRef,
                                  output: TensorRef, mode: GEMMOutputMode) throws {
            guard input.rows >= 1 else { throw NativeError.invalidShape("prefill needs rows") }
            try encode(encoder, weight: weight, input: input, output: output, mode: mode)
        }

        struct GemmParams {
            var outRows: UInt32
            var rows: UInt32
            var inputStride: UInt32
            var outputStride: UInt32
            var accumulate: UInt32
        }

        private func encode(_ encoder: MTLComputeCommandEncoder, weight: QuantTensor, input: TensorRef,
                            output: TensorRef, mode: GEMMOutputMode) throws {
            guard let gemm else { throw NativeError.resource("ReferenceQuantGEMM.prepare not called") }
            let layout = try ReferenceMetal.Layout(weight)
            try ReferenceMetal.checkF32(input, rows: input.rows, cols: weight.cols, "GEMM input")
            try ReferenceMetal.checkF32(output, rows: input.rows, cols: weight.rows, "GEMM output")
            encoder.setBuffer(weight.buffer, offset: weight.offset, index: 0)
            encoder.setBuffer(input.buffer, offset: input.offset, index: 1)
            encoder.setBuffer(output.buffer, offset: output.offset, index: 2)
            ReferenceMetal.setBytes(encoder, GemmParams(outRows: UInt32(weight.rows), rows: UInt32(input.rows),
                                                        inputStride: UInt32(input.rowStride),
                                                        outputStride: UInt32(output.rowStride),
                                                        accumulate: mode == .accumulate ? 1 : 0), index: 3)
            ReferenceMetal.setBytes(encoder, layout, index: 4)
            ReferenceMetal.dispatch(encoder, gemm, width: weight.rows, height: input.rows)
        }

        struct GetRowsParams {
            var tokens: UInt32
            var vocab: UInt32
            var outputStride: UInt32
        }

        public func encodeGetRows(_ encoder: MTLComputeCommandEncoder, weight: QuantTensor, tokens: TokenSlot,
                                  output: TensorRef) throws {
            guard let getRows else { throw NativeError.resource("ReferenceQuantGEMM.prepare not called") }
            let layout = try ReferenceMetal.Layout(weight)
            try ReferenceMetal.checkF32(output, rows: tokens.count, cols: weight.cols, "get_rows output")
            encoder.setBuffer(weight.buffer, offset: weight.offset, index: 0)
            encoder.setBuffer(tokens.buffer, offset: tokens.offset, index: 1)
            encoder.setBuffer(output.buffer, offset: output.offset, index: 2)
            ReferenceMetal.setBytes(encoder, GetRowsParams(tokens: UInt32(tokens.count), vocab: UInt32(weight.rows),
                                                           outputStride: UInt32(output.rowStride)), index: 3)
            ReferenceMetal.setBytes(encoder, layout, index: 4)
            ReferenceMetal.dispatch(encoder, getRows, width: Int(layout.blocksPerRow), height: tokens.count)
        }
    }

    /// The reference `KernelCalibration`: one variant per type, every family.
    public final class ReferenceKernelCalibration: KernelCalibration {
        public let family: GPUFamilyClass
        public init(family: GPUFamilyClass) { self.family = family }
        public func variant(for type: GGMLType, rows: GEMMRowClass) -> String {
            switch rows {
            case .gemv: return "reference.\(type).gemv"
            case .verify(let n): return "reference.\(type).verify\(n)"
            case .prefill(let n): return "reference.\(type).prefill\(n)"
            }
        }
    }
}

extension Native.ReferenceCPU {
    /// CPU twin of `ref_quant_gemm`: `y[r][o] = sum_i W[o][i] * x[r][i]`, the
    /// same block order and F32 accumulation order. `x` is `rows x cols`
    /// row-major; the result is `rows x weight.rows`.
    public static func gemm(weight: Native.QuantTensor, input x: [Float], rows: Int) -> [Float] {
        let cols = weight.cols
        var y = [Float](repeating: 0, count: rows * weight.rows)
        for o in 0..<weight.rows {
            let w = Native.ReferenceDequant.row(of: weight, o)
            for r in 0..<rows {
                var acc: Float = 0
                for i in 0..<cols { acc += w[i] * x[r * cols + i] }
                y[r * weight.rows + o] = acc
            }
        }
        return y
    }

    /// CPU twin of `ref_get_rows`.
    public static func getRows(weight: Native.QuantTensor, tokens: [Int32]) -> [Float] {
        tokens.flatMap { Native.ReferenceDequant.row(of: weight, Int($0)) }
    }
}

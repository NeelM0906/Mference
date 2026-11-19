import Foundation
import Metal

extension Native {
    /// Pure-Swift twins of the reference kernels, in the kernels' arithmetic
    /// order (F32, sequential accumulation). Tests compare the Metal
    /// references with these; Phase 1 workers may use them as a second
    /// oracle on small shapes.
    public enum ReferenceCPU {
        public static func silu(_ x: Float) -> Float { x / (1 + exp(-x)) }
        public static func sigmoid(_ x: Float) -> Float { 1 / (1 + exp(-x)) }

        /// `ref_rms_norm` over `rows x cols`.
        public static func rmsNorm(_ x: [Float], rows: Int, cols: Int, weight: [Float], eps: Float) -> [Float] {
            var y = [Float](repeating: 0, count: rows * cols)
            for r in 0..<rows {
                var sum: Float = 0
                for i in 0..<cols { sum += x[r * cols + i] * x[r * cols + i] }
                let mean = sum / Float(cols)
                let scale = 1 / (mean + eps).squareRoot()
                for i in 0..<cols { y[r * cols + i] = (x[r * cols + i] * scale) * weight[i] }
            }
            return y
        }

        public static func swiglu(gate: [Float], up: [Float]) -> [Float] {
            zip(gate, up).map { silu($0) * $1 }
        }
    }

    /// The reference `GlueOps`: one thread per row (RMSNorm) or element.
    public final class ReferenceGlueOps: GlueOps {
        let context: MetalContext
        private var eps: Float = 0
        private var rmsNorm: MTLComputePipelineState?
        private var add: MTLComputePipelineState?
        private var swiglu: MTLComputePipelineState?
        private var copy: MTLComputePipelineState?

        public init(context: MetalContext) {
            self.context = context
        }

        public func prepare(config: Qwen35Config) throws {
            eps = config.rmsNormEpsilon
            rmsNorm = try context.modulePipeline(module: ReferenceMetal.module, name: "ref_rms_norm")
            add = try context.modulePipeline(module: ReferenceMetal.module, name: "ref_add")
            swiglu = try context.modulePipeline(module: ReferenceMetal.module, name: "ref_swiglu")
            copy = try context.modulePipeline(module: ReferenceMetal.module, name: "ref_copy_bytes")
        }

        struct RowParams {
            var rows: UInt32
            var cols: UInt32
            var aStride: UInt32
            var bStride: UInt32
            var outStride: UInt32
            var eps: Float
            var weightType: UInt32
        }

        private func pipelines() throws -> (MTLComputePipelineState, MTLComputePipelineState,
                                            MTLComputePipelineState, MTLComputePipelineState) {
            guard let rmsNorm, let add, let swiglu, let copy else {
                throw NativeError.resource("ReferenceGlueOps.prepare not called")
            }
            return (rmsNorm, add, swiglu, copy)
        }

        public func encodeRMSNorm(_ encoder: MTLComputeCommandEncoder, input: TensorRef, weight: QuantTensor,
                                  output: TensorRef) throws {
            let p = try pipelines().0
            guard weight.type == .f32 || weight.type == .f16, weight.layout == .gguf, weight.rows == 1,
                  weight.cols == input.cols else {
                throw NativeError.invalidShape("RMSNorm weight \(weight.name) \(weight.type) \(weight.shape)")
            }
            try ReferenceMetal.checkF32(input, rows: input.rows, cols: input.cols, "RMSNorm input")
            try ReferenceMetal.checkF32(output, rows: input.rows, cols: input.cols, "RMSNorm output")
            encoder.setBuffer(input.buffer, offset: input.offset, index: 0)
            encoder.setBuffer(weight.buffer, offset: weight.offset, index: 1)
            encoder.setBuffer(output.buffer, offset: output.offset, index: 2)
            ReferenceMetal.setBytes(encoder, RowParams(rows: UInt32(input.rows), cols: UInt32(input.cols),
                                                       aStride: UInt32(input.rowStride), bStride: 0,
                                                       outStride: UInt32(output.rowStride), eps: eps,
                                                       weightType: weight.type.rawValue), index: 3)
            ReferenceMetal.dispatch(encoder, p, width: input.rows, height: 1)
        }

        public func encodeAdd(_ encoder: MTLComputeCommandEncoder, a: TensorRef, b: TensorRef,
                              output: TensorRef) throws {
            try binary(encoder, try pipelines().1, a: a, b: b, output: output, what: "add")
        }

        public func encodeSwiGLU(_ encoder: MTLComputeCommandEncoder, gate: TensorRef, up: TensorRef,
                                 output: TensorRef) throws {
            try binary(encoder, try pipelines().2, a: gate, b: up, output: output, what: "SwiGLU")
        }

        private func binary(_ encoder: MTLComputeCommandEncoder, _ p: MTLComputePipelineState, a: TensorRef,
                            b: TensorRef, output: TensorRef, what: String) throws {
            try ReferenceMetal.checkF32(a, rows: a.rows, cols: a.cols, "\(what) a")
            try ReferenceMetal.checkF32(b, rows: a.rows, cols: a.cols, "\(what) b")
            try ReferenceMetal.checkF32(output, rows: a.rows, cols: a.cols, "\(what) output")
            encoder.setBuffer(a.buffer, offset: a.offset, index: 0)
            encoder.setBuffer(b.buffer, offset: b.offset, index: 1)
            encoder.setBuffer(output.buffer, offset: output.offset, index: 2)
            ReferenceMetal.setBytes(encoder, RowParams(rows: UInt32(a.rows), cols: UInt32(a.cols),
                                                       aStride: UInt32(a.rowStride), bStride: UInt32(b.rowStride),
                                                       outStride: UInt32(output.rowStride), eps: 0,
                                                       weightType: 0), index: 3)
            ReferenceMetal.dispatch(encoder, p, width: a.cols, height: a.rows)
        }

        public func encodeCopy(_ encoder: MTLComputeCommandEncoder, source: TensorRef,
                               destination: TensorRef) throws {
            let p = try pipelines().3
            guard source.dtype == destination.dtype, source.rows == destination.rows,
                  source.cols == destination.cols else {
                throw NativeError.invalidShape("copy \(source.dtype) \(source.rows)x\(source.cols) to "
                    + "\(destination.dtype) \(destination.rows)x\(destination.cols)")
            }
            let rowBytes = source.cols * source.dtype.byteSize
            encoder.setBuffer(source.buffer, offset: source.offset, index: 0)
            encoder.setBuffer(destination.buffer, offset: destination.offset, index: 2)
            ReferenceMetal.setBytes(encoder, RowParams(rows: UInt32(source.rows), cols: UInt32(rowBytes),
                                                       aStride: UInt32(source.rowStride), bStride: 0,
                                                       outStride: UInt32(destination.rowStride), eps: 0,
                                                       weightType: 0), index: 3)
            ReferenceMetal.dispatch(encoder, p, width: rowBytes, height: source.rows)
        }
    }
}

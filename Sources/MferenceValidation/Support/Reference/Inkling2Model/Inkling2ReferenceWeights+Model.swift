import Foundation
import Metal
import Mference

extension Inkling2ReferenceWeights {
    /// The CPU reference's view of a loaded `inkling2Loop` install: every
    /// matrix an `Inkling2BufferMatrix` over the install's own mapped bytes
    /// (the bytes the GPU runner reads), the vectors copied (norms, clips,
    /// convolution taps, queries: at most 16K floats each); the experts over
    /// each layer file mapped read-only (`inkling2MappedExpertLayer`), whose
    /// pages fault in only where the reference reads.
    public init(model: Model) throws {
        let c = model.config
        let d = c.hiddenSize
        func view(_ t: TensorView, rows: Int, cols: Int,
                  _ format: Inkling2BufferMatrix.Format) -> Inkling2BufferMatrix {
            Inkling2BufferMatrix(buffer: t.buffer, offset: Int(t.offset), rows: rows, cols: cols, format: format)
        }
        func q8(_ t: TensorView, rows: Int, cols: Int) -> Inkling2BufferMatrix { view(t, rows: rows, cols: cols, .q8_0) }
        func bf16(_ t: TensorView, rows: Int, cols: Int) -> Inkling2BufferMatrix { view(t, rows: rows, cols: cols, .bf16) }
        func f32m(_ t: TensorView, rows: Int, cols: Int) -> Inkling2BufferMatrix { view(t, rows: rows, cols: cols, .f32) }
        func vector(_ t: TensorView, _ count: Int) -> [Float] {
            precondition(Int(t.length) == count * 4, "F32 vector of \(count) values is \(t.length) bytes")
            return Inkling2BufferMatrix.floats(t.buffer, offset: Int(t.offset), count: count)
        }
        func rows(_ t: TensorView, _ n: Int) -> [[Float]] {
            (0..<n).map { Inkling2BufferMatrix.floats(t.buffer, offset: Int(t.offset) + $0 * d * 4, count: d) }
        }

        let q = c.numHeads * c.headDim, kv = c.numKVHeads * c.headDim
        let rel = c.numHeads * c.relativePosition.dRel, k = c.sconvKernelSize
        let taps = c.numKVHeads * k, ff = c.moeIntermediateSize, rank = c.inkling2.loopGateRank
        let layout = try model.inkling2ExpertLayout()
        var layers: [Inkling2ReferenceLayerWeights] = []
        var loops: [Inkling2ReferenceLoopWeights] = []
        for L in 0..<c.numLayers {
            let global = c.fullAttentionLayerMask[L] == 1
            let slab = try model.inkling2MappedExpertLayer(L)
            let blob = layout.layers[L]
            let nvfp4 = blob.format == .nvfp4
            let gateUpScales = nvfp4 ? vector(try model.inkling2ExpertGateUpScale(layer: L), c.numExperts) : []
            let downScales = nvfp4 ? vector(try model.inkling2ExpertDownScale(layer: L), c.numExperts) : []
            func expert(_ e: Int, down: Bool) -> Inkling2BufferMatrix {
                let at = e * slab.expertStride + Int(down ? blob.down.offset : blob.gateUp.offset)
                let format: Inkling2BufferMatrix.Format = nvfp4
                    ? .nvfp4(scale: down ? downScales[e] : gateUpScales[e]) : .q8_0
                return Inkling2BufferMatrix(buffer: slab.buffer, offset: at, rows: down ? d : 2 * ff,
                                            cols: down ? ff : d, format: format)
            }
            let resQueries = L == 0 ? [] : rows(try model.inkling2AttnResQuery(layer: L), global ? 3 : 1)
            layers.append(Inkling2ReferenceLayerWeights(
                wq: q8(try model.inkling2AttnQ(layer: L), rows: q, cols: d),
                wk: q8(try model.inkling2AttnK(layer: L), rows: kv, cols: d),
                wv: q8(try model.inkling2AttnV(layer: L), rows: kv, cols: d),
                wo: q8(try model.inkling2AttnOutput(layer: L), rows: d, cols: q),
                outputGate: q8(try model.inkling2AttnGate(layer: L), rows: q, cols: d),
                rel: q8(try model.inkling2AttnRel(layer: L), rows: rel, cols: d),
                sinkProj: bf16(try model.inkling2AttnSinkProj(layer: L), rows: c.numHeads, cols: d),
                qClipScale: vector(try model.inkling2AttnQClip(layer: L), q),
                relClipScale: vector(try model.inkling2AttnRelClip(layer: L), rel),
                relBias: vector(try model.inkling2AttnRelBias(layer: L), c.relativePosition.extent * c.relativePosition.dRel),
                kConv: vector(try model.inkling2AttnKConv(layer: L), kv * k),
                vConv: vector(try model.inkling2AttnVConv(layer: L), kv * k),
                attnPostConv: vector(try model.inkling2AttnPostConv(layer: L), d * k),
                ffnPostConv: vector(try model.inkling2FFNPostConv(layer: L), d * k),
                attnNorm: global ? nil : vector(try model.inkling2AttnNorm(layer: L), d),
                qInputNorm: global ? vector(try model.inkling2AttnQInputNorm(layer: L), d) : nil,
                kInputNorm: global ? vector(try model.inkling2AttnKInputNorm(layer: L), d) : nil,
                vInputNorm: global ? vector(try model.inkling2AttnVInputNorm(layer: L), d) : nil,
                kConvProj: global ? bf16(try model.inkling2AttnKConvProj(layer: L), rows: taps, cols: d) : nil,
                vConvProj: global ? bf16(try model.inkling2AttnVConvProj(layer: L), rows: taps, cols: d) : nil,
                attnResQueries: resQueries,
                ffnResQuery: vector(try model.inkling2FFNResQuery(layer: L), d),
                ffnNorm: vector(try model.inkling2FFNNorm(layer: L), d),
                router: f32m(try model.inkling2Router(layer: L), rows: c.numExperts, cols: d),
                routerBias: vector(try model.inkling2RouterSelectionBias(layer: L), c.numExperts),
                expertGateUp: (0..<c.numExperts).map { expert($0, down: false) },
                expertDown: (0..<c.numExperts).map { expert($0, down: true) }))
            if c.inkling2.layerIsLooped(L) {
                let gates = Inkling2MessageGate.allCases
                loops.append(Inkling2ReferenceLoopWeights(
                    messageQueries: rows(try model.inkling2LoopMessageQuery(layer: L), 4),
                    controlDown: try gates.map { f32m(try model.inkling2LoopGateDown(layer: L, $0), rows: rank, cols: 2 * d) },
                    controlGain: try gates.map { f32m(try model.inkling2LoopGateGain(layer: L, $0), rows: d, cols: rank) },
                    controlForget: try gates.map { f32m(try model.inkling2LoopGateForget(layer: L, $0), rows: d, cols: rank) }))
            }
        }
        self.init(
            embedding: q8(try model.inkling2Embedding, rows: c.vocabSize, cols: d),
            embeddingNorm: vector(try model.inkling2EmbeddingNorm, d),
            outputResQuery: vector(try model.inkling2OutputResQuery, d),
            outputNorm: vector(try model.inkling2OutputNorm, d),
            output: q8(try model.inkling2OutputHead, rows: c.vocabSize, cols: d),
            layers: layers, loopLayers: loops)
    }
}

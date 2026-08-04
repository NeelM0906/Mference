import Testing
import Foundation
@testable import Mference

/// Integration check against a real installed Inkling-Small `.gturbo`.
/// Skipped unless `MFERENCE_INKLING_GTURBO` points at one, so the suite stays
/// runnable without the 148 GB checkpoint.
@Suite struct InklingInstalledModelTests {

    @Test func installedManifestValidatesAgainstBaseline() throws {
        guard let path = ProcessInfo.processInfo
            .environment["MFERENCE_INKLING_GTURBO"] else { return }
        let url = URL(fileURLWithPath: path)

        #expect(try ManifestReader.peekFamily(directoryURL: url) == .inklingSmall)

        // `load(directoryURL:expecting:)` cross-checks every arch and quant
        // field against the compiled baseline and throws on any divergence.
        let expected = try #require(ArchConfig.knownArchitectures[.inklingSmall])
        let manifest = try ManifestReader.load(directoryURL: url,
                                               expecting: expected)

        #expect(manifest.arch.numLayers == 42)
        #expect(manifest.arch.family == "inklingSmall")
        #expect(manifest.arch.relDRel == 16)
        #expect(manifest.arch.sconvKernelSize == 4)
        #expect(manifest.arch.numSharedExperts == 2)
        let quant = try #require(manifest.quant)
        #expect(quant.routedExpert.weightBits == 4)
        #expect(quant.attention.weightBits == 4)
        #expect(quant.embedding.weightBits == 4)
        #expect(quant.routedExpert.groupSize == 64)
        #expect(quant.routedExpert.scheme.lowercased() == "affine")
        #expect(manifest.expertsPerLayer == 256)
    }
}

extension InklingInstalledModelTests {
    /// Bring-up probe: CPU-decode the scales/biases of the L2 attention
    /// projections straight from the resident file. Run with
    /// MFERENCE_INKLING_GTURBO pointing at a real install.
    @Test func dumpWqDuScales() throws {
        guard let path = ProcessInfo.processInfo
            .environment["MFERENCE_INKLING_GTURBO"] else { return }
        let ctx = try MetalContext()
        let expected = try #require(ArchConfig.knownArchitectures[.inklingSmall])
        let model = try Model.load(directoryURL: URL(fileURLWithPath: path),
                                   device: ctx.device,
                                   expecting: expected)
        for (label, view) in [
            ("wq_du", try model.inklingWqDu(layer: 2)),
            ("wk_dv", try model.inklingWkDv(layer: 2)),
            ("wo_ud", try model.inklingWoUd(layer: 2)),
        ] {
            print("[dump] \(label) off=\(view.offset) len=\(view.length) "
                + "scaleOff=\(view.scaleOffset) scaleLen=\(view.scaleLength) "
                + "biasOff=\(view.biasOffset) biasLen=\(view.biasLength) "
                + "dtype=\(view.dtype) shape=\(view.shape)")
            let scaleCount = Int(view.scaleLength) / 2
            let sp = view.buffer.contents().advanced(by: Int(view.scaleOffset))
                .assumingMemoryBound(to: UInt16.self)
            var bad = 0
            var mx: Float = 0
            for i in 0..<scaleCount {
                let v = Quantization.bf16ToFloat(sp[i])
                if !v.isFinite { bad += 1 } else { mx = max(mx, abs(v)) }
            }
            let first = (0..<8).map { Quantization.bf16ToFloat(sp[$0]) }
            print("[dump] \(label) scales n=\(scaleCount) nonfinite=\(bad) "
                + "maxAbs=\(mx) first=\(first)")
        }
        for L in [7, 20, 28] {
            for (name, view) in [
                ("attn_sconv", try model.inklingAttnSconv(layer: L)),
                ("mlp_sconv", try model.inklingMlpSconv(layer: L)),
                ("k_sconv", try model.inklingKSconv(layer: L)),
            ] {
                let n = Int(view.length) / 2
                let p = view.buffer.contents().advanced(by: Int(view.offset))
                    .assumingMemoryBound(to: UInt16.self)
                var mx: Float = 0
                var sum: Float = 0
                for i in 0..<n {
                    let v = Quantization.bf16ToFloat(p[i])
                    mx = max(mx, abs(v)); sum += abs(v)
                }
                print("[taps] L\(L) \(name) n=\(n) maxAbs=\(mx) meanAbs=\(sum / Float(n)) "
                    + "first8=\((0..<8).map { Quantization.bf16ToFloat(p[$0]) })")
            }
        }
    }
}

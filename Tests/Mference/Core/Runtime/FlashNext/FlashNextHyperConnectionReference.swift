import Foundation
@testable import Mference

/// `Qwen4ExpTextGatedResidual` and the group RMSNorm under it, in float32,
/// extracted so the Metal hyper-connection kernels can be gated against the same
/// arithmetic the CPU reference forward runs.
///
/// Like `FlashNextRouterReference` this is a transcription of
/// `FlashNextReferenceRunner` (`groupRMSNorm` and `gatedResidual`), not a
/// re-derivation, and `FlashNextHyperConnectionReferenceTieBackTests` discharges
/// the transcription by replaying the runner's own captures through it.
enum FlashNextHyperConnectionReference {

    struct Geometry {
        let hidden: Int
        let hcCount: Int
        let lowRank: Int
        let eps: Float
        var bundle: Int { hidden * hcCount }
    }

    struct Weights {
        let norm: [Float]        // [bundle]
        let mixDown: [Float]     // [lowRank, bundle]
        let mixUp: [Float]       // [bundle, lowRank]
        let inject: [Float]?     // [hcCount, bundle]; nil for the global mixer
    }

    struct Mix {
        let normed: [Float]      // [rows * bundle]
        let mixed: [Float]       // [rows * hidden]
        let inject: [Float]?     // [rows * hcCount]
    }

    /// Each of the `hcCount` streams normalized over its own `hidden` channels,
    /// against its own slice of the bundle-wide weight.
    static func groupRMSNorm(_ x: [Float], offset: Int,
                             weight: [Float], g: Geometry) -> [Float] {
        var out = [Float](repeating: 0, count: g.bundle)
        for j in 0..<g.hcCount {
            let base = offset + j * g.hidden
            var ms: Float = 0
            for d in 0..<g.hidden { ms += x[base + d] * x[base + d] }
            let inv = 1 / sqrtf(ms / Float(g.hidden) + g.eps)
            for d in 0..<g.hidden {
                out[j * g.hidden + d] = x[base + d] * inv * weight[j * g.hidden + d]
            }
        }
        return out
    }

    /// Three separate `/hc_count` divisions — before the SiLU, in the mean over
    /// streams, and before the injection sigmoid — and the mix and inject both
    /// read the NORMED stream.
    static func gatedResidual(_ hyper: [Float], _ w: Weights,
                              rows: Int, g: Geometry) -> Mix {
        var normedAll = [Float](repeating: 0, count: rows * g.bundle)
        var mixed = [Float](repeating: 0, count: rows * g.hidden)
        var inject = w.inject == nil
            ? nil : [Float](repeating: 0, count: rows * g.hcCount)

        for t in 0..<rows {
            let normed = groupRMSNorm(hyper, offset: t * g.bundle,
                                      weight: w.norm, g: g)
            for i in 0..<g.bundle { normedAll[t * g.bundle + i] = normed[i] }

            var down = FlashNextRouterReference.matVec(
                w.mixDown, rows: g.lowRank, cols: g.bundle, x: normed)
            for i in 0..<g.lowRank { down[i] = silu(down[i] / Float(g.hcCount)) }
            let up = FlashNextRouterReference.matVec(
                w.mixUp, rows: g.bundle, cols: g.lowRank, x: down)

            for d in 0..<g.hidden {
                var acc: Float = 0
                for j in 0..<g.hcCount {
                    let i = j * g.hidden + d
                    acc += sigmoid(up[i]) * normed[i]
                }
                mixed[t * g.hidden + d] = acc / Float(g.hcCount)
            }

            if let injectWeight = w.inject {
                let raw = FlashNextRouterReference.matVec(
                    injectWeight, rows: g.hcCount, cols: g.bundle, x: normed)
                for j in 0..<g.hcCount {
                    inject![t * g.hcCount + j] = 2 * sigmoid(raw[j] / Float(g.hcCount))
                }
            }
        }
        return Mix(normed: normedAll, mixed: mixed, inject: inject)
    }

    /// `hyper + flatten(block_out[None, H] * inject[hc, None])`.
    static func injectBlock(_ hyper: [Float], block: [Float], inject: [Float],
                            rows: Int, g: Geometry) -> [Float] {
        var out = hyper
        for t in 0..<rows {
            for j in 0..<g.hcCount {
                let gate = inject[t * g.hcCount + j]
                let base = t * g.bundle + j * g.hidden
                for d in 0..<g.hidden {
                    out[base + d] += gate * block[t * g.hidden + d]
                }
            }
        }
        return out
    }

    static func silu(_ x: Float) -> Float { x / (1 + expf(-x)) }
    static func sigmoid(_ x: Float) -> Float { 1 / (1 + expf(-x)) }
}

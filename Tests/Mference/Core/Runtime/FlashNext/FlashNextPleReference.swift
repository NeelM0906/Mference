import Foundation
@testable import Mference

/// `Qwen4ExpTextPLELayer`'s mixing block in float32, extracted from
/// `FlashNextReferenceRunner.pleBlock` so the Metal PLE kernels can be gated
/// against the same arithmetic.
///
/// Stateful, because the conv is: `(kernel - 1) * dilation` rows of the normed
/// gated value carry between calls, which is exactly what makes a cached decode
/// step equal to the corresponding row of a full prefill.
///
/// `FlashNextPleReferenceTieBackTests` discharges the transcription by replaying
/// the runner's own captures — prefill and cached decode — through this type.
final class FlashNextPleReference {

    struct Weights {
        let keyProj: [Float]      // [bundle, hidden]
        let valueProj: [Float]    // [hidden, hidden]
        let conv: [Float]         // [bundle, kernel]
        let normKey: [Float]      // [bundle]
        let normQuery: [Float]    // [bundle]
        let normConv: [Float]     // [bundle]
    }

    private let g: FlashNextHyperConnectionReference.Geometry
    private let convKernel: Int
    private let dilation: Int

    /// `[stateLength * bundle]` of the NORMED gated value.
    private(set) var convState: [Float]

    var stateLength: Int { (convKernel - 1) * dilation }

    init(geometry: FlashNextHyperConnectionReference.Geometry,
         convKernel: Int, dilation: Int) {
        self.g = geometry
        self.convKernel = convKernel
        self.dilation = dilation
        self.convState = [Float](repeating: 0,
                                 count: (convKernel - 1) * dilation
                                     * geometry.hidden * geometry.hcCount)
    }

    func reset() {
        convState = [Float](repeating: 0, count: stateLength * g.bundle)
    }

    /// The PLE output for `rows` tokens. `hyper` is the RAW stream before this
    /// block's add; `embeds` is the gathered n-gram embedding `[rows * hidden]`.
    func mix(hyper: [Float], embeds: [Float], w: Weights, rows: Int) -> [Float] {
        var gatedValue = [Float](repeating: 0, count: rows * g.bundle)
        var normedGated = [Float](repeating: 0, count: rows * g.bundle)

        for t in 0..<rows {
            let e = Array(embeds[(t * g.hidden)..<((t + 1) * g.hidden)])
            let keyProjected = FlashNextRouterReference.matVec(
                w.keyProj, rows: g.bundle, cols: g.hidden, x: e)
            let keyNormed = FlashNextHyperConnectionReference.groupRMSNorm(
                keyProjected, offset: 0, weight: w.normKey, g: g)
            let value = FlashNextRouterReference.matVec(
                w.valueProj, rows: g.hidden, cols: g.hidden, x: e)
            let queryNormed = FlashNextHyperConnectionReference.groupRMSNorm(
                hyper, offset: t * g.bundle, weight: w.normQuery, g: g)

            for j in 0..<g.hcCount {
                var dot: Float = 0
                for d in 0..<g.hidden {
                    dot += keyNormed[j * g.hidden + d] * queryNormed[j * g.hidden + d]
                }
                var value2 = dot / sqrtf(Float(g.hidden))
                // Signed sqrt with the 1e-6 floor on the MAGNITUDE before the
                // root, so |g'| >= 1e-3 unless the dot is exactly zero.
                let sign: Float = value2 > 0 ? 1 : (value2 < 0 ? -1 : 0)
                value2 = sqrtf(max(abs(value2), 1e-6)) * sign
                let gate = FlashNextHyperConnectionReference.sigmoid(value2)
                for d in 0..<g.hidden {
                    gatedValue[t * g.bundle + j * g.hidden + d] = gate * value[d]
                }
            }
            let normed = FlashNextHyperConnectionReference.groupRMSNorm(
                gatedValue, offset: t * g.bundle, weight: w.normConv, g: g)
            for i in 0..<g.bundle { normedGated[t * g.bundle + i] = normed[i] }
        }

        var padded = convState
        padded.append(contentsOf: normedGated)
        var out = gatedValue
        for t in 0..<rows {
            for c in 0..<g.bundle {
                var acc: Float = 0
                for j in 0..<convKernel {
                    let row = t + stateLength - (convKernel - 1 - j) * dilation
                    acc += w.conv[c * convKernel + j] * padded[row * g.bundle + c]
                }
                out[t * g.bundle + c] += FlashNextHyperConnectionReference.silu(acc)
            }
        }
        convState = Array(padded[(padded.count - stateLength * g.bundle)...])
        return out
    }
}

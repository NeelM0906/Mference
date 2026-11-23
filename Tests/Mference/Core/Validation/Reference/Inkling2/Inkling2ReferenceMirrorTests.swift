import Foundation
import Testing
import MferenceValidationSupport

/// The CPU reference against a second forward pass laid out like the
/// transformers reference (`Inkling2TransformersMirror`). The two share only
/// the separately tested small operations, so this pins the wiring: the loop
/// restart, the message bank, which message feeds which controller, the
/// block-boundary rule, the depth mixes, and the cached (incremental) state
/// against whole-sequence attention and convolution.
@Suite("Inkling2ReferenceMirror")
struct Inkling2ReferenceMirrorTests {

    private func maxAbsDiff(_ a: [[Float]], _ b: [[Float]]) -> Float {
        precondition(a.count == b.count)
        var m: Float = 0
        for (x, y) in zip(a, b) {
            precondition(x.count == y.count)
            for i in 0..<x.count { m = max(m, abs(x[i] - y[i])) }
        }
        return m
    }

    @Test func referenceMatchesTheTransformersLayout() throws {
        // 13 tokens: past the window (8), the bias extent (12) and the log floor (4).
        let weights = Inkling2ToyCheckpoint.weights()
        let tokens = Inkling2ToyCheckpoint.tokens(count: 13)

        let model = try Inkling2ReferenceModel(config: .toy, weights: weights)
        model.captureTrace = true
        // Two chunks, so the cached path crosses a chunk boundary.
        var logits = model.forward(tokens: Array(tokens[..<7]), logits: .all)
        let firstTrace = try #require(model.lastTrace)
        logits += model.forward(tokens: Array(tokens[7...]), logits: .all)
        let secondTrace = try #require(model.lastTrace)

        var mirror = Inkling2TransformersMirror(c: .toy, w: weights)
        let mirrorLogits = mirror.forward(tokens: tokens)

        #expect(mirror.visitOutputs.count == 48)
        for visit in 0..<48 {
            let cached = firstTrace.visitOutputs[visit] + secondTrace.visitOutputs[visit]
            let diff = maxAbsDiff(cached, mirror.visitOutputs[visit])
            #expect(diff < 1e-4, "visit \(visit) (layer \(Inkling2ReferenceConfig.toy.schedule[visit].layer)) differs by \(diff)")
        }
        let diff = maxAbsDiff(logits, mirrorLogits)
        #expect(diff < 1e-4, "logits differ by \(diff)")
    }
}

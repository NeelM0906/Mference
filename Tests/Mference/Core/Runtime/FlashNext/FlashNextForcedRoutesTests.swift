import Foundation
import Testing
@testable import Mference

/// The U5 route-forcing diagnostic's file reader and weights: off without the
/// variable, strict about the file, and the select kernel's softmax over the
/// forced set.
@Suite struct FlashNextForcedRoutesTests {
    static func data(_ values: [Int32]) -> Data {
        values.withUnsafeBytes { Data($0) }
    }

    @Test func offWithoutTheVariable() throws {
        #expect(try FlashNextForcedRoutes.fromEnvironment(layers: 2, topK: 3, numExperts: 8, environment: [:]) == nil)
    }

    @Test func readsPositionMajorRows() throws {
        // 2 positions x 2 layers x top-3
        let r = try FlashNextForcedRoutes(data: Self.data([0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 3, 5]),
                                          layers: 2, topK: 3, numExperts: 8)
        #expect(r.positions == 2)
        #expect(r.ids(position: 0, layer: 1) == [3, 4, 5])
        #expect(r.ids(position: 1, layer: 0) == [6, 7, 0])
        #expect(r.ids(position: 2, layer: 0) == nil)
        #expect(!r.covers(position: 2))
    }

    @Test func refusesMalformedFiles() {
        #expect(throws: FlashNextForcedRoutes.LoadError.sizeMismatch(bytes: 20, rowBytes: 24)) {
            try FlashNextForcedRoutes(data: Self.data([0, 1, 2, 3, 4]), layers: 2, topK: 3, numExperts: 8)
        }
        #expect(throws: FlashNextForcedRoutes.LoadError.expertOutOfRange(8)) {
            try FlashNextForcedRoutes(data: Self.data([0, 1, 8]), layers: 1, topK: 3, numExperts: 8)
        }
        #expect(throws: FlashNextForcedRoutes.LoadError.duplicateExpert(position: 0, layer: 0)) {
            try FlashNextForcedRoutes(data: Self.data([1, 1, 2]), layers: 1, topK: 3, numExperts: 8)
        }
    }

    @Test func weightsAreTheSoftmaxOverTheForcedSet() {
        let logits: [Float] = [0, 1, 2, 3, -1]
        let w = logits.withUnsafeBufferPointer { FlashNextForcedRoutes.weights(logits: $0.baseAddress!, ids: [3, 1]) }
        let e1 = Foundation.exp(-2.0)
        #expect(abs(Double(w[0]) - 1 / (1 + e1)) < 1e-7)
        #expect(abs(Double(w[1]) - e1 / (1 + e1)) < 1e-7)
    }
}

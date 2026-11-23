import Testing
import MferenceValidationSupport

/// `RelError.computeStrict`: `compute` on finite inputs, `.infinity` as soon as
/// either input holds a NaN or an infinity, at any position.
@Suite struct RelErrorStrictTests {
    static let reference: [Float] = [1, -2, 0.5, 4]

    @Test func finiteInputsGiveComputesValue() {
        let actual: [Float] = [1.001, -2, 0.5, 3.99]
        let strict = RelError.computeStrict(actual: actual, reference: Self.reference)
        #expect(strict == RelError.compute(actual: actual, reference: Self.reference))
        #expect(strict > 0 && strict < 1e-2)
        #expect(RelError.computeStrict(actual: Self.reference, reference: Self.reference) == 0)
        #expect(RelError.computeStrict(actual: [], reference: []) == 0)
    }

    @Test(arguments: [Float.nan, .infinity, -.infinity])
    func nonFiniteActualIsCaught(value: Float) {
        for i in Self.reference.indices {
            var actual = Self.reference
            actual[i] = value
            #expect(RelError.computeStrict(actual: actual, reference: Self.reference) == .infinity,
                    "actual[\(i)] = \(value)")
        }
    }

    @Test(arguments: [Float.nan, .infinity, -.infinity])
    func nonFiniteReferenceIsCaught(value: Float) {
        for i in Self.reference.indices {
            var reference = Self.reference
            reference[i] = value
            #expect(RelError.computeStrict(actual: Self.reference, reference: reference) == .infinity,
                    "reference[\(i)] = \(value)")
        }
    }
}

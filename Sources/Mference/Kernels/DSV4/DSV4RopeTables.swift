import Foundation

/// Per-pair inverse-frequency tables for DeepSeek-V4's interleaved partial
/// RoPE. The upstream reference keys rope parameters by rope type: `main`
/// (sliding-window layers, plain rotation at `ropeTheta`) and `compress`
/// (CSA/HCA layers, their compressors, and the lightning indexer, at
/// `compressRopeTheta` with YaRN frequency correction and attention_factor
/// forced to 1.0 — the corrected frequencies apply at every position and
/// cos/sin are never rescaled).
enum DSV4RopeTables {

    /// Plain RoPE inverse frequencies: `theta^(-2i/dim)` for pair index i.
    static func plainInvFreq(theta: Double, ropeDim: Int) -> [Float] {
        (0..<(ropeDim / 2)).map { i in
            Float(pow(theta, -2.0 * Double(i) / Double(ropeDim)))
        }
    }

    /// YaRN-corrected inverse frequencies (transformers
    /// `_compute_yarn_parameters` with `truncate == true`): interpolated by
    /// `factor` below the beta_slow wavelength, unchanged above the
    /// beta_fast wavelength, linear ramp between.
    static func yarnInvFreq(theta: Double, ropeDim: Int,
                            factor: Double,
                            originalMaxPositions: Int,
                            betaFast: Double,
                            betaSlow: Double) -> [Float] {
        let dim = Double(ropeDim)
        let half = ropeDim / 2

        func correctionDim(_ numRotations: Double) -> Double {
            dim * log(Double(originalMaxPositions) / (numRotations * 2.0 * .pi))
                / (2.0 * log(theta))
        }
        let low = max(correctionDim(betaFast).rounded(.down), 0)
        var high = min(correctionDim(betaSlow).rounded(.up), dim - 1)
        if high == low { high += 0.001 }

        return (0..<half).map { i in
            let posFreq = pow(theta, 2.0 * Double(i) / dim)
            let extrapolation = 1.0 / posFreq
            let interpolation = 1.0 / (factor * posFreq)
            let ramp = min(max((Double(i) - low) / (high - low), 0), 1)
            let extrapolationFactor = 1.0 - ramp
            return Float(interpolation * (1.0 - extrapolationFactor)
                + extrapolation * extrapolationFactor)
        }
    }
}

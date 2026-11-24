import Foundation

/// Inkling 2 Mini attention-side shape, from the vendor `text_config` and the
/// upstream GGUF metadata (llama.cpp `inkling2`). The references take every
/// shape as a parameter; these are the production values.
public enum Inkling2RefShape {
    public static let hidden = 2304
    public static let numQHeads = 24
    public static let numKVHeads = 8
    public static let headDim = 128
    public static let dRel = 16
    /// Distances 0 ..< relExtent carry a bias; farther keys get exactly 0.
    public static let relExtent = 1024
    public static let slidingWindow = 1024
    public static let convKernel = 4
    /// `qk_l2_clip_threshold`, sqrt(512).
    public static let qkClip: Float = 22.627416997969522
    /// `rel_l2_clip_threshold`.
    public static let relClip: Float = 8.0
    public static let rmsEps: Float = 1e-6
    /// Clip denominators add this to the L2 norm (`norm + 1e-6`).
    public static let clipEps: Float = 1e-6
    public static let logScaleFloor = 8192
    public static let logScaleAlpha: Float = 0.1
    public static let loopGateRank = 64
    public static let storedLayers = 24
    public static let loopStart = 6
    public static let loopEnd = 18
    public static let loopPasses = 3

    /// Layers 5, 11, 17 and 23 attend globally; the other 20 use the window.
    public static func isGlobal(layer: Int) -> Bool { (layer + 1) % 6 == 0 }
}

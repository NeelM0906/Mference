import Foundation
import Metal

/// The Inkling 2 attention-side shader module (`Metal/Inkling2/inkling2_attention.metal`),
/// compiled on demand rather than into the shared runtime library, with its
/// own pipeline cache. One instance per `MetalContext` is enough; it is safe
/// to share between encoders and threads.
///
/// It finds the source itself, with the same bundle lookup as
/// `MetalContext` (the packaged app's resource bundle first, then
/// `Bundle.module`), so the shared shader tables stay unchanged; the module
/// can move into `MetalContext.shaderSubdirectories` later without changing
/// any caller.
public final class Inkling2KernelLibrary: @unchecked Sendable {
    public static let module = "inkling2_attention"
    static let subdirectory = "Metal/Inkling2"

    public let context: MetalContext
    private let library: MTLLibrary
    private var pipelines: [String: MTLComputePipelineState] = [:]
    private let lock = NSLock()

    public init(context: MetalContext) throws {
        self.context = context
        self.library = try Self.compile(device: context.device)
    }

    private static let resourceBundle: Bundle = {
        if let resources = Bundle.main.resourceURL {
            let packaged = resources.appendingPathComponent("Mference_Mference.bundle")
            if FileManager.default.fileExists(atPath: packaged.path), let bundle = Bundle(url: packaged) {
                return bundle
            }
        }
        return .module
    }()

    static func compile(device: MTLDevice) throws -> MTLLibrary {
        guard let url = resourceBundle.url(forResource: module, withExtension: "metal", subdirectory: subdirectory) else {
            throw MetalError.missingShaderResource(module)
        }
        let source = try String(contentsOf: url, encoding: .utf8)
        let options = MTLCompileOptions()
        options.languageVersion = MetalContext.shaderLanguageVersion
        do {
            return try device.makeLibrary(source: source, options: options)
        } catch {
            throw MetalError.libraryCompileFailed("\(error)")
        }
    }

    public func pipeline(_ name: String) throws -> MTLComputePipelineState {
        lock.lock()
        defer { lock.unlock() }
        if let cached = pipelines[name] { return cached }
        guard let function = library.makeFunction(name: name) else {
            throw MetalError.missingFunction(name)
        }
        let state = try context.device.makeComputePipelineState(function: function)
        pipelines[name] = state
        return state
    }
}

/// Fixed shape of the Inkling 2 Mini attention block (vendor `text_config`).
public enum Inkling2AttentionShape {
    public static let hidden = 2304
    public static let numQHeads = 24
    public static let numKVHeads = 8
    public static let headDim = 128
    public static let dRel = 16
    public static let relExtent = 1024
    public static let slidingWindow = 1024
    public static let convKernel = 4
    public static let qkClip: Float = 22.627416997969522
    public static let relClip: Float = 8.0
    public static let rmsEps: Float = 1e-6
    public static let logScaleFloor = 8192
    public static let logScaleAlpha: Float = 0.1
    /// Q, K, V widths: 24 x 128, 8 x 128, 8 x 128.
    public static let qWidth = numQHeads * headDim
    public static let kvWidth = numKVHeads * headDim
    /// Relative states: 24 x 16.
    public static let relWidth = numQHeads * dRel
    /// Dynamic K/V convolution taps of the global layers: 8 heads x 4.
    public static let dynamicTapWidth = numKVHeads * convKernel

    /// Layers 5, 11, 17 and 23 attend globally; the others use the window.
    public static func isGlobal(layer: Int) -> Bool { (layer + 1) % 6 == 0 }
}

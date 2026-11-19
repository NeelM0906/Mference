import Testing
import Metal
@testable import Mference

/// The runtime compiles its whole shader library from source at startup, at
/// one MSL version. A wrong version either drops the tensor-ops kernels
/// silently (slower, still correct) or fails the library and the runtime.
@Suite struct MetalShaderLanguageTests {

    /// Every supported OS (macOS 26 and later) compiles at MSL 4.0, the
    /// version the MPP tensor-ops prefill kernels need.
    @Test func shadersCompileAtMSL4() {
        #expect(MetalContext.shaderLanguageVersion == .version4_0)
        // `MTLLanguageVersion` encodes `(major << 16) + minor`.
        #expect(MetalContext.shaderLanguageVersion.rawValue == 4 << 16)
    }

    /// The tensor-ops gate is a run-time question about the GPU, not the OS.
    @Test func tensorOpsGateIsTheApple10Family() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        #expect(device.supportsApple10TensorOps == device.supportsFamily(.apple10))
    }

    /// The combined production library builds on this GPU, and the kernels
    /// that use only baseline MSL are all in it. The GDN (gated-DeltaNet)
    /// kernels back the Qwen 3.6 path; naming them rather than counting keeps
    /// the check meaningful as kernels are added.
    @Test func combinedShaderLibraryKeepsTheBaselineKernels() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let library = try MetalContext.compileShaderLibrary(device: device)
        let required: Set = [
            "gdn_conv_mix_decode",
            "gdn_conv_mix_prefill",
            "gdn_conv_tail_update",
            "gdn_delta_step_decode",
            "gdn_delta_step_prefill",
            "gdn_gated_norm",
            "gdn_in_proj_gemv_simd",
            "gdn_qk_norm",
        ]
        let missing = required.subtracting(library.functionNames)
        #expect(missing.isEmpty, "baseline kernels missing: \(missing.sorted())")
    }
}

import Testing
import Metal
@testable import Mference

/// Guards the raw-value lookups in `MetalSDKCompatibility.swift`.
///
/// Those values decide whether shaders compile at MSL 4.0. Get the encoding
/// wrong in one direction and a Metal 4 machine silently drops its tensor-ops
/// kernels — everything still builds and produces correct output, just slower.
/// Get it wrong in the other and the entire shader library fails to compile.
@Suite struct MetalSDKCompatibilityTests {

    /// `MTLLanguageVersion` encodes its raw value as `(major << 16) + minor`.
    /// Pin that against cases the macOS 15 SDK declares, so a wrong assumption
    /// is caught on the floor toolchain, which cannot name MSL 4.0 at all.
    @Test func languageVersionRawValueUsesMajorMinorEncoding() {
        #expect(MTLLanguageVersion(rawValue: 3 << 16) == .version3_0)
        #expect(MTLLanguageVersion(rawValue: (3 << 16) + 2) == .version3_2)
    }

    /// The value handed to `MTLCompileOptions` on macOS 26 must be exactly
    /// MSL 4.0. It is built by raw value, so nothing else type-checks it.
    @Test func msl4ShimCarriesTheMSL4RawValue() {
        #expect(MTLLanguageVersion.msl4_0?.rawValue == 4 << 16)
    }

    /// Metal's enums import as non-frozen, so `init?(rawValue:)` constructs
    /// undeclared values instead of rejecting them. That means nil-ness cannot
    /// be used to detect whether the SDK knows a case, which is why
    /// `MetalContext.shaderLanguageVersion` gates on `#available` instead.
    /// If this ever starts failing, that guard could be simplified — until
    /// then, removing it would hand MSL 4.0 to a macOS 15 Metal compiler.
    @Test func unknownRawValuesAreConstructedNotRejected() {
        #expect(MTLLanguageVersion(rawValue: 99 << 16) != nil)
        #expect(MTLGPUFamily(rawValue: 9_999) != nil)
    }

    // The assertions below name `MTLGPUFamily.apple10` and
    // `MTLLanguageVersion.version4_0` — the two cases this file exists to avoid
    // naming — so they must not reach the macOS 15 SDK, where neither is
    // declared and the test target would fail to compile. `#available` cannot
    // gate that: the cases are absent from the enum, not merely unavailable.
    // `MetalPerformancePrimitives` is macOS 26-only, so importing it is a
    // stand-in for "this SDK knows the Metal 4 symbols".
    #if canImport(MetalPerformancePrimitives)

    /// Pins the raw values against the cases themselves wherever the SDK still
    /// declares them. This is the only assertion that would catch a wrong
    /// constant; every other check here is consistent with any encoding.
    @Test func shimsMatchTheSDKDeclaredCases() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        // `apple10` carries no availability annotation, but `version4_0` is
        // macOS 26+, so naming it needs the deployment-target guard too.
        #expect(MTLGPUFamily.apple10.rawValue == 1010)
        #expect(device.supportsApple10TensorOps == device.supportsFamily(.apple10))
        guard #available(macOS 26.0, iOS 26.0, *) else { return }
        #expect(MTLLanguageVersion.msl4_0 == .version4_0)
    }

    #endif

    /// Every production shader module must build at the MSL 3.2 floor, not
    /// just at 4.0. Kernels that genuinely need tensor operations are expected
    /// to drop out via `__HAVE_TENSOR__`; anything else failing to compile
    /// would take the whole runtime down on macOS 15, since the library is
    /// compiled from source at startup.
    @Test func combinedShaderLibraryCompilesAtTheMSL32Floor() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let library = try MetalContext.compileShaderLibrary(
            device: device,
            languageVersion: .version3_2)
        // The GDN (gated-DeltaNet) kernels back the Qwen 3.6 path and use only
        // baseline MSL, so none of them may be lost at the floor. Naming them
        // rather than counting keeps the check meaningful as kernels are added.
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
        #expect(missing.isEmpty, "GDN kernels missing at MSL 3.2: \(missing.sorted())")
    }

    /// Only the tensor-ops prefill kernel may differ between the floor and
    /// MSL 4.0. Anything else disappearing at 3.2 means a kernel silently
    /// stopped existing on macOS 15.
    @Test func onlyTensorOpsKernelsDropOutAtTheMSL32Floor() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        // `__HAVE_TENSOR__` needs both MSL 4.0 and a GPU with tensor support,
        // so requiring macOS 26 alone would make this fail on the non-Apple10
        // hosted runners, where neither library has the kernel.
        guard #available(macOS 26.0, iOS 26.0, *),
              let msl4 = MTLLanguageVersion.msl4_0,
              device.supportsApple10TensorOps else { return }
        let floor = try MetalContext.compileShaderLibrary(device: device,
                                                          languageVersion: .version3_2)
        let latest = try MetalContext.compileShaderLibrary(device: device,
                                                           languageVersion: msl4)
        let floorNames = Set(floor.functionNames)
        let latestNames = Set(latest.functionNames)
        #expect(latestNames.subtracting(floorNames)
                == ["attention_prefill_full_tensorops_2d_validity_v2"])
        #expect(floorNames.subtracting(latestNames).isEmpty)
    }
}

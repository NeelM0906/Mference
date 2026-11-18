import Foundation
import Testing
@testable import Mference

/// WS4 ruling R1: the Unsloth UD-IQ3_XXS GGUF install is its own family,
/// `qwen38flashnextudiq3xxs`, with its own baseline. U4 wired it onto
/// `FlashNextForwardRunner`, so it is no longer gated.
@Suite struct FlashNextGGUFFamilyTests {
    /// U4: the family runs on `FlashNextForwardRunner`, so the capability gate
    /// no longer lists it. The vendor family stays ungated too.
    @Test func ggufFamilyIsNoLongerGated() throws {
        #expect(ManifestReader.capabilityRefusal(family: "qwen38flashnextudiq3xxs") == nil)
        #expect(ManifestReader.missingRunnerAxes(forRawFamily: "qwen38flashnextudiq3xxs") == nil)
        #expect(ManifestReader.capabilityRefusal(family: "qwen38flashnext") == nil)
    }

    /// The baseline is the vendor Flash-Next baseline field for field, except
    /// the family and the two storage-shaped PLE fields.
    @Test func baselineDiffersFromVendorOnlyInFamilyAndPleStorage() throws {
        let gguf = try #require(ArchConfig.knownArchitectures[.qwen38flashnextudiq3xxs])
        let vendor = try #require(ArchConfig.knownArchitectures[.qwen38flashnext])
        #expect(gguf.family == .qwen38flashnextudiq3xxs)
        #expect(gguf.flashNext.pleNgramShardCount == 1 && gguf.flashNext.pleNgramVocabSizeBase == 0)
        #expect(vendor.flashNext.pleNgramShardCount == 128 && vendor.flashNext.pleNgramVocabSizeBase == 20_000_000)

        func fields(_ value: Any, skipping: Set<String>) -> [String: String] {
            var out: [String: String] = [:]
            for child in Mirror(reflecting: value).children {
                guard let label = child.label, !skipping.contains(label) else { continue }
                out[label] = String(describing: child.value)
            }
            return out
        }
        #expect(fields(gguf, skipping: ["family", "flashNext"]) == fields(vendor, skipping: ["family", "flashNext"]))
        let pleStorage: Set<String> = ["pleNgramShardCount", "pleNgramVocabSizeBase"]
        #expect(fields(gguf.flashNext, skipping: pleStorage) == fields(vendor.flashNext, skipping: pleStorage))
        #expect(ModelFamily.qwen38flashnextudiq3xxs.maximumContext == ModelFamily.qwen38flashnext.maximumContext)
    }
}

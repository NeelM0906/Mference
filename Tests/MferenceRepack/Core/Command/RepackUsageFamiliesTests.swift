import Foundation
import Testing

@testable import MferenceRepackCore

/// `./mference-ui.sh install <family>` validates its argument by reading the
/// `--model <a|b|c>` alternation out of MferenceRepack's own usage text rather
/// than keeping a second copy of the list. That makes the alternation a
/// contract: a source added to `SupportedModelSource.all` but left out of the
/// usage line would be installable by hand and rejected by the launcher.
@Suite struct RepackUsageFamiliesTests {
    @Test func qatSourceIsPinnedAndSeparateFromCurrentGemma() throws {
        let source = try #require(SupportedModelSource.named("gemma4qat"))
        #expect(source.repoID == "mlx-community/gemma-4-26B-A4B-it-qat-q4_0-mlx-aligned")
        #expect(source.revision == "745a97a754ed4b7713163c7d0e9c11da41809e0c")
        #expect(source.sourceIndexSHA256 == "7dbbeef0345505798abcf0ac54434116a48c2f1e7aad828071c17a7a871adfe7")
        #expect(source.modelID == "gemma-4-26b-a4b-it-qat-q4_0-mlx-aligned")
        #expect(source.kind == .preQuantized)
        #expect(source.modelID != SupportedModelSource.gemma4.modelID)
        #expect(SupportedModelSource.default == .gemma4)
    }

    @Test func usageListsEverySupportedModelSourceInOrder() throws {
        let usageSource = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent("Sources/MferenceRepack/Command/main.swift")
        let text = try String(contentsOf: usageSource, encoding: .utf8)
        let marker = try #require(text.range(of: "--model <"),
                                  "usage no longer documents --model")
        let rest = text[marker.upperBound...]
        let close = try #require(rest.firstIndex(of: ">"))
        let listed = rest[..<close].split(separator: "|").map(String.init)
        #expect(listed == SupportedModelSource.all.map(\.name))
    }
}

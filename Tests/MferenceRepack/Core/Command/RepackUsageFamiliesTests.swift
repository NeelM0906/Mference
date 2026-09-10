import Foundation
import Testing

@testable import MferenceRepackCore

/// `./mference-ui.sh install <family>` validates its argument by reading the
/// `--model <a|b|c>` alternation out of MferenceRepack's own usage text rather
/// than keeping a second copy of the list. That makes the alternation a
/// contract: a source added to `SupportedModelSource.all` but left out of the
/// usage line would be installable by hand and rejected by the launcher.
@Suite struct RepackUsageFamiliesTests {
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

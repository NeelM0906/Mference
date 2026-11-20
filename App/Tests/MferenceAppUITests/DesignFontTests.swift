import AppKit
import CryptoKit
import Foundation
import MferenceDesign
import Testing

@Suite struct DesignFontTests {
    @Test func geistRegistersAllFourWeights() {
        #expect(MFFontRegistry.registerBundledFonts() == MFFontRegistry.postScriptNames)
        let members = NSFontManager.shared.availableMembers(ofFontFamily: MFFont.familyName) ?? []
        #expect(members.count >= 4, "Geist family should expose its four bundled weights")
    }

    /// Each file's SHA-256 is the one on its own row of Resources/Fonts/SOURCES.md.
    @Test func bundledFilesMatchRecordedHashes() throws {
        let directory = try #require(MFFontRegistry.fontsDirectory())
        let sources = try String(contentsOf: directory.appendingPathComponent("SOURCES.md"), encoding: .utf8)
        let rows = sources.split(separator: "\n").filter { $0.hasPrefix("| `") }
        for name in MFFontRegistry.fileNames + ["OFL.txt"] {
            let data = try Data(contentsOf: directory.appendingPathComponent(name))
            let hex = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
            let row = try #require(rows.first { $0.hasPrefix("| `\(name)` |") }, "no SOURCES.md row for \(name)")
            #expect(row.contains("`\(hex)`"), "\(name) hash \(hex) is not on its row")
        }
    }
}

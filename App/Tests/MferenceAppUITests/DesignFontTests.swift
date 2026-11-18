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

    /// The files match the hashes in Resources/Fonts/SOURCES.md.
    @Test func bundledFilesMatchRecordedHashes() throws {
        let directory = try #require(MFFontRegistry.fontsDirectory())
        let sources = try String(contentsOf: directory.appendingPathComponent("SOURCES.md"), encoding: .utf8)
        for name in MFFontRegistry.fileNames + ["OFL.txt"] {
            let data = try Data(contentsOf: directory.appendingPathComponent(name))
            let hex = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
            #expect(sources.contains("`\(name)`") && sources.contains(hex), "\(name) hash \(hex) not recorded")
        }
    }
}

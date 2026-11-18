import CoreText
import Foundation

/// Registers the bundled Geist faces (App/Resources/Fonts, decision P6) for
/// this process so `MFFont` resolves to Geist instead of the system fallback.
///
/// Call `registerBundledFonts()` once at launch, before the first view
/// renders. In the app the faces live in `Contents/Resources/Fonts`
/// (`Scripts/bundle.sh`); in `swift test` and `swift run` they are read from
/// the source tree.
public enum MFFontRegistry {
    /// The bundled faces, one per weight the tokens use (400/500/600/700).
    public static let fileNames = ["Geist-Regular.otf", "Geist-Medium.otf", "Geist-SemiBold.otf", "Geist-Bold.otf"]
    /// PostScript names of those faces.
    public static let postScriptNames = ["Geist-Regular", "Geist-Medium", "Geist-SemiBold", "Geist-Bold"]

    /// Lazily initialised once per process (Swift guarantees a thread-safe
    /// one-time initialisation for static stored properties).
    private static let registration: Void = {
        guard let directory = fontsDirectory() else { return }
        for name in fileNames {
            // An already-registered face reports an error; that is fine.
            CTFontManagerRegisterFontsForURL(directory.appendingPathComponent(name) as CFURL, .process, nil)
        }
    }()

    /// Registers the faces once per process. Returns the PostScript names
    /// that are available afterwards (all four when the files were found).
    @discardableResult
    public static func registerBundledFonts() -> [String] {
        _ = registration
        return postScriptNames.filter { CTFontCreateWithName($0 as CFString, 12, nil).postScriptName == $0 }
    }

    /// The first directory that holds every bundled face: the app's
    /// `Resources/Fonts`, then `App/Resources/Fonts` in the source tree.
    public static func fontsDirectory() -> URL? {
        let sourceTree = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // MferenceDesign
            .deletingLastPathComponent()  // Sources
            .deletingLastPathComponent()  // App
            .appendingPathComponent("Resources/Fonts")
        let candidates = [Bundle.main.resourceURL?.appendingPathComponent("Fonts"), sourceTree]
        return candidates.compactMap { $0 }.first { directory in
            fileNames.allSatisfy { FileManager.default.fileExists(atPath: directory.appendingPathComponent($0).path) }
        }
    }
}

private extension CTFont {
    var postScriptName: String { CTFontCopyPostScriptName(self) as String }
}

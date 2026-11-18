import AppKit
import MferenceDesign
import SwiftUI
import Testing

@Suite struct DesignComponentTests {
    @Test func badgeKindsMatchSpecWording() {
        let kinds: [MFBadge.Kind] = [.loaded, .ready, .preview, .later, .installed]
        #expect(kinds.map(\.title) == ["Loaded", "Ready", "Preview", "Later", "Installed"])
        #expect(MFBadge.Kind.loaded.tone == .success)
        #expect(MFBadge.Kind.ready.tone == .neutral)
        #expect(MFBadge.Kind.later.tone == .preview)
    }

    @Test func progressBadgeText() {
        #expect(MFProgressBadge.text(label: "Downloading", fraction: 0.629) == "Downloading · 62%")
        #expect(MFProgressBadge.text(label: "Downloading", fraction: 1.4) == "Downloading · 100%")
        #expect(MFProgressBadge.text(label: "Loading Deep", fraction: nil) == "Loading Deep")
    }

    /// Light and dark renders of every component, at the mockup size.
    @Test(arguments: [ColorScheme.light, .dark])
    func galleryRenders(colorScheme: ColorScheme) throws {
        let url = try Snapshot.render(MFComponentGallery(), name: "design-components", colorScheme: colorScheme)
        let image = try #require(NSImage(contentsOf: url))
        let rep = try #require(image.representations.first)
        #expect(rep.pixelsWide == 2560 && rep.pixelsHigh == 1640)
    }

    /// Ink text on the surfaces, and each tone on its fill, keep 4.5:1 (G-A7).
    @Test(arguments: [NSAppearance.Name.aqua, .darkAqua])
    func tokenContrast(appearanceName: NSAppearance.Name) throws {
        let appearance = try #require(NSAppearance(named: appearanceName))
        let pairs: [(String, Color, Color)] = [
            ("ink/surface", MFColor.ink, MFColor.surface),
            ("secondary/surface", MFColor.secondary, MFColor.surface),
            ("tertiary/canvas", MFColor.tertiary, MFColor.canvas),
            ("onInk/ink", MFColor.onInk, MFColor.ink),
            ("success/successFill", MFColor.success, MFColor.successFill),
            ("accent/accentFill", MFColor.accent, MFColor.accentFill),
            ("preview/previewFill", MFColor.preview, MFColor.previewFill),
            ("secondary/fill", MFColor.secondary, MFColor.fill),
        ]
        for (name, foreground, background) in pairs {
            let ratio = Contrast.ratio(foreground, background, appearance: appearance)
            #expect(ratio >= 4.5, "\(name) is \(ratio):1 in \(appearanceName.rawValue)")
        }
    }
}

enum Contrast {
    static func ratio(_ a: Color, _ b: Color, appearance: NSAppearance) -> Double {
        let la = luminance(a, appearance), lb = luminance(b, appearance)
        return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)
    }

    static func luminance(_ color: Color, _ appearance: NSAppearance) -> Double {
        var rgb: (Double, Double, Double) = (0, 0, 0)
        appearance.performAsCurrentDrawingAppearance {
            let ns = NSColor(color).usingColorSpace(.sRGB) ?? .black
            rgb = (Double(ns.redComponent), Double(ns.greenComponent), Double(ns.blueComponent))
        }
        func channel(_ c: Double) -> Double { c <= 0.03928 ? c / 12.92 : pow((c + 0.055) / 1.055, 2.4) }
        return 0.2126 * channel(rgb.0) + 0.7152 * channel(rgb.1) + 0.0722 * channel(rgb.2)
    }
}

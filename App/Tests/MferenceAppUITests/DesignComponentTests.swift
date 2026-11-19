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

    /// Non-finite progress (a zero total, a bad division) is unknown, never a trap.
    @Test(arguments: [Double.nan, .infinity, -.infinity])
    func nonFiniteProgressIsUnknown(value: Double) throws {
        #expect(MFProgress.fraction(value) == nil)
        #expect(MFProgress.percent(value) == nil)
        #expect(MFProgressBadge.text(label: "Downloading", fraction: value) == "Downloading")
        let bar = MFProgressBar(value, label: "Downloading").frame(width: 200)
        #expect(ImageRenderer(content: bar).cgImage != nil)
    }

    /// Light and dark renders of every component at the mockup size; the
    /// window body really changes with the appearance.
    @Test func galleryRendersInBothModes() throws {
        var bodies: [ColorScheme: Double] = [:]
        for scheme in [ColorScheme.light, .dark] {
            let url = try Snapshot.render(MFComponentGallery(), name: "design-components", colorScheme: scheme)
            let rep = try #require(NSImage(contentsOf: url)?.representations.first as? NSBitmapImageRep)
            #expect(rep.pixelsWide == 2560 && rep.pixelsHigh == 1640)
            // A point inside the window, below the gallery content.
            let color = try #require(rep.colorAt(x: 2400, y: 1500)?.usingColorSpace(.sRGB))
            bodies[scheme] = Double(color.redComponent + color.greenComponent + color.blueComponent) / 3
        }
        let light = try #require(bodies[.light]), dark = try #require(bodies[.dark])
        #expect(light > 0.85, "light window body should be near white, got \(light)")
        #expect(dark < 0.25, "dark window body should be near black, got \(dark)")
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
        // Non-text contrast (3:1): the edge of a switch, on and off.
        let edges: [(String, Color, Color)] = [
            ("toggle on/surface", MFColor.ink, MFColor.surface),
            ("toggle off border/surface", MFColor.controlBorder, MFColor.surface),
        ]
        for (name, foreground, background) in edges {
            let ratio = Contrast.ratio(foreground, background, appearance: appearance)
            #expect(ratio >= 3, "\(name) is \(ratio):1 in \(appearanceName.rawValue)")
        }
        // The knob stays visible on the off track in both modes (it vanished
        // in dark when it used the surface color); the mockup's light pair is
        // white on #C9CCD3, about 1.6:1, with a shadow for its edge.
        let knob = Contrast.ratio(MFColor.toggleKnob, MFColor.toggleOff, appearance: appearance)
        #expect(knob >= 1.5, "knob/off track is \(knob):1 in \(appearanceName.rawValue)")
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

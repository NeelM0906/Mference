import AppKit
import MferenceDesign
import SwiftUI
import Testing

/// The G-A3 snapshot harness every team uses.
///
/// `Snapshot.render(view, name:)` renders a view with `ImageRenderer` at the
/// mockup size (1280 x 820 points, scale 2, so 2560 x 1640 pixels like
/// `$APP/mockups/png`) and writes `<name>.png`:
///
/// - by default into a fresh temporary directory (the suite writes nothing
///   else), so the render path is exercised on every test run;
/// - into `$MFERENCE_SNAPSHOT_DIR` when that variable is set, for example
///   `MFERENCE_SNAPSHOT_DIR=/Users/zidanemitra/Desktop/Mference/scratch/app-program/results/visual`,
///   to produce the G-A3 pairs next to the mockup names.
///
/// `framing: .mockup` draws the view as the mockup boards do: the pastel
/// wallpaper filling 1280 x 820 and a 1200 x 748 window inset 40/36 points.
/// `framing: .fill` renders the view edge to edge at 1280 x 820.
/// `colorScheme: .dark` renders in dark mode and appends `-dark` to the name.
enum Snapshot {
    static let directoryVariable = "MFERENCE_SNAPSHOT_DIR"

    enum Framing { case mockup, fill }

    /// Where `render` writes: `$MFERENCE_SNAPSHOT_DIR`, or a per-process temp directory.
    static let outputDirectory: URL = {
        if let path = ProcessInfo.processInfo.environment[directoryVariable], !path.isEmpty {
            return URL(fileURLWithPath: path, isDirectory: true)
        }
        return FileManager.default.temporaryDirectory
            .appendingPathComponent("mference-snapshots-\(ProcessInfo.processInfo.processIdentifier)", isDirectory: true)
    }()

    /// The mockup board geometry (Main.dc.html: padding 36px 40px, window 1200 x 748).
    static let canvas = MFLayout.mockupWindow
    static let windowInset = EdgeInsets(top: 36, leading: 40, bottom: 36, trailing: 40)
    static let windowSize = CGSize(width: 1200, height: 748)
    static let scale: CGFloat = 2

    /// Renders `view` and writes `<name>[-dark].png`. Returns the file URL.
    @discardableResult
    static func render<V: View>(_ view: V, name: String, framing: Framing = .mockup,
                                colorScheme: ColorScheme = .light) throws -> URL {
        MFFontRegistry.registerBundledFonts()
        let content = framed(view, framing: framing)
            .environment(\.colorScheme, colorScheme)
        let renderer = ImageRenderer(content: content)
        renderer.scale = scale
        renderer.proposedSize = ProposedViewSize(canvas)
        let appearance = NSAppearance(named: colorScheme == .dark ? .darkAqua : .aqua)
        var image: CGImage?
        (appearance ?? NSAppearance.currentDrawing()).performAsCurrentDrawingAppearance {
            image = renderer.cgImage
        }
        let cgImage = try #require(image, "ImageRenderer produced no image for \(name)")
        let bitmap = NSBitmapImageRep(cgImage: cgImage)
        let png = try #require(bitmap.representation(using: .png, properties: [:]))
        try FileManager.default.createDirectory(at: outputDirectory, withIntermediateDirectories: true)
        let url = outputDirectory.appendingPathComponent(name + (colorScheme == .dark ? "-dark" : "") + ".png")
        try png.write(to: url)
        return url
    }

    @ViewBuilder
    static func framed<V: View>(_ view: V, framing: Framing) -> some View {
        switch framing {
        case .fill:
            view.frame(width: canvas.width, height: canvas.height)
        case .mockup:
            ZStack(alignment: .topLeading) {
                Rectangle().fill(MFColor.wallpaper)
                view
                    .frame(width: windowSize.width, height: windowSize.height)
                    .clipShape(RoundedRectangle(cornerRadius: MFRadius.l))
                    .overlay(RoundedRectangle(cornerRadius: MFRadius.l).strokeBorder(MFColor.surface.opacity(0.8)))
                    .shadow(color: MFColor.ink.opacity(0.18), radius: 40, y: 30)
                    .padding(windowInset)
            }
            .frame(width: canvas.width, height: canvas.height)
        }
    }
}

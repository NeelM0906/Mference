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
/// wallpaper filling the board and a window inset 40/36 points (1200 x 748 on
/// a 1280 x 820 board). `.centered(width:)` centers a narrower window (first
/// run: 1040). `.fill` renders the view edge to edge. `canvas:` sets the
/// board size for taller boards (Settings 1280 x 1520, library 1280 x 1000).
/// `colorScheme: .dark` renders in dark mode and appends `-dark` to the name.
enum Snapshot {
    static let directoryVariable = "MFERENCE_SNAPSHOT_DIR"

    enum Framing {
        /// The Main board: a 1200-wide window inset 40/36 on the wallpaper.
        case mockup
        /// A window of the given width, centered, inset 36 top and bottom
        /// (the first-run board's 1040-wide window).
        case centered(width: CGFloat)
        /// Edge to edge, no window.
        case fill
    }

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
    static let scale: CGFloat = 2

    /// Renders `view` and writes `<name>[-dark].png`. Returns the file URL.
    @discardableResult
    /// - Parameter canvas: the board size; most boards are 1280 x 820, the
    ///   Settings board is 1280 x 1520 and the model library 1280 x 1000.
    static func render<V: View>(_ view: V, name: String, framing: Framing = .mockup,
                                canvas: CGSize = Snapshot.canvas,
                                colorScheme: ColorScheme = .light) throws -> URL {
        MFFontRegistry.registerBundledFonts()
        let content = framed(view.environment(\.mfRendersStatically, true), framing: framing, canvas: canvas)
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
    static func framed<V: View>(_ view: V, framing: Framing, canvas: CGSize = Snapshot.canvas) -> some View {
        switch framing {
        case .fill:
            view.frame(width: canvas.width, height: canvas.height)
        case .mockup:
            window(view, size: CGSize(width: canvas.width - windowInset.leading - windowInset.trailing,
                                      height: canvas.height - windowInset.top - windowInset.bottom),
                   canvas: canvas)
        case .centered(let width):
            window(view, size: CGSize(width: width, height: canvas.height - windowInset.top - windowInset.bottom),
                   canvas: canvas)
        }
    }

    private static func window<V: View>(_ view: V, size: CGSize, canvas: CGSize) -> some View {
        ZStack {
            Rectangle().fill(MFColor.wallpaper)
            // The shadow sits on a shape behind the window: on the content it
            // would be drawn under every child view.
            RoundedRectangle(cornerRadius: MFRadius.l)
                .fill(MFColor.surface)
                .frame(width: size.width, height: size.height)
                .shadow(color: MFColor.ink.opacity(0.18), radius: 40, y: 30)
            view
                .frame(width: size.width, height: size.height)
                .clipShape(RoundedRectangle(cornerRadius: MFRadius.l))
                .overlay(RoundedRectangle(cornerRadius: MFRadius.l).strokeBorder(MFColor.surface.opacity(0.8)))
        }
        .frame(width: canvas.width, height: canvas.height)
    }
}

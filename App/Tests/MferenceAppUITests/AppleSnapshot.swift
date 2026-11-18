import AppKit
import MferenceAppCore
@testable import MferenceAppUI
import MferenceDesign
import SwiftData
import SwiftUI
import Testing

/// G-A3 snapshots of the Scan, Calendar and Mail screens with fixture data,
/// framed like the mockup boards (1280 x 820 at 2x). Written to
/// `MFERENCE_SNAPSHOT_DIR` when set, else to a temporary folder.
@MainActor enum AppleSnapshot {
    static var outputDirectory: URL {
        if let path = ProcessInfo.processInfo.environment["MFERENCE_SNAPSHOT_DIR"] {
            return URL(filePath: path, directoryHint: .isDirectory)
        }
        return URL.temporaryDirectory.appending(path: "mference-snapshots", directoryHint: .isDirectory)
    }

    /// The board: wallpaper, a 1200 x 748 window inset 40/36, a stand-in for
    /// the shell's 70 pt chrome (U1 draws the real one), the screen below.
    static func board<Content: View>(_ section: AppSection, @ViewBuilder content: () -> Content) -> some View {
        ZStack(alignment: .topLeading) {
            Rectangle().fill(MFColor.wallpaper)
            VStack(spacing: 0) {
                HStack(spacing: MFSpacing.m) {
                    Text("Mference").font(MFFont.headline).foregroundStyle(MFColor.ink)
                    ForEach(AppSection.pills) { pill in
                        Text(pill.title)
                            .font(MFFont.body)
                            .foregroundStyle(pill == section ? MFColor.onInk : MFColor.secondary)
                            .padding(.horizontal, MFSpacing.m)
                            .padding(.vertical, MFSpacing.xs + MFSpacing.xxs)
                            .background(Capsule().fill(pill == section ? MFColor.ink : .clear))
                    }
                    Spacer()
                }
                .padding(.horizontal, MFSpacing.xl)
                .frame(height: MFLayout.toolbarHeight)
                Rectangle().fill(MFColor.hairline).frame(height: 1)
                content()
            }
            .frame(width: 1200, height: 748)
            .background(MFColor.surface)
            .clipShape(RoundedRectangle(cornerRadius: MFRadius.window))
            .compositingGroup()
            .shadow(color: .black.opacity(0.12), radius: 24, y: 10)
            .offset(x: 40, y: 36)
        }
        .frame(width: MFLayout.mockupWindow.width, height: MFLayout.mockupWindow.height)
    }

    @discardableResult
    static func render<V: View>(_ view: V, name: String, colorScheme: ColorScheme = .light) throws -> URL {
        let renderer = ImageRenderer(content: view.environment(\.colorScheme, colorScheme)
            .environment(\.appleStaticRendering, true))
        renderer.scale = 2
        renderer.proposedSize = ProposedViewSize(MFLayout.mockupWindow)
        let appearance = NSAppearance(named: colorScheme == .dark ? .darkAqua : .aqua)
        var image: CGImage?
        appearance?.performAsCurrentDrawingAppearance { image = renderer.cgImage }
        guard let image else { throw CocoaError(.fileWriteUnknown) }
        try FileManager.default.createDirectory(at: outputDirectory, withIntermediateDirectories: true)
        let url = outputDirectory.appending(path: colorScheme == .dark ? "\(name)-dark.png" : "\(name).png")
        let bitmap = NSBitmapImageRep(cgImage: image)
        guard let png = bitmap.representation(using: .png, properties: [:]) else { throw CocoaError(.fileWriteUnknown) }
        try png.write(to: url)
        #expect(image.width == 2_560 && image.height == 1_640)
        return url
    }
}

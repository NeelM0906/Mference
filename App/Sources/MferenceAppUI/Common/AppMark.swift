import AppKit
import MferenceDesign
import SwiftUI

/// The app's mark: the bundle's own icon (`CFBundleIconFile`, App/Resources/AppIcon.icns)
/// when the app runs from its bundle; a lettered tile otherwise (tests, previews, `swift run`).
struct AppMark: View {
    let side: CGFloat
    let radius: CGFloat
    let letterFont: Font
    var borderWidth: CGFloat = 1

    var body: some View {
        if let icon = Self.bundleIcon {
            // The icon sits on the macOS grid (an 824-point body on a 1024 canvas), so it is
            // drawn larger than `side` and the body itself fills the mark's frame.
            Image(nsImage: icon)
                .resizable()
                .interpolation(.high)
                .frame(width: side * Self.canvasPerBody, height: side * Self.canvasPerBody)
                .frame(width: side, height: side)
        } else {
            Text("M")
                .font(letterFont)
                .foregroundStyle(MFColor.secondary)
                .frame(width: side, height: side)
                .background(
                    RoundedRectangle(cornerRadius: radius)
                        .fill(LinearGradient(colors: [MFColor.surface, MFColor.hairlineStrong],
                                             startPoint: .topLeading, endPoint: .bottomTrailing)))
                .overlay(RoundedRectangle(cornerRadius: radius).strokeBorder(MFColor.hairlineStrong, lineWidth: borderWidth))
        }
    }

    /// The icon canvas over the icon body on the macOS app icon grid.
    static let canvasPerBody: CGFloat = 1024 / 824

    /// The running bundle's icon, or nil when the bundle names none (an unbundled build).
    static let bundleIcon: NSImage? = icon(in: .main)

    static func icon(in bundle: Bundle) -> NSImage? {
        guard let name = bundle.object(forInfoDictionaryKey: "CFBundleIconFile") as? String, !name.isEmpty else {
            return nil
        }
        let base = (name as NSString).deletingPathExtension
        guard let url = bundle.url(forResource: base, withExtension: "icns") else { return nil }
        return NSImage(contentsOf: url)
    }
}

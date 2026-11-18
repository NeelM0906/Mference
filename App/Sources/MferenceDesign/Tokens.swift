import SwiftUI

// SHARED CONTRACT (App Program spec section 6). Token names and meanings
// change only through a PR the coordinator approves; values may be tuned by
// U1 to match the mockup. Every view uses these tokens, never raw literals.
//
// Light values are taken from the mockup sources ($APP/mockups/src). The
// mockup is light only; dark values are a derived palette (U1 checks both).

/// The app's color tokens.
public enum MFColor {
    /// Primary text, selected nav pill, primary buttons. Mockup `#17181C`.
    public static let ink = Color.mf(light: 0x17181C, dark: 0xF2F3F5)
    /// Secondary text. Mockup `#4F535D`.
    public static let secondary = Color.mf(light: 0x4F535D, dark: 0xB4B8C2)
    /// Tertiary text: captions, engine labels, section headers. Mockup `#5E626C`.
    public static let tertiary = Color.mf(light: 0x5E626C, dark: 0x9A9EA8)
    /// Strong hairline: card borders. Mockup `#DFE1E6`.
    public static let hairlineStrong = Color.mf(light: 0xDFE1E6, dark: 0x3A3D45)
    /// Hairline: separators, chip borders. Mockup `#E3E5EA`.
    public static let hairline = Color.mf(light: 0xE3E5EA, dark: 0x33363E)
    /// Faint hairline: table rows. Mockup `#E9EAEE`.
    public static let hairlineFaint = Color.mf(light: 0xE9EAEE, dark: 0x2B2E36)
    /// Neutral fill: selected list row, user bubble, quiet buttons. Mockup `#F0F1F4`.
    public static let fill = Color.mf(light: 0xF0F1F4, dark: 0x2A2C33)
    /// Subtle fill: badges, keycaps. Mockup `#ECEDF0`.
    public static let fillSubtle = Color.mf(light: 0xECEDF0, dark: 0x25272D)
    /// Surface: cards, composer, popovers.
    public static let surface = Color.mf(light: 0xFFFFFF, dark: 0x1C1D22)
    /// Window body behind content (over the material).
    public static let canvas = Color.mf(light: 0xFBFBFC, dark: 0x16171B)
    /// Links, selection, citations. Mockup `#2F47B8`.
    public static let accent = Color.mf(light: 0x2F47B8, dark: 0x8FA2FF)
    /// Accent text on accent fills. Mockup `#1F338F`.
    public static let accentStrong = Color.mf(light: 0x1F338F, dark: 0xB4C1FF)
    /// Accent fill: "Web on", citation number chips, selected picker row. Mockup `#E8ECFF`.
    public static let accentFill = Color.mf(light: 0xE8ECFF, dark: 0x262E55)
    /// Faint accent fill. Mockup `#F1F3FF`.
    public static let accentFillFaint = Color.mf(light: 0xF1F3FF, dark: 0x1F2440)
    /// Success text: "On this Mac", "Loaded", "Installed". Mockup `#1C6B43`.
    public static let success = Color.mf(light: 0x1C6B43, dark: 0x6FD3A0)
    /// Success fill. Mockup `#E3F4EA`.
    public static let successFill = Color.mf(light: 0xE3F4EA, dark: 0x173527)
    /// Preview / "Later" text. Mockup `#8A4B06`.
    public static let preview = Color.mf(light: 0x8A4B06, dark: 0xF2B36B)
    /// Preview fill. Mockup `#FFF1DE`.
    public static let previewFill = Color.mf(light: 0xFFF1DE, dark: 0x3A2810)
    /// Text on ink (selected pill, primary button).
    public static let onInk = Color.mf(light: 0xFFFFFF, dark: 0x17181C)
    /// Destructive actions.
    public static let danger = Color.mf(light: 0xB42318, dark: 0xFF8A80)
    /// Sidebar background (Settings, chat list). Mockup `#F6F6F8`.
    public static let sidebar = Color.mf(light: 0xF6F6F8, dark: 0x1A1B20)
    /// Selected sidebar row. Mockup `#E6E7EC`.
    public static let sidebarSelected = Color.mf(light: 0xE6E7EC, dark: 0x2E3038)
    /// Switch track when off. Mockup `#C9CCD3`.
    public static let toggleOff = Color.mf(light: 0xC9CCD3, dark: 0x4A4D56)
    /// Outline of an off switch, so its edge keeps 3:1 against the surface (G-A7).
    public static let controlBorder = Color.mf(light: 0x8C9099, dark: 0x80848E)
    /// Approval card fill and border. Mockup `#FFF9F1` / `#F2D6B0`.
    public static let approvalFill = Color.mf(light: 0xFFF9F1, dark: 0x2A2014)
    public static let approvalBorder = Color.mf(light: 0xF2D6B0, dark: 0x5C4321)
    /// Toolbar and window-chrome divider. Mockup `#ECEDF0`.
    public static let chromeDivider = Color.mf(light: 0xECEDF0, dark: 0x2E3038)

    /// The soft pastel wallpaper behind the frosted window (mockup backdrop).
    public static let wallpaper = LinearGradient(
        colors: [Color.mf(light: 0xE7E3FF, dark: 0x1D1A33),
                 Color.mf(light: 0xF3F4F9, dark: 0x16171B),
                 Color.mf(light: 0xFFE5D4, dark: 0x2A1E18)],
        startPoint: .topLeading, endPoint: .bottomTrailing)
}

/// Font tokens. Geist is bundled by U1 (P6); until it is registered these
/// fall back to the system font at the same size and weight.
public enum MFFont {
    public static let familyName = "Geist"

    // The scale measured from the mockup CSS (coordinator correction,
    // 2026-11-18): every size here is the .dc.html px value.

    /// 24/600: hero and first-run page titles.
    public static let largeTitle = geist(24, .semibold)
    /// 19/600: screen and reader titles.
    public static let title = geist(19, .semibold)
    /// 15/600: panel headers (chat title, "Important", "To-dos"), banner titles.
    public static let headline = geist(15, .semibold)
    /// 14/400: messages, mail body, page copy.
    public static let body = geist(14, .regular)
    /// 14/500: emphasised body, large button labels.
    public static let bodyEmphasis = geist(14, .medium)
    /// 13.5/600: list row titles, settings rows.
    public static let rowTitle = geist(13.5, .semibold)
    /// 13/400: rows, nav, buttons.
    public static let callout = geist(13, .regular)
    /// 13/500: nav pills, button labels.
    public static let calloutEmphasis = geist(13, .medium)
    /// 12.5/400: previews and row subtitles.
    public static let preview = geist(12.5, .regular)
    /// 12/400: captions, engine labels.
    public static let caption = geist(12, .regular)
    /// 11.5/500: chips and badges.
    public static let chip = geist(11.5, .medium)
    /// 11/600, uppercase, tracked (`sectionHeaderTracking`): sidebar, picker
    /// and table section headers.
    public static let sectionHeader = geist(11, .semibold)
    /// 0.06em of the 11pt section header.
    public static let sectionHeaderTracking: CGFloat = 0.66
    /// 10.5/600: citation marks.
    public static let micro = geist(10.5, .semibold)

    // Variants the CSS uses that are not in the base scale.

    /// 16/600: settings section titles (`h2`), first-run card titles.
    public static let sectionTitle = geist(16, .semibold)
    /// 14/600: the app name in the chrome.
    public static let brand = geist(14, .semibold)
    /// 13.5/400 and 13.5/500: first-run card body and step list (current step).
    public static let bodySmall = geist(13.5, .regular)
    public static let bodySmallEmphasis = geist(13.5, .medium)
    /// 13/600: table row titles, inline emphasis.
    public static let calloutSemibold = geist(13, .semibold)
    /// 11 monospaced: keycaps (Geist Mono in the mockup; not bundled).
    public static let keycap = Font.system(size: 11, weight: .medium, design: .monospaced)

    public static func geist(_ size: CGFloat, _ weight: Font.Weight) -> Font {
        .custom(familyName, size: size, relativeTo: .body).weight(weight)
    }
}

/// Extra line spacing (points). The mockup's line heights are 1.35-1.6;
/// message and mail bodies use `relaxed`.
public enum MFLineSpacing {
    public static let tight: CGFloat = 2
    public static let normal: CGFloat = 4
    public static let relaxed: CGFloat = 6
}

/// Spacing scale (points).
public enum MFSpacing {
    public static let xxs: CGFloat = 2
    public static let xs: CGFloat = 4
    public static let s: CGFloat = 8
    public static let m: CGFloat = 12
    public static let l: CGFloat = 16
    public static let xl: CGFloat = 24
    public static let xxl: CGFloat = 32

    // Steps between the base scale, as the mockup CSS uses them.

    /// 6: icon-to-label gaps (`.nv`, `.btn`).
    public static let snug: CGFloat = 6
    /// 10: row control gaps, compact paddings.
    public static let compact: CGFloat = 10
    /// 14: row gaps, card padding.
    public static let comfortable: CGFloat = 14
    /// 20: gaps between page sections.
    public static let roomy: CGFloat = 20
}

/// Corner radii (points), from the mockup CSS.
public enum MFRadius {
    /// Keycaps, small badges.
    public static let xs: CGFloat = 5
    /// Citation number chips, table cells.
    public static let s: CGFloat = 8
    /// List rows, buttons.
    public static let m: CGFloat = 10
    /// Cards, tables.
    public static let l: CGFloat = 14
    /// Composer, popovers.
    public static let xl: CGFloat = 18
    /// The window.
    public static let window: CGFloat = 22
    /// Pills and capsules use `Capsule()`; this is the CSS 999px.
    public static let pill: CGFloat = 999
}

/// Fixed sizes from the mockup (1280 x 820 window at 1x).
public enum MFLayout {
    public static let mockupWindow = CGSize(width: 1280, height: 820)
    /// The app window as drawn on the mockup boards, and its default size.
    public static let defaultWindow = CGSize(width: 1200, height: 748)
    /// The smallest useful window.
    public static let minimumWindow = CGSize(width: 960, height: 640)
    /// Hairline separators and borders.
    public static let hairline: CGFloat = 1
    /// Emphasised strokes (selected card outline, current step ring).
    public static let strokeEmphasis: CGFloat = 2
    public static let sidebarWidth: CGFloat = 270
    public static let toolbarHeight: CGFloat = 70
    public static let contentMaxWidth: CGFloat = 960
}

extension Color {
    /// A color that resolves to `light` or `dark` with the appearance.
    public static func mf(light: UInt32, dark: UInt32) -> Color {
        Color(nsColor: NSColor(name: nil) { appearance in
            let isDark = appearance.bestMatch(from: [.aqua, .darkAqua]) == .darkAqua
            return NSColor(mfHex: isDark ? dark : light)
        })
    }
}

extension NSColor {
    convenience init(mfHex hex: UInt32) {
        self.init(srgbRed: CGFloat((hex >> 16) & 0xFF) / 255,
                  green: CGFloat((hex >> 8) & 0xFF) / 255,
                  blue: CGFloat(hex & 0xFF) / 255,
                  alpha: 1)
    }
}

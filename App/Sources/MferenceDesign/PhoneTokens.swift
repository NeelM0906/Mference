import SwiftUI

// Phone sizes for the iPhone and iPhone Duo app (Mobile/), as locked on the
// mobile design canvas (2026-11-19). Colors, radii and spacing are the shared
// MFColor / MFRadius / MFSpacing tokens unchanged; only type and control sizes
// grow for touch. Every font is relative to an iOS text style, so it scales
// with Dynamic Type.

extension MFFont {
    /// Geist at `size`, scaling with Dynamic Type like `style`.
    public static func geist(_ size: CGFloat, _ weight: Font.Weight, relativeTo style: Font.TextStyle) -> Font {
        .custom(familyName, size: size, relativeTo: style).weight(weight)
    }
}

/// The phone type scale (Mac size in brackets).
public enum MFPhoneFont {
    /// 30/600 (24): screen titles.
    public static let largeTitle = MFFont.geist(30, .semibold, relativeTo: .largeTitle)
    /// 22/600 (19): sheet and pane titles, the scanned merchant.
    public static let title = MFFont.geist(22, .semibold, relativeTo: .title2)
    /// 17/600 (15): navigation titles, card titles.
    public static let headline = MFFont.geist(17, .semibold, relativeTo: .headline)
    /// 17/400 (14): messages and page copy.
    public static let body = MFFont.geist(17, .regular, relativeTo: .body)
    /// 16/600 (13.5): list row titles.
    public static let rowTitle = MFFont.geist(16, .semibold, relativeTo: .callout)
    /// 16/400 (13): row text.
    public static let callout = MFFont.geist(16, .regular, relativeTo: .callout)
    /// 16/500 (13): button labels.
    public static let button = MFFont.geist(16, .medium, relativeTo: .callout)
    /// 17/600: full-width primary button labels.
    public static let buttonLarge = MFFont.geist(17, .semibold, relativeTo: .body)
    /// 15/400 (12.5): row subtitles and previews.
    public static let preview = MFFont.geist(15, .regular, relativeTo: .subheadline)
    /// 15/500: compact buttons, segmented controls.
    public static let previewEmphasis = MFFont.geist(15, .medium, relativeTo: .subheadline)
    /// 14/500: composer controls.
    public static let control = MFFont.geist(14, .medium, relativeTo: .subheadline)
    /// 13/400 (12): captions and engine labels.
    public static let caption = MFFont.geist(13, .regular, relativeTo: .footnote)
    /// 13/500 (11.5): chips and badges.
    public static let chip = MFFont.geist(13, .medium, relativeTo: .footnote)
    /// 12/600 (11), uppercase and tracked: section headers.
    public static let sectionHeader = MFFont.geist(12, .semibold, relativeTo: .caption)
    /// 0.06em of the 12pt section header.
    public static let sectionHeaderTracking: CGFloat = 0.72
    /// 12/400: time labels in the day grid, weekday letters.
    public static let micro = MFFont.geist(12, .regular, relativeTo: .caption)
    /// 11/600 (10.5): citation marks.
    public static let citation = MFFont.geist(11, .semibold, relativeTo: .caption2)
}

/// Phone control and layout sizes.
public enum MFPhoneMetrics {
    /// The minimum side of anything tappable.
    public static let touchTarget: CGFloat = 44
    /// Buttons (Mac 30).
    public static let buttonHeight: CGFloat = 44
    /// Compact buttons inside cards and banners.
    public static let compactButtonHeight: CGFloat = 40
    /// Full-width primary buttons.
    public static let primaryButtonHeight: CGFloat = 52
    /// Chips and badges (Mac 22); their hit area is padded to 44.
    public static let chipHeight: CGFloat = 28
    /// Composer chips and the + button.
    public static let composerControl: CGFloat = 34
    /// The composer send / dictate button.
    public static let sendButton: CGFloat = 36
    /// Numbered source chips under an answer.
    public static let sourceChipHeight: CGFloat = 32
    /// Inline citation mark.
    public static let citation: CGFloat = 18
    /// One-line and two-line list rows.
    public static let rowHeight: CGFloat = 48
    public static let rowHeightWithSubtitle: CGFloat = 64
    /// The composer's corner (half its resting height).
    public static let composerRadius: CGFloat = 24
    /// User bubble corner and widest width.
    public static let bubbleRadius: CGFloat = 18
    public static let bubbleMaxWidth: CGFloat = 300
    /// Side margin on iPhone and Duo closed; inside each Duo pane.
    public static let screenMargin: CGFloat = 16
    public static let paneMargin: CGFloat = 20
    /// Nothing sits within this distance of the Duo fold.
    public static let foldKeepOut: CGFloat = 12
    /// At or above this window width the app uses two panes (Duo open).
    public static let twoPaneMinWidth: CGFloat = 700
    /// The radio dot in the model picker.
    public static let radio: CGFloat = 20
    /// One hour in the day grid.
    public static let hourHeight: CGFloat = 64
}

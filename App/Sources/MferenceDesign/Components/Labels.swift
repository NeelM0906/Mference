import SwiftUI

// Chips, badges, source chips and progress from the mockup CSS (`.chip`,
// `.cite`, `.bar`).

/// The color pair a chip or badge uses.
public enum MFTone: Sendable, Hashable {
    /// Grey fill, secondary text ("Ready", "Built in", "Always").
    case neutral
    /// Green: "On this Mac", "Loaded", "Installed".
    case success
    /// Blue: downloading, "Web on".
    case accent
    /// Amber: "Preview", "Later".
    case preview
    /// Ink fill, white text (a selected filter chip).
    case ink

    var foreground: Color {
        switch self {
        case .neutral: MFColor.secondary
        case .success: MFColor.success
        case .accent: MFColor.accent
        case .preview: MFColor.preview
        case .ink: MFColor.onInk
        }
    }

    var background: Color {
        switch self {
        case .neutral: MFColor.fill
        case .success: MFColor.successFill
        case .accent: MFColor.accentFill
        case .preview: MFColor.previewFill
        case .ink: MFColor.ink
        }
    }
}

// MARK: - Chip

/// A small capsule label with an optional icon (`.chip`).
public struct MFChip: View {
    let title: String
    let systemImage: String?
    let tone: MFTone

    public init(_ title: String, systemImage: String? = nil, tone: MFTone = .neutral) {
        self.title = title
        self.systemImage = systemImage
        self.tone = tone
    }

    public var body: some View {
        HStack(spacing: MFMetrics.chipIconGap) {
            if let systemImage {
                Image(systemName: systemImage)
                    .font(.system(size: MFMetrics.chipIconSize, weight: .semibold))
                    .accessibilityHidden(true)
            }
            Text(title)
                .lineLimit(1)
        }
        .font(MFFont.chip)
        .foregroundStyle(tone.foreground)
        .padding(.horizontal, MFSpacing.s)
        .frame(height: MFMetrics.chipHeight)
        .background(Capsule().fill(tone.background))
        .accessibilityElement(children: .combine)
    }
}

// MARK: - Badge

/// A model or feature state badge.
public struct MFBadge: View {
    /// The fixed states of spec section 6, plus any other text with a tone.
    public enum Kind: Sendable, Hashable {
        case loaded, ready, preview, later, installed
        case custom(String, MFTone)

        public var title: String {
            switch self {
            case .loaded: "Loaded"
            case .ready: "Ready"
            case .preview: "Preview"
            case .later: "Later"
            case .installed: "Installed"
            case .custom(let title, _): title
            }
        }

        public var tone: MFTone {
            switch self {
            case .loaded, .installed: .success
            case .ready: .neutral
            case .preview, .later: .preview
            case .custom(_, let tone): tone
            }
        }
    }

    let kind: Kind

    public init(_ kind: Kind) { self.kind = kind }

    public var body: some View {
        MFChip(kind.title, tone: kind.tone)
    }
}

// MARK: - Progress

/// Progress values from the outside world: anything that is not a finite
/// number counts as unknown; the rest is clamped to 0...1.
public enum MFProgress {
    public static func fraction(_ value: Double?) -> Double? {
        guard let value, value.isFinite else { return nil }
        return min(max(value, 0), 1)
    }

    /// Whole percent, rounded down; nil when unknown.
    public static func percent(_ value: Double?) -> Int? {
        fraction(value).map { Int(($0 * 100).rounded(.down)) }
    }
}

/// A download or load in progress: "Downloading · 62%" on the accent fill.
/// The percentage is left out while the fraction is unknown (nil, NaN, infinite).
public struct MFProgressBadge: View {
    let label: String
    let fraction: Double?

    public init(_ label: String, fraction: Double?) {
        self.label = label
        self.fraction = fraction
    }

    public var body: some View {
        MFChip(Self.text(label: label, fraction: fraction), tone: .accent)
    }

    /// "Downloading · 62%", or the label alone when the fraction is unknown.
    public static func text(label: String, fraction: Double?) -> String {
        guard let percent = MFProgress.percent(fraction) else { return label }
        return "\(label) · \(percent)%"
    }
}

/// The thin progress bar of the model library (`.bar`). An unknown fraction
/// draws the empty track and reads as "unknown".
public struct MFProgressBar: View {
    let fraction: Double?
    let label: String

    /// - Parameter label: what is progressing, for VoiceOver ("Downloading Deep").
    public init(_ fraction: Double?, label: String) {
        self.fraction = MFProgress.fraction(fraction)
        self.label = label
    }

    public var body: some View {
        GeometryReader { proxy in
            ZStack(alignment: .leading) {
                Capsule().fill(MFColor.hairline)
                if let fraction {
                    Capsule().fill(MFColor.accent)
                        .frame(width: proxy.size.width * fraction)
                }
            }
        }
        .frame(height: MFMetrics.progressBar)
        .accessibilityElement()
        .accessibilityLabel(label)
        .accessibilityValue(MFProgress.percent(fraction).map { "\($0) percent" } ?? "unknown")
    }
}

// MARK: - Sources

/// A citation number inline with text (`.cite`).
public struct MFCitationMark: View {
    let number: Int

    public init(_ number: Int) { self.number = number }

    public var body: some View {
        Text("\(number)")
            .font(MFFont.citation)
            .foregroundStyle(MFColor.accent)
            .frame(width: MFMetrics.citation, height: MFMetrics.citation)
            .background(RoundedRectangle(cornerRadius: MFRadius.xs).fill(MFColor.accentFill))
            .accessibilityLabel("Source \(number)")
    }
}

/// A numbered source under a web answer: "1 stride-desks.example".
public struct MFSourceChip: View {
    let number: Int
    let title: String
    let action: () -> Void

    public init(number: Int, title: String, action: @escaping () -> Void) {
        self.number = number
        self.title = title
        self.action = action
    }

    public var body: some View {
        Button(action: action) {
            HStack(spacing: MFSpacing.snug) {
                Text("\(number)")
                    .font(MFFont.calloutEmphasis)
                    .foregroundStyle(MFColor.accent)
                Text(title)
                    .font(MFFont.callout)
                    .foregroundStyle(MFColor.ink)
                    .lineLimit(1)
            }
            .padding(.horizontal, MFSpacing.compact)
            .frame(height: MFMetrics.sourceChipHeight)
            .background(Capsule().fill(MFColor.surface))
            .overlay(Capsule().strokeBorder(MFColor.hairline, lineWidth: MFLayout.hairline))
            .contentShape(Capsule())
        }
        .buttonStyle(.plain)
        .accessibilityLabel("Source \(number), \(title)")
    }
}

// MARK: - Section header

/// The uppercase, tracked header of sidebars, pickers and tables (`.lab`).
public struct MFSectionHeader: View {
    let title: String

    public init(_ title: String) { self.title = title }

    public var body: some View {
        Text(title.uppercased())
            .font(MFFont.sectionHeader)
            .tracking(0.7)
            .foregroundStyle(MFColor.tertiary)
            .accessibilityAddTraits(.isHeader)
    }
}

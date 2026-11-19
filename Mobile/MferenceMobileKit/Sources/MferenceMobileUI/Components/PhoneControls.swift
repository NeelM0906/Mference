import MferenceDesign
import SwiftUI

// The Mac's capsules at phone sizes. Colors are MFColor; sizes are
// MFPhoneMetrics; type is MFPhoneFont.

/// Capsule buttons: primary (ink), secondary (white with a hairline), quiet
/// (grey fill).
struct PhoneButtonStyle: ButtonStyle {
    enum Kind { case primary, secondary, quiet }
    enum Size { case regular, compact, large }

    var kind: Kind = .secondary
    var size: Size = .regular
    var fullWidth = false

    func makeBody(configuration: Configuration) -> some View {
        StyledBody(configuration: configuration, style: self)
    }

    private struct StyledBody: View {
        let configuration: Configuration
        let style: PhoneButtonStyle
        @Environment(\.isEnabled) private var isEnabled

        var body: some View {
            configuration.label
                .font(style.size == .large ? MFPhoneFont.buttonLarge
                      : style.size == .compact ? MFPhoneFont.previewEmphasis : MFPhoneFont.button)
                .foregroundStyle(style.kind == .primary ? MFColor.onInk : MFColor.ink)
                .lineLimit(1)
                .padding(.horizontal, style.size == .compact ? MFSpacing.comfortable : MFSpacing.l + 2)
                .frame(maxWidth: style.fullWidth ? .infinity : nil)
                .frame(minHeight: height)
                .background(Capsule().fill(fill(pressed: configuration.isPressed)))
                .overlay(Capsule().strokeBorder(style.kind == .secondary ? MFColor.hairlineStrong : .clear,
                                                lineWidth: MFLayout.hairline))
                .contentShape(Capsule())
                .opacity(isEnabled ? (configuration.isPressed ? 0.85 : 1) : 0.5)
        }

        private var height: CGFloat {
            switch style.size {
            case .regular: MFPhoneMetrics.buttonHeight
            case .compact: MFPhoneMetrics.compactButtonHeight
            case .large: MFPhoneMetrics.primaryButtonHeight
            }
        }

        private func fill(pressed: Bool) -> Color {
            switch style.kind {
            case .primary: MFColor.ink
            case .secondary: pressed ? MFColor.fill : MFColor.surface
            case .quiet: pressed ? MFColor.hairline : MFColor.fill
            }
        }
    }
}

extension MFTone {
    var phoneForeground: Color {
        switch self {
        case .neutral: MFColor.secondary
        case .success: MFColor.success
        case .accent: MFColor.accent
        case .preview: MFColor.preview
        case .ink: MFColor.onInk
        }
    }

    var phoneBackground: Color {
        switch self {
        case .neutral: MFColor.fill
        case .success: MFColor.successFill
        case .accent: MFColor.accentFill
        case .preview: MFColor.previewFill
        case .ink: MFColor.ink
        }
    }
}

/// A capsule label: "Loaded", "On this iPhone", "Check".
struct PhoneChip: View {
    let title: String
    var systemImage: String?
    var tone: MFTone = .neutral

    var body: some View {
        HStack(spacing: 5) {
            if let systemImage {
                Image(systemName: systemImage).font(.system(size: 11, weight: .semibold)).accessibilityHidden(true)
            }
            Text(title).lineLimit(1)
        }
        .font(MFPhoneFont.chip)
        .foregroundStyle(tone.phoneForeground)
        .padding(.horizontal, MFSpacing.compact)
        .frame(minHeight: MFPhoneMetrics.chipHeight - 2)
        .background(Capsule().fill(tone.phoneBackground))
        .fixedSize()
        .accessibilityElement(children: .combine)
    }
}

/// The uppercase, tracked header over a group of rows.
struct PhoneSectionHeader: View {
    let title: String

    init(_ title: String) { self.title = title }

    var body: some View {
        Text(title.uppercased())
            .font(MFPhoneFont.sectionHeader)
            .tracking(MFPhoneFont.sectionHeaderTracking)
            .foregroundStyle(MFColor.tertiary)
            .padding(.horizontal, MFSpacing.xs)
            .frame(maxWidth: .infinity, alignment: .leading)
            .accessibilityAddTraits(.isHeader)
    }
}

/// A white card whose rows are separated by an inset hairline.
struct PhoneCard<Content: View>: View {
    @ViewBuilder let content: Content

    var body: some View {
        VStack(spacing: 0) {
            Group(subviews: content) { subviews in
                ForEach(Array(subviews.enumerated()), id: \.element.id) { index, subview in
                    if index > 0 {
                        Rectangle().fill(MFColor.hairlineFaint).frame(height: MFLayout.hairline)
                            .padding(.leading, MFSpacing.l)
                    }
                    subview
                }
            }
        }
        .background(MFColor.surface)
        .clipShape(RoundedRectangle(cornerRadius: MFRadius.l))
        .overlay(RoundedRectangle(cornerRadius: MFRadius.l).strokeBorder(MFColor.hairline, lineWidth: MFLayout.hairline))
    }
}

/// A card row: bold title, secondary line, trailing accessory.
struct PhoneRow<Trailing: View>: View {
    let title: String
    var subtitle: String?
    @ViewBuilder var trailing: Trailing

    var body: some View {
        HStack(spacing: MFSpacing.m) {
            VStack(alignment: .leading, spacing: MFSpacing.xxs) {
                Text(title).font(MFPhoneFont.rowTitle).foregroundStyle(MFColor.ink)
                if let subtitle {
                    Text(subtitle).font(MFPhoneFont.preview).foregroundStyle(MFColor.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            trailing
        }
        .padding(.horizontal, MFSpacing.l)
        .padding(.vertical, MFSpacing.compact + 1)
        .frame(minHeight: subtitle == nil ? MFPhoneMetrics.rowHeight : MFPhoneMetrics.rowHeightWithSubtitle)
        .accessibilityElement(children: .combine)
    }
}

extension PhoneRow where Trailing == EmptyView {
    init(title: String, subtitle: String? = nil) {
        self.init(title: title, subtitle: subtitle) { EmptyView() }
    }
}

/// A pill track with an ink selected pill (Day / Month / To-dos, the scan kinds).
struct SegmentedPills<Value: Hashable>: View {
    let options: [Value]
    @Binding var selection: Value
    let title: (Value) -> String
    var onDark = false

    var body: some View {
        HStack(spacing: 2) {
            ForEach(options, id: \.self) { option in
                let isSelected = option == selection
                Button {
                    withAnimation(.snappy(duration: 0.2)) { selection = option }
                } label: {
                    Text(title(option))
                        .font(MFPhoneFont.previewEmphasis)
                        .lineLimit(1)
                        .foregroundStyle(foreground(isSelected))
                        .padding(.horizontal, MFSpacing.m)
                        .frame(maxWidth: onDark ? nil : .infinity, minHeight: 34)
                        .background(Capsule().fill(background(isSelected)))
                        .contentShape(Capsule())
                }
                .buttonStyle(.plain)
                .accessibilityAddTraits(isSelected ? .isSelected : [])
            }
        }
        .padding(3)
        .background(Capsule().fill(onDark ? Color.clear : MFColor.fill))
    }

    private func foreground(_ selected: Bool) -> Color {
        if onDark { return selected ? Color(white: 0.09) : Color(white: 0.72) }
        return selected ? MFColor.onInk : MFColor.secondary
    }

    private func background(_ selected: Bool) -> Color {
        guard selected else { return .clear }
        return onDark ? .white : MFColor.ink
    }
}

/// Answer text with citation marks drawn as small accent chips.
struct CitedText: View {
    let text: String
    var font: Font = MFPhoneFont.body

    var body: some View {
        Text(attributed)
            .font(font)
            .foregroundStyle(MFColor.ink)
            .lineSpacing(MFLineSpacing.normal)
            .fixedSize(horizontal: false, vertical: true)
    }

    private var attributed: AttributedString {
        var result = AttributedString()
        for run in CitationText.runs(text) {
            switch run {
            case .text(let string):
                result += AttributedString(string)
            case .citation(let number):
                var mark = AttributedString("\u{2009}\(number)\u{2009}")
                mark.font = MFPhoneFont.citation
                mark.foregroundColor = MFColor.accent
                mark.backgroundColor = MFColor.accentFill
                mark.baselineOffset = 1
                result += AttributedString(" ") + mark
            }
        }
        return result
    }
}

/// The section pills at the top of the Duo's left pane (the Mac pill bar at
/// phone size).
struct SectionPillBar: View {
    @Binding var selection: MobileSection
    var onDark = false

    var body: some View {
        HStack(spacing: 2) {
            ForEach(MobileSection.allCases) { section in
                let isSelected = section == selection
                Button {
                    selection = section
                } label: {
                    HStack(spacing: MFSpacing.snug) {
                        Image(systemName: section.systemImage).font(.system(size: 15, weight: .medium))
                        if section != .settings { Text(section.title) }
                    }
                    .font(MFPhoneFont.previewEmphasis)
                    .foregroundStyle(foreground(isSelected))
                    .frame(maxWidth: section == .settings ? 52 : .infinity, minHeight: 36)
                    .background(Capsule().fill(background(isSelected)))
                    .contentShape(Capsule())
                }
                .buttonStyle(.plain)
                .accessibilityLabel(section.title)
                .accessibilityAddTraits(isSelected ? .isSelected : [])
            }
        }
        .padding(4)
        .background(Capsule().fill(onDark ? Color.white.opacity(0.14) : MFColor.fillSubtle))
        .accessibilityElement(children: .contain)
        .accessibilityLabel("Sections")
    }

    private func foreground(_ selected: Bool) -> Color {
        if onDark { return selected ? Color(white: 0.09) : Color(white: 0.72) }
        return selected ? MFColor.onInk : MFColor.secondary
    }

    private func background(_ selected: Bool) -> Color {
        guard selected else { return .clear }
        return onDark ? .white : MFColor.ink
    }
}

extension EventTint {
    var fill: Color {
        switch self {
        case .blue: MFColor.accentFill
        case .orange: MFColor.previewFill
        case .green: MFColor.successFill
        case .grey: MFColor.fill
        case .purple: Color.mf(light: 0xEEE8FF, dark: 0x2D2546)
        }
    }

    var text: Color {
        switch self {
        case .blue: MFColor.accentStrong
        case .orange: MFColor.preview
        case .green: MFColor.success
        case .grey: MFColor.ink
        case .purple: Color.mf(light: 0x4B2E9E, dark: 0xC9B8FF)
        }
    }
}

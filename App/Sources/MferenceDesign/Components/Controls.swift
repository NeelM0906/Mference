import SwiftUI

// Pills, buttons, toggles and keycaps from the mockup CSS (`.nv`, `.btn`,
// `.sw`, `.kbd`). Token-only: every color, font, spacing and radius comes from
// Tokens.swift or the component metrics below.

/// Component sizes taken from the mockup CSS that are not general tokens.
enum MFMetrics {
    /// `.nv` and `.btn` height.
    static let controlHeight: CGFloat = 30
    /// `.nv` horizontal padding.
    static let pillPadding: CGFloat = 12
    /// `.btn` horizontal padding.
    static let buttonPadding: CGFloat = 13
    /// `.sw` track and knob.
    static let toggleSize = CGSize(width: 40, height: 24)
    static let toggleKnob: CGFloat = 18
    /// `.kbd` minimum side.
    static let keycap: CGFloat = 20
    /// Chip height (`.chip`, 22-24 in the boards).
    static let chipHeight: CGFloat = 22
    /// `.cite` side.
    static let citation: CGFloat = 17
    /// `.bar` height in the model library.
    static let progressBar: CGFloat = 6
    /// Pill icons.
    static let iconSize: CGFloat = 13
}

extension MFFont {
    /// 13.5 semibold: list row titles (`.r .t b`).
    static let rowTitle = geist(13.5, .semibold)
    /// 12.5: list row subtitles (`.r .t span`).
    static let rowSubtitle = geist(12.5, .regular)
    /// 11.5 medium: chips and badges (`.chip`).
    static let chip = geist(11.5, .medium)
    /// 11 monospaced: keycaps (`.kbd`, Geist Mono in the mockup).
    static let keycap = Font.system(size: 11, weight: .medium, design: .monospaced)
    /// 10.5 semibold: citation numbers (`.cite`).
    static let citation = geist(10.5, .semibold)
}

// MARK: - Pill

/// A navigation pill (`.nv`): selected is ink filled with white text.
public struct MFPill: View {
    let title: String
    let systemImage: String?
    let isSelected: Bool
    let action: () -> Void

    public init(_ title: String, systemImage: String? = nil, isSelected: Bool, action: @escaping () -> Void) {
        self.title = title
        self.systemImage = systemImage
        self.isSelected = isSelected
        self.action = action
    }

    public var body: some View {
        Button(action: action) {
            HStack(spacing: MFSpacing.s - 2) {
                if let systemImage {
                    Image(systemName: systemImage)
                        .font(.system(size: MFMetrics.iconSize, weight: .medium))
                        .accessibilityHidden(true)
                }
                Text(title)
            }
        }
        .buttonStyle(MFPillButtonStyle(isSelected: isSelected))
        .accessibilityAddTraits(isSelected ? .isSelected : [])
    }
}

/// The `.nv` look for any button: capsule, 30 high, ink when selected.
public struct MFPillButtonStyle: ButtonStyle {
    let isSelected: Bool

    public init(isSelected: Bool = false) { self.isSelected = isSelected }

    public func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(MFFont.calloutEmphasis)
            .foregroundStyle(isSelected ? MFColor.onInk : MFColor.secondary)
            .padding(.horizontal, MFMetrics.pillPadding)
            .frame(height: MFMetrics.controlHeight)
            .background(Capsule().fill(isSelected ? MFColor.ink : (configuration.isPressed ? MFColor.hairline : .clear)))
            .contentShape(Capsule())
    }
}

// MARK: - Buttons

/// The mockup `.btn` (white capsule with a hairline border) and `.btn.pri`
/// (ink capsule).
public struct MFButtonStyle: ButtonStyle {
    public enum Kind: Sendable { case secondary, primary, quiet }

    let kind: Kind

    public init(_ kind: Kind = .secondary) { self.kind = kind }

    public func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(MFFont.calloutEmphasis)
            .foregroundStyle(kind == .primary ? MFColor.onInk : MFColor.ink)
            .padding(.horizontal, MFMetrics.buttonPadding)
            .frame(minHeight: MFMetrics.controlHeight)
            .background(Capsule().fill(fill(pressed: configuration.isPressed)))
            .overlay(Capsule().strokeBorder(kind == .secondary ? MFColor.hairlineStrong : .clear))
            .contentShape(Capsule())
            .opacity(configuration.isPressed ? 0.85 : 1)
    }

    private func fill(pressed: Bool) -> Color {
        switch kind {
        case .primary: MFColor.ink
        case .secondary: pressed ? MFColor.fill : MFColor.surface
        case .quiet: pressed ? MFColor.hairline : MFColor.fill
        }
    }
}

extension ButtonStyle where Self == MFButtonStyle {
    public static var mfSecondary: MFButtonStyle { MFButtonStyle(.secondary) }
    public static var mfPrimary: MFButtonStyle { MFButtonStyle(.primary) }
    public static var mfQuiet: MFButtonStyle { MFButtonStyle(.quiet) }
}

// MARK: - Toggle

/// The mockup switch (`.sw`): ink track when on, grey when off, white knob.
/// Keeps the label for accessibility and stays keyboard-operable.
public struct MFToggleStyle: ToggleStyle {
    public init() {}

    public func makeBody(configuration: Configuration) -> some View {
        Button {
            configuration.isOn.toggle()
        } label: {
            ZStack(alignment: configuration.isOn ? .trailing : .leading) {
                Capsule()
                    .fill(configuration.isOn ? MFColor.ink : MFColor.hairlineStrong)
                Circle()
                    .fill(MFColor.surface)
                    .frame(width: MFMetrics.toggleKnob, height: MFMetrics.toggleKnob)
                    .shadow(color: .black.opacity(0.25), radius: 1.5, y: 1)
                    .padding(.horizontal, (MFMetrics.toggleSize.height - MFMetrics.toggleKnob) / 2)
            }
            .frame(width: MFMetrics.toggleSize.width, height: MFMetrics.toggleSize.height)
            .animation(.snappy(duration: 0.15), value: configuration.isOn)
        }
        .buttonStyle(.plain)
        .accessibilityRepresentation {
            Toggle(isOn: configuration.$isOn) { configuration.label }
        }
    }
}

extension ToggleStyle where Self == MFToggleStyle {
    public static var mf: MFToggleStyle { MFToggleStyle() }
}

// MARK: - Keycap

/// A keyboard shortcut cap (`.kbd`), on a light surface or on ink (inside a
/// primary button).
public struct MFKeycap: View {
    let keys: String
    let onInk: Bool

    public init(_ keys: String, onInk: Bool = false) {
        self.keys = keys
        self.onInk = onInk
    }

    public var body: some View {
        Text(keys)
            .font(MFFont.keycap)
            .foregroundStyle(onInk ? MFColor.onInk : MFColor.secondary)
            .padding(.horizontal, MFSpacing.xs + 1)
            .frame(minWidth: MFMetrics.keycap, minHeight: MFMetrics.keycap)
            .background(RoundedRectangle(cornerRadius: MFRadius.xs).fill(onInk ? MFColor.secondary.opacity(0.35) : MFColor.surface))
            .overlay(RoundedRectangle(cornerRadius: MFRadius.xs).strokeBorder(onInk ? MFColor.tertiary : MFColor.hairlineStrong))
            .accessibilityLabel("Shortcut \(keys)")
    }
}

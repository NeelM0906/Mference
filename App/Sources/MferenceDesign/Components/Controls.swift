import SwiftUI

// Pills, buttons, toggles and keycaps from the mockup CSS (`.nv`, `.btn`,
// `.sw`, `.kbd`). Colors, fonts and spacing come from Tokens.swift; sizes that
// belong to one component are named in `MFMetrics`.

/// Component sizes taken from the mockup CSS that are not shared tokens.
enum MFMetrics {
    /// `.nv` and `.btn` height.
    static let controlHeight: CGFloat = 30
    /// `.nv` horizontal padding.
    static let pillPadding: CGFloat = 12
    /// `.btn` horizontal padding.
    static let buttonPadding: CGFloat = 13
    /// First-run buttons: 34 and 38 high, the large one padded 18.
    static let mediumControlHeight: CGFloat = 34
    static let largeControlHeight: CGFloat = 38
    static let largeButtonPadding: CGFloat = 18
    /// `.sw` track and knob.
    static let toggleSize = CGSize(width: 40, height: 24)
    static let toggleKnob: CGFloat = 18
    static let toggleKnobShadow: Double = 0.25
    /// `.kbd` minimum side and horizontal padding.
    static let keycap: CGFloat = 20
    static let keycapPadding: CGFloat = 5
    /// Keycap fill on an ink button.
    static let keycapOnInkFill: Double = 0.35
    /// Chip height (`.chip`) and its icon gap.
    static let chipHeight: CGFloat = 22
    static let chipIconGap: CGFloat = 5
    /// `.cite` side.
    static let citation: CGFloat = 17
    /// Numbered source chip height.
    static let sourceChipHeight: CGFloat = 28
    /// `.bar` height in the model library.
    static let progressBar: CGFloat = 6
    /// Pill icons, and the smaller chip icons.
    static let iconSize: CGFloat = 13
    static let chipIconSize: CGFloat = 11
    /// Padding inside the grey pill track.
    static let pillTrackPadding: CGFloat = 3
    /// A pressed secondary or primary button.
    static let pressedOpacity: Double = 0.85
    /// Any disabled control.
    static let disabledOpacity: Double = 0.5
    /// White over the material: the mockup window is white at about 90%.
    static let windowTint: Double = 0.88
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
            HStack(spacing: MFSpacing.snug) {
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
        PillBody(configuration: configuration, isSelected: isSelected)
    }

    private struct PillBody: View {
        let configuration: Configuration
        let isSelected: Bool
        @Environment(\.isEnabled) private var isEnabled

        var body: some View {
            configuration.label
                .font(MFFont.calloutEmphasis)
                .foregroundStyle(isSelected ? MFColor.onInk : MFColor.secondary)
                .padding(.horizontal, MFMetrics.pillPadding)
                .frame(height: MFMetrics.controlHeight)
                .background(Capsule().fill(isSelected ? MFColor.ink : (configuration.isPressed ? MFColor.hairline : .clear)))
                .contentShape(Capsule())
                .opacity(isEnabled ? 1 : MFMetrics.disabledOpacity)
        }
    }
}

// MARK: - Buttons

/// The mockup `.btn` (white capsule with a hairline border) and `.btn.pri`
/// (ink capsule).
public struct MFButtonStyle: ButtonStyle {
    public enum Kind: Sendable { case secondary, primary, quiet }

    /// `.regular` is the app's `.btn` (30 high, 13pt); `.medium` (34, 13pt)
    /// and `.large` (38, 14pt) are the first-run buttons.
    public enum Size: Sendable {
        case regular, medium, large

        var height: CGFloat {
            switch self {
            case .regular: MFMetrics.controlHeight
            case .medium: MFMetrics.mediumControlHeight
            case .large: MFMetrics.largeControlHeight
            }
        }

        var padding: CGFloat {
            switch self {
            case .regular: MFMetrics.buttonPadding
            case .medium: MFSpacing.l
            case .large: MFMetrics.largeButtonPadding
            }
        }

        var font: Font { self == .large ? MFFont.buttonLarge : MFFont.calloutEmphasis }
    }

    let kind: Kind
    let size: Size

    public init(_ kind: Kind = .secondary, size: Size = .regular) {
        self.kind = kind
        self.size = size
    }

    public func makeBody(configuration: Configuration) -> some View {
        ButtonBody(configuration: configuration, kind: kind, size: size)
    }

    private struct ButtonBody: View {
        let configuration: Configuration
        let kind: Kind
        let size: Size
        @Environment(\.isEnabled) private var isEnabled

        var body: some View {
            configuration.label
                .font(size.font)
                .foregroundStyle(kind == .primary ? MFColor.onInk : MFColor.ink)
                .padding(.horizontal, size.padding)
                .frame(minHeight: size.height)
                .background(Capsule().fill(fill))
                .overlay(Capsule().strokeBorder(kind == .secondary ? MFColor.hairlineStrong : .clear,
                                                lineWidth: MFLayout.hairline))
                .contentShape(Capsule())
                .opacity(!isEnabled ? MFMetrics.disabledOpacity : configuration.isPressed ? MFMetrics.pressedOpacity : 1)
        }

        private var fill: Color {
            switch kind {
            case .primary: MFColor.ink
            case .secondary: configuration.isPressed ? MFColor.fill : MFColor.surface
            case .quiet: configuration.isPressed ? MFColor.hairline : MFColor.fill
            }
        }
    }
}

extension ButtonStyle where Self == MFButtonStyle {
    public static var mfSecondary: MFButtonStyle { MFButtonStyle(.secondary) }
    public static var mfPrimary: MFButtonStyle { MFButtonStyle(.primary) }
    public static var mfQuiet: MFButtonStyle { MFButtonStyle(.quiet) }
}

// MARK: - Toggle

/// The mockup switch (`.sw`): ink track when on; grey track with an outline
/// when off, so its edge keeps 3:1 against the surface. White knob. Keeps the
/// label for accessibility, stays keyboard-operable, dims when disabled.
public struct MFToggleStyle: ToggleStyle {
    public init() {}

    public func makeBody(configuration: Configuration) -> some View {
        ToggleBody(configuration: configuration)
    }

    private struct ToggleBody: View {
        let configuration: Configuration
        @Environment(\.isEnabled) private var isEnabled

        var body: some View {
            Button {
                configuration.isOn.toggle()
            } label: {
                ZStack(alignment: configuration.isOn ? .trailing : .leading) {
                    Capsule()
                        .fill(configuration.isOn ? MFColor.ink : MFColor.toggleOff)
                    if !configuration.isOn {
                        Capsule().strokeBorder(MFColor.controlBorder, lineWidth: MFLayout.hairline)
                    }
                    Circle()
                        .fill(MFColor.surface)
                        .frame(width: MFMetrics.toggleKnob, height: MFMetrics.toggleKnob)
                        .shadow(color: .black.opacity(MFMetrics.toggleKnobShadow), radius: 1.5, y: 1)
                        .padding(.horizontal, (MFMetrics.toggleSize.height - MFMetrics.toggleKnob) / 2)
                }
                .frame(width: MFMetrics.toggleSize.width, height: MFMetrics.toggleSize.height)
                .opacity(isEnabled ? 1 : MFMetrics.disabledOpacity)
                .animation(.snappy(duration: 0.15), value: configuration.isOn)
            }
            .buttonStyle(.plain)
            .accessibilityRepresentation {
                Toggle(isOn: configuration.$isOn) { configuration.label }
            }
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
            .padding(.horizontal, MFMetrics.keycapPadding)
            .frame(minWidth: MFMetrics.keycap, minHeight: MFMetrics.keycap)
            .background(RoundedRectangle(cornerRadius: MFRadius.xs)
                .fill(onInk ? MFColor.secondary.opacity(MFMetrics.keycapOnInkFill) : MFColor.surface))
            .overlay(RoundedRectangle(cornerRadius: MFRadius.xs)
                .strokeBorder(onInk ? MFColor.tertiary : MFColor.hairlineStrong, lineWidth: MFLayout.hairline))
            .accessibilityLabel("Shortcut \(keys)")
    }
}

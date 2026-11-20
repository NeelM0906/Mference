import MferenceAppCore
import MferenceDesign
import SwiftUI

/// The dictation bar (board 2, `Dictation.dc.html` `.dbar`): it floats at
/// the bottom of the editor while dictating, with the level meter,
/// "Listening", "Text lands after each pause", the language chip, the Clean
/// up fillers chip and Done (Esc). N1's editor overlays it with
/// `.overlay(alignment: .bottom)` while `controller.isActive`, or while an
/// error is showing.
public struct DictationBar: View {
    @Bindable var controller: DictationController

    public init(controller: DictationController) { self.controller = controller }

    public var body: some View {
        HStack(spacing: MFSpacing.m) {
            LevelMeter(level: controller.level, isActive: controller.phase == .listening)
            VStack(alignment: .leading, spacing: MFSpacing.xxs) {
                Text(Self.title(controller.phase, error: controller.error))
                    .font(MFFont.rowTitle)
                    .foregroundStyle(controller.error == nil ? MFColor.ink : MFColor.danger)
                Text(Self.detail(controller.phase, error: controller.error))
                    .font(MFFont.caption)
                    .foregroundStyle(MFColor.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .accessibilityElement(children: .combine)
            languageChip
            cleanUpChip
            Button {
                Task { await done() }
            } label: {
                HStack(spacing: MFSpacing.s) {
                    Text(controller.isActive ? "Done" : "Close")
                    MFKeycap("esc", onInk: true)
                }
            }
            .buttonStyle(MFButtonStyle(.primary, size: .medium))
            .keyboardShortcut(.cancelAction)
            .accessibilityLabel(controller.isActive ? "Stop dictating" : "Close")
        }
        .padding(.horizontal, MFSpacing.comfortable)
        .padding(.vertical, MFSpacing.m)
        .frame(width: DictationMetrics.barWidth)
        .background(RoundedRectangle(cornerRadius: MFRadius.xl).fill(MFColor.surface))
        .overlay(RoundedRectangle(cornerRadius: MFRadius.xl).strokeBorder(MFColor.hairlineStrong, lineWidth: MFLayout.hairline))
        .shadow(color: MFColor.ink.opacity(DictationMetrics.shadowOpacity), radius: DictationMetrics.shadowRadius,
                y: DictationMetrics.shadowY)
        .padding(.bottom, DictationMetrics.bottomInset)
        .accessibilityElement(children: .contain)
        .accessibilityLabel("Dictation")
    }

    private func done() async {
        if controller.isActive {
            await controller.stop()
        } else {
            controller.clearError()
        }
    }

    static func title(_ phase: DictationController.Phase, error: SpeechError?) -> String {
        if error != nil { return "Dictation stopped" }
        switch phase {
        case .idle, .listening: return "Listening"
        case .starting: return "Starting…"
        case .stopping: return "Finishing the last phrase…"
        }
    }

    static func detail(_ phase: DictationController.Phase, error: SpeechError?) -> String {
        if let error { return error.message }
        return "Text lands after each pause"
    }

    // MARK: Chips

    private var languageChip: some View {
        Menu {
            Button("Automatic") { Task { await controller.setLanguage(.automatic) } }
            Divider()
            ForEach(VoiceFormat.languages, id: \.code) { language in
                Button(language.name) { Task { await controller.setLanguage(.fixed(language.code)) } }
            }
        } label: {
            DictationChip(VoiceFormat.languageChip(controller.preferences.language,
                                                   detected: controller.detectedLanguage),
                          isOn: false)
        }
        .menuStyle(.button)
        .buttonStyle(.plain)
        .menuIndicator(.hidden)
        .fixedSize()
        .accessibilityLabel("Language")
        .accessibilityValue(VoiceFormat.languageChip(controller.preferences.language,
                                                     detected: controller.detectedLanguage))
        .help("Pin a language, or go back to Automatic")
    }

    private var cleanUpChip: some View {
        Button {
            controller.setCleanUpFillers(!controller.preferences.cleanUpFillers)
        } label: {
            DictationChip("Clean up fillers", isOn: controller.preferences.cleanUpFillers)
        }
        .buttonStyle(.plain)
        .accessibilityLabel("Clean up fillers")
        .accessibilityValue(controller.preferences.cleanUpFillers ? "On" : "Off")
        .accessibilityAddTraits(.isToggle)
        .help("Drops um, uh and repeated starts. Same switch as in Settings › Speech.")
    }
}

/// A bar chip (`.chip`, `.chip.on`): grey, or accent when on.
struct DictationChip: View {
    let title: String
    let isOn: Bool

    init(_ title: String, isOn: Bool) {
        self.title = title
        self.isOn = isOn
    }

    var body: some View {
        Text(title)
            .font(MFFont.chip)
            .lineLimit(1)
            .foregroundStyle(isOn ? MFColor.accentStrong : MFColor.secondary)
            .padding(.horizontal, DictationMetrics.chipPadding)
            .frame(height: DictationMetrics.chipHeight)
            .background(Capsule().fill(isOn ? MFColor.accentFill : MFColor.fillSubtle))
            .contentShape(Capsule())
    }
}

/// The level pill (`.lvl`): five bars that rise with the input level.
struct LevelMeter: View {
    let level: Float
    let isActive: Bool

    /// Each bar's share of the level, so the meter reads as a waveform.
    static let shape: [CGFloat] = [0.45, 0.8, 1, 0.7, 0.4]

    var body: some View {
        HStack(spacing: DictationMetrics.meterGap) {
            ForEach(Array(Self.shape.enumerated()), id: \.offset) { _, share in
                RoundedRectangle(cornerRadius: DictationMetrics.meterBarRadius)
                    .fill(MFColor.accent.opacity(isActive ? 1 : 0.4))
                    .frame(width: DictationMetrics.meterBar,
                           height: Self.height(level: level, share: share))
            }
        }
        .frame(width: DictationMetrics.meterSize.width, height: DictationMetrics.meterSize.height)
        .background(Capsule().fill(MFColor.accentFillFaint))
        .animation(.easeOut(duration: 0.1), value: level)
        .accessibilityHidden(true)
    }

    /// From a resting 4 pt to the full 18 pt. Speech sits around 0.3–0.6,
    /// so the curve lifts low levels to read as a waveform.
    static func height(level: Float, share: CGFloat) -> CGFloat {
        let clamped = CGFloat(min(max(level, 0), 1))
        let lifted = clamped == 0 ? 0 : clamped.squareRoot()
        return DictationMetrics.meterRest + (DictationMetrics.meterPeak - DictationMetrics.meterRest) * lifted * share
    }
}

/// Dictation bar sizes from `Dictation.dc.html` (`.dbar`, `.lvl`, `.chip`).
enum DictationMetrics {
    static let barWidth: CGFloat = 560
    static let bottomInset: CGFloat = 22
    static let shadowOpacity: Double = 0.14
    static let shadowRadius: CGFloat = 20
    static let shadowY: CGFloat = 14
    static let meterSize = CGSize(width: 56, height: 28)
    static let meterBar: CGFloat = 3
    static let meterBarRadius: CGFloat = 2
    static let meterGap: CGFloat = 3
    static let meterRest: CGFloat = 4
    static let meterPeak: CGFloat = 18
    static let chipHeight: CGFloat = 24
    static let chipPadding: CGFloat = 9
}

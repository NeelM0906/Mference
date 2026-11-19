import MferenceAppCore
import MferenceDesign
import SwiftUI

/// The per-chat model picker (mockup 02b, decisions L2/L3): Recommended
/// (Quick, Deep, Auto "Coming later") and Also on this Mac (Qwen 3.6,
/// MiniCPM5, Maple), each with its live state and, small under it, the engine.
struct ModelPicker: View {
    @Bindable var model: ChatViewModel
    @Binding var isOpen: Bool
    let actions: ChatActions

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 0) {
                header("Recommended")
                ForEach(AppModel.allCases.filter { $0.group == .recommended }) { item in
                    row(item)
                }
                AutoRow()
                divider
                header("Also on this Mac")
                ForEach(AppModel.allCases.filter { $0.group == .alsoOnThisMac }) { item in
                    row(item)
                }
                divider
                HStack(spacing: MFSpacing.s) {
                    Text("Switching unloads the current model first.")
                        .foregroundStyle(MFColor.secondary)
                    Spacer(minLength: 0)
                    Button("Manage models") {
                        isOpen = false
                        actions.openModelLibrary()
                    }
                    .buttonStyle(.plain)
                    .foregroundStyle(MFColor.accent)
                    .font(MFFont.caption.weight(.medium))
                }
                .font(MFFont.caption)
                .padding(.horizontal, MFSpacing.s + MFSpacing.xxs)
                .padding(.top, MFSpacing.s + MFSpacing.xxs)
                .padding(.bottom, MFSpacing.xs + MFSpacing.xxs)
            }
            .padding(MFSpacing.xs + MFSpacing.xxs)
        }
        .scrollBounceBehavior(.basedOnSize)
        .frame(width: ChatMetrics.pickerWidth)
        .frame(maxHeight: ChatMetrics.pickerMaxHeight)
        .fixedSize(horizontal: false, vertical: true)
        .background(MFColor.surface, in: RoundedRectangle(cornerRadius: MFRadius.l))
        .overlay(RoundedRectangle(cornerRadius: MFRadius.l).strokeBorder(MFColor.hairline))
        .shadow(color: MFColor.ink.opacity(0.16), radius: MFSpacing.xl, y: MFSpacing.l)
        .accessibilityElement(children: .contain)
        .accessibilityLabel("Model")
        .accessibilityAddTraits(.isModal)
    }

    private func header(_ title: String) -> some View {
        Text(title.uppercased())
            .font(MFFont.sectionHeader)
            .tracking(MFFont.sectionHeaderTracking)
            .foregroundStyle(MFColor.tertiary)
            .padding(.horizontal, MFSpacing.s + MFSpacing.xxs)
            .padding(.top, MFSpacing.s + MFSpacing.xxs)
            .padding(.bottom, MFSpacing.xs)
            .accessibilityAddTraits(.isHeader)
    }

    private var divider: some View {
        Rectangle()
            .fill(MFColor.hairlineFaint)
            .frame(height: ChatMetrics.hairlineWidth)
            .padding(.top, MFSpacing.xs)
    }

    private func row(_ item: AppModel) -> some View {
        let option = model.option(for: item)
        let availability = option?.availability ?? .notInstalled(reason: "Not installed on this Mac.")
        return ModelRow(model: item, availability: availability, isSelected: item == model.model) {
            model.selectModel(item)
            isOpen = false
        }
    }
}

struct ModelRow: View {
    let model: AppModel
    let availability: ModelAvailability
    let isSelected: Bool
    let onPick: () -> Void
    @State private var isHovered = false

    private var isInstalled: Bool {
        if case .notInstalled = availability { return false }
        return true
    }

    var body: some View {
        Button(action: onPick) {
            HStack(alignment: .top, spacing: MFSpacing.s + MFSpacing.xxs) {
                RadioDot(isOn: isSelected)
                VStack(alignment: .leading, spacing: MFSpacing.xxs) {
                    HStack(alignment: .firstTextBaseline, spacing: MFSpacing.xs) {
                        Text(model.name)
                            .font(MFFont.rowTitle)
                            .foregroundStyle(MFColor.ink)
                        Text(model.detail)
                            .font(MFFont.chip)
                            .foregroundStyle(model.isPreview ? MFColor.preview : MFColor.tertiary)
                    }
                    Text(model.summary)
                        .font(MFFont.caption)
                        .foregroundStyle(MFColor.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                Spacer(minLength: MFSpacing.s)
                VStack(alignment: .trailing, spacing: MFSpacing.xxs + 1) {
                    Text(statusText)
                        .font(MFFont.chip)
                        .foregroundStyle(availability == .loaded ? MFColor.success : MFColor.tertiary)
                    Text(model.engineLabel)
                        .font(MFFont.micro.weight(.regular))
                        .foregroundStyle(MFColor.tertiary)
                }
                .padding(.top, MFSpacing.xxs)
            }
            .padding(MFSpacing.s + MFSpacing.xxs)
            .background(background, in: RoundedRectangle(cornerRadius: MFRadius.m))
            .contentShape(RoundedRectangle(cornerRadius: MFRadius.m))
            .opacity(isInstalled ? 1 : 0.6)
        }
        .buttonStyle(.plain)
        .disabled(!isInstalled)
        .onHover { isHovered = $0 }
        .help(helpText)
        .accessibilityLabel("\(model.name), \(model.detail)")
        .accessibilityValue(statusText)
        .accessibilityHint(model.summary)
        .accessibilityAddTraits(isSelected ? .isSelected : [])
    }

    private var background: Color {
        if isSelected { return MFColor.accentFillFaint }
        return isHovered && isInstalled ? MFColor.fill : .clear
    }

    private var statusText: String {
        switch availability {
        case .loaded: "Loaded"
        case .ready: "Ready"
        case .loading: "Loading"
        case .unloading: "Unloading"
        case .notInstalled: "Not installed"
        }
    }

    private var helpText: String {
        if case .notInstalled(let reason) = availability { return reason }
        if availability == .loading { return "Loading cannot be interrupted." }
        return model.summary
    }
}

/// Auto: shown, not choosable yet (decision L3).
struct AutoRow: View {
    var body: some View {
        HStack(alignment: .top, spacing: MFSpacing.s + MFSpacing.xxs) {
            RadioDot(isOn: false)
            VStack(alignment: .leading, spacing: MFSpacing.xxs) {
                HStack(alignment: .firstTextBaseline, spacing: MFSpacing.xs) {
                    Text("Auto")
                        .font(MFFont.rowTitle)
                        .foregroundStyle(MFColor.ink)
                    Text("Preview")
                        .font(MFFont.chip)
                        .foregroundStyle(MFColor.preview)
                }
                Text("Coming later. For now you choose the model yourself.")
                    .font(MFFont.caption)
                    .foregroundStyle(MFColor.secondary)
            }
            Spacer(minLength: 0)
        }
        .padding(MFSpacing.s + MFSpacing.xxs)
        .accessibilityElement(children: .combine)
        .accessibilityLabel("Auto, preview. Coming later. For now you choose the model yourself.")
        .accessibilityAddTraits(.isStaticText)
    }
}

struct RadioDot: View {
    let isOn: Bool

    var body: some View {
        Circle()
            .strokeBorder(MFColor.tertiary, lineWidth: ChatMetrics.radioStroke)
            .frame(width: ChatMetrics.radioSize, height: ChatMetrics.radioSize)
            .overlay {
                if isOn {
                    Circle().fill(MFColor.accent).frame(width: ChatMetrics.radioDot, height: ChatMetrics.radioDot)
                }
            }
            .padding(.top, MFSpacing.xxs)
            .accessibilityHidden(true)
    }
}

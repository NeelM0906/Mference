import MferenceAppCore
import MferenceDesign
import SwiftUI

/// The per-chat model picker: Auto ("Coming later") on top, then the models
/// grouped by size (Small, Large, Mixture of experts) or by family, as the
/// switch at the top says (`ModelPickerGrouping`, remembered). Quick and Deep
/// carry a Recommended tag in their group. Apple's on-device model shows only
/// while Settings > Models > Apple Intelligence model is on and this Mac can
/// run it. Each row has its live state and, small under it, the engine.
struct ModelPicker: View {
    @Bindable var model: ChatViewModel
    @Binding var isOpen: Bool
    let actions: ChatActions
    @Environment(\.openURL) private var openURL
    @AppStorage(ModelPickerGrouping.storageKey) private var grouping: ModelPickerGrouping = .size

    /// Every model the picker lists: Apple's only while it is offered.
    private var offeredModels: [AppModel] {
        AppModel.allCases.filter { $0 != .apple || model.option(for: .apple) != nil }
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 0) {
                Picker("Group models", selection: $grouping) {
                    ForEach(ModelPickerGrouping.allCases) { option in
                        Text(option.title).tag(option)
                    }
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                .padding(.horizontal, MFSpacing.s + MFSpacing.xxs)
                .padding(.top, MFSpacing.s)
                .padding(.bottom, MFSpacing.xxs)
                AutoRow()
                ForEach(grouping.sections(for: offeredModels)) { section in
                    divider
                    header(section.title, hint: section.hint)
                    ForEach(section.models) { item in
                        if item == .apple, let apple = model.option(for: .apple) {
                            appleRow(apple)
                        } else {
                            row(item)
                        }
                    }
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
        // Apple Intelligence may have been turned on or finished preparing
        // since the list was read.
        .task { model.refreshModels() }
    }

    /// Apple's row: dimmed with its reason while Apple Intelligence is off or
    /// getting ready, with a way to the System Settings pane when off.
    @ViewBuilder
    private func appleRow(_ option: ModelOption) -> some View {
        let unavailableReason: String? = if case .notInstalled(let reason) = option.availability { reason } else { nil }
        ModelRow(model: .apple, availability: option.availability, isSelected: model.model == .apple,
                 statusOverride: unavailableReason.map(AppleChatRouting.status(forUnavailable:)),
                 summaryOverride: unavailableReason) {
            model.selectModel(.apple)
            isOpen = false
        }
        if unavailableReason == AppleChatRouting.turnedOffReason {
            Button("Open System Settings") {
                openURL(AppleChatRouting.systemSettingsURL)
            }
            .buttonStyle(.plain)
            .font(MFFont.caption.weight(.medium))
            .foregroundStyle(MFColor.accent)
            .padding(.leading, ChatMetrics.radioSize + 2 * (MFSpacing.s + MFSpacing.xxs))
            .padding(.bottom, MFSpacing.s)
            .accessibilityHint("Opens Apple Intelligence & Siri in System Settings")
        }
    }

    private func header(_ title: String, hint: String? = nil) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: MFSpacing.s) {
            Text(title.uppercased())
                .font(MFFont.sectionHeader)
                .tracking(MFFont.sectionHeaderTracking)
            if let hint {
                Text(hint)
                    .font(MFFont.caption)
            }
        }
        .foregroundStyle(MFColor.tertiary)
        .padding(.horizontal, MFSpacing.s + MFSpacing.xxs)
        .padding(.top, MFSpacing.s + MFSpacing.xxs)
        .padding(.bottom, MFSpacing.xs)
        .accessibilityElement(children: .combine)
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
    /// Replaces "Not installed" (Apple's row: "Turned off", "Getting ready").
    var statusOverride: String?
    /// Replaces the summary line (Apple's row: why it cannot run).
    var summaryOverride: String?
    let onPick: () -> Void
    @State private var isHovered = false

    /// Quick and Deep: tagged in their group instead of a section of their own.
    private var isRecommended: Bool { model.group == .recommended }

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
                        if isRecommended {
                            Text("Recommended")
                                .font(MFFont.micro)
                                .foregroundStyle(MFColor.accent)
                                .padding(.horizontal, MFSpacing.xs)
                                .padding(.vertical, MFSpacing.xxs / 2)
                                .background(MFColor.accentFillFaint, in: RoundedRectangle(cornerRadius: MFRadius.xs))
                        }
                    }
                    Text(summaryOverride ?? model.summary)
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
        .accessibilityLabel("\(model.name), \(model.detail)\(isRecommended ? ", recommended" : "")")
        .accessibilityValue(statusText)
        .accessibilityHint(summaryOverride ?? model.summary)
        .accessibilityAddTraits(isSelected ? .isSelected : [])
    }

    private var background: Color {
        if isSelected { return MFColor.accentFillFaint }
        return isHovered && isInstalled ? MFColor.fill : .clear
    }

    private var statusText: String {
        if let statusOverride { return statusOverride }
        return switch availability {
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

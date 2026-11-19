import MferenceDesign
import SwiftUI

/// The composer: text on top; attach, "Web via Mac", the model chip and
/// send / dictate below. Same on every screen size.
struct Composer: View {
    @Bindable var model: MobileAppModel
    @FocusState private var isFocused: Bool

    private var hasDraft: Bool { !model.draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }

    var body: some View {
        VStack(alignment: .leading, spacing: MFSpacing.compact) {
            TextField("Ask anything. It stays on this iPhone.", text: $model.draft, axis: .vertical)
                .font(MFPhoneFont.body)
                .foregroundStyle(MFColor.ink)
                .lineLimit(1...6)
                .focused($isFocused)
                .submitLabel(.send)
                .onSubmit { model.send() }
                .padding(.top, 2)
            HStack(spacing: MFSpacing.s) {
                Menu {
                    Button("Photo", systemImage: "photo") {}
                    Button("File", systemImage: "doc") {}
                    Button("Scan a document", systemImage: "viewfinder") { model.selection = .scan }
                } label: {
                    Image(systemName: "plus")
                        .font(.system(size: 15, weight: .semibold))
                        .foregroundStyle(MFColor.secondary)
                        .frame(width: MFPhoneMetrics.composerControl, height: MFPhoneMetrics.composerControl)
                        .background(Circle().fill(MFColor.fill))
                }
                .accessibilityLabel("Add a photo or file")

                if model.pairedMac != nil {
                    webChip
                }

                Spacer(minLength: 0)

                Button {
                    model.isModelPickerPresented = true
                } label: {
                    HStack(spacing: 4) {
                        Text(model.selectedModel.shortName)
                        Image(systemName: "chevron.down").font(.system(size: 11, weight: .bold))
                    }
                    .font(MFPhoneFont.control)
                    .foregroundStyle(MFColor.ink)
                    .padding(.leading, MFSpacing.m)
                    .padding(.trailing, MFSpacing.compact)
                    .frame(height: MFPhoneMetrics.composerControl)
                    .background(Capsule().fill(MFColor.fill))
                }
                .buttonStyle(.plain)
                .accessibilityLabel("Model: \(model.selectedModel.name)")

                Button {
                    if hasDraft { model.send() } else { isFocused = true }
                } label: {
                    Image(systemName: hasDraft ? "arrow.up" : "mic")
                        .font(.system(size: 16, weight: .semibold))
                        .foregroundStyle(MFColor.onInk)
                        .frame(width: MFPhoneMetrics.sendButton, height: MFPhoneMetrics.sendButton)
                        .background(Circle().fill(MFColor.ink))
                        .contentTransition(.symbolEffect(.replace))
                }
                .buttonStyle(.plain)
                .accessibilityLabel(hasDraft ? "Send" : "Dictate")
            }
        }
        .padding(.leading, 14)
        .padding(.trailing, MFSpacing.compact)
        .padding(.top, MFSpacing.m)
        .padding(.bottom, MFSpacing.compact)
        .background(RoundedRectangle(cornerRadius: MFPhoneMetrics.composerRadius).fill(MFColor.surface)
            .shadow(color: .black.opacity(0.06), radius: 10, y: 4))
        .overlay(RoundedRectangle(cornerRadius: MFPhoneMetrics.composerRadius)
            .strokeBorder(MFColor.hairline, lineWidth: MFLayout.hairline))
        .sensoryFeedback(.impact(weight: .light), trigger: model.currentMessages.count)
    }

    /// "Web via Mac" when on, "Web off" when off. Searches go through
    /// SearXNG on the paired Mac.
    private var webChip: some View {
        Button {
            model.webSearchViaMac.toggle()
        } label: {
            HStack(spacing: 5) {
                Image(systemName: "globe").font(.system(size: 13, weight: .semibold))
                Text(model.webSearchViaMac ? "Web via Mac" : "Web off")
            }
            .font(MFPhoneFont.control)
            .foregroundStyle(model.webSearchViaMac ? MFColor.accent : MFColor.secondary)
            .padding(.horizontal, 11)
            .frame(height: MFPhoneMetrics.composerControl)
            .background(Capsule().fill(model.webSearchViaMac ? MFColor.accentFill : MFColor.surface))
            .overlay(Capsule().strokeBorder(model.webSearchViaMac ? Color.clear : MFColor.hairline,
                                            lineWidth: MFLayout.hairline))
        }
        .buttonStyle(.plain)
        .accessibilityLabel("Web search via paired Mac")
        .accessibilityValue(model.webSearchViaMac ? "On" : "Off")
    }
}

/// "Better on your Mac": offered, sent, or kept on the iPhone.
struct HandoffCard: View {
    let offer: HandoffOffer
    let macName: String
    let modelName: String
    let choose: (Bool) -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: MFSpacing.m) {
            HStack(spacing: MFSpacing.m) {
                Image(systemName: "laptopcomputer")
                    .font(.system(size: 20, weight: .regular))
                    .foregroundStyle(MFColor.ink)
                    .frame(width: 44, height: 44)
                    .background(RoundedRectangle(cornerRadius: 12).fill(MFColor.fill))
                    .accessibilityHidden(true)
                VStack(alignment: .leading, spacing: 2) {
                    Text(title)
                        .font(MFPhoneFont.headline)
                        .foregroundStyle(MFColor.ink)
                    Text("\(macName) · same Wi-Fi · \(modelName) ready")
                        .font(MFPhoneFont.caption)
                        .foregroundStyle(MFColor.secondary)
                }
            }
            switch offer.state {
            case .offered:
                Text(offer.reason)
                    .font(MFPhoneFont.callout)
                    .foregroundStyle(MFColor.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                Button("Send to my Mac") { choose(true) }
                    .buttonStyle(PhoneButtonStyle(kind: .primary, size: .large, fullWidth: true))
                Button("Try on iPhone anyway") { choose(false) }
                    .buttonStyle(PhoneButtonStyle(kind: .secondary, fullWidth: true))
                Text("Encrypted between your devices. Nothing goes through a server.")
                    .font(MFPhoneFont.caption)
                    .foregroundStyle(MFColor.tertiary)
                    .frame(maxWidth: .infinity)
                    .multilineTextAlignment(.center)
            case .sentToMac:
                HStack(spacing: MFSpacing.s) {
                    ProgressView().controlSize(.small)
                    Text("Sent to \(macName). The result comes back here.")
                        .font(MFPhoneFont.preview)
                        .foregroundStyle(MFColor.secondary)
                }
            case .returnedFromMac:
                Label("Came back from \(macName).", systemImage: "checkmark.circle.fill")
                    .font(MFPhoneFont.preview)
                    .foregroundStyle(MFColor.success)
            case .keptOnIPhone:
                Text("Trying on this iPhone with Quick. Long documents may be cut short.")
                    .font(MFPhoneFont.preview)
                    .foregroundStyle(MFColor.secondary)
            }
        }
        .padding(MFSpacing.l)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(RoundedRectangle(cornerRadius: MFRadius.l).fill(MFColor.surface))
        .overlay(RoundedRectangle(cornerRadius: MFRadius.l).strokeBorder(MFColor.hairline, lineWidth: MFLayout.hairline))
        .sensoryFeedback(.warning, trigger: offer.state == .offered)
    }

    private var title: String {
        switch offer.state {
        case .offered, .keptOnIPhone: "Better on your Mac"
        case .sentToMac: "Working on your Mac"
        case .returnedFromMac: "Done on your Mac"
        }
    }
}

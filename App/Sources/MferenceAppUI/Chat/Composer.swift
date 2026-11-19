import MferenceAppCore
import MferenceDesign
import SwiftUI

/// The message box: text, +, Web on/off, model button, send / stop.
struct Composer: View {
    @Bindable var model: ChatViewModel
    @Binding var pickerOpen: Bool
    @FocusState private var focused: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: MFSpacing.s) {
            TextField(model.webEnabled && model.webBlockedReason == nil ? "Ask anything. Web searches go out through SearXNG."
                                        : "Ask anything. It stays on this Mac.",
                      text: $model.draft, axis: .vertical)
                .textFieldStyle(.plain)
                .font(MFFont.body)
                .foregroundStyle(MFColor.ink)
                .lineLimit(ChatMetrics.composerLines)
                .focused($focused)
                .onSubmit { model.send() }
                .accessibilityLabel("Message")
            HStack(spacing: MFSpacing.xs + MFSpacing.xxs) {
                moreMenu
                webChip
                Spacer(minLength: MFSpacing.s)
                modelButton
                sendButton
            }
        }
        .padding(.leading, MFSpacing.l)
        .padding(.trailing, MFSpacing.m)
        .padding(.top, MFSpacing.m)
        .padding(.bottom, MFSpacing.s + MFSpacing.xxs)
        .background(MFColor.surface, in: RoundedRectangle(cornerRadius: MFRadius.xl))
        .overlay(RoundedRectangle(cornerRadius: MFRadius.xl).strokeBorder(MFColor.hairlineStrong))
        .shadow(color: MFColor.ink.opacity(0.05), radius: MFSpacing.m, y: MFSpacing.xs)
        .onAppear { focused = true }
        .confirmationDialog("Turn on web search?", isPresented: Binding(
            get: { model.webConsentRequested },
            set: { if !$0 { model.declineWebConsent() } })) {
            Button("Turn on web search") { model.confirmWebConsent() }
            Button("Not now", role: .cancel) { model.declineWebConsent() }
        } message: {
            Text("The model still runs on this Mac. When it searches, its queries, which can include words from your chat, go to search engines through SearXNG on this Mac, and the pages it reads are fetched from this Mac. You can turn this off in Settings > Privacy.")
        }
        .onKeyPress(.escape) {
            guard pickerOpen else { return .ignored }
            pickerOpen = false
            return .handled
        }
    }

    private var moreMenu: some View {
        Menu {
            Toggle("Private chat", isOn: Binding(get: { model.isPrivate }, set: { model.setPrivate($0) }))
            Button("New private chat") { model.newChat(isPrivate: true) }
        } label: {
            Image(systemName: "plus")
                .font(MFFont.calloutEmphasis)
                .foregroundStyle(MFColor.secondary)
                .frame(width: ChatMetrics.controlHeight, height: ChatMetrics.controlHeight)
                .background(MFColor.fill, in: Circle())
        }
        .menuStyle(.button)
        .buttonStyle(.plain)
        .menuIndicator(.hidden)
        .fixedSize()
        .accessibilityLabel("More options")
    }

    private var webChip: some View {
        let blocked = model.webBlockedReason
        let isOn = model.webEnabled && blocked == nil
        return Button {
            model.setWebEnabled(!model.webEnabled)
        } label: {
            HStack(spacing: MFSpacing.xs + 1) {
                Image(systemName: blocked == nil ? "globe" : "wifi.slash").imageScale(.small)
                Text(blocked ?? (isOn ? "Web on" : "Web off"))
            }
            .font(MFFont.chip)
            .foregroundStyle(isOn ? MFColor.accent : (blocked == nil ? MFColor.secondary : MFColor.tertiary))
            .padding(.horizontal, MFSpacing.s + 1)
            .frame(height: ChatMetrics.controlHeight)
            .background(isOn ? MFColor.accentFill : MFColor.fill, in: Capsule())
            .contentShape(Capsule())
        }
        .buttonStyle(.plain)
        .disabled(blocked != nil)
        .accessibilityLabel("Web search")
        .accessibilityValue(blocked ?? (isOn ? "On" : "Off"))
        .accessibilityHint(blocked == nil ? "Searches go through SearXNG on this Mac; pages are fetched from this Mac"
                                          : "Turn off offline mode in Settings to search the web")
    }

    private var modelButton: some View {
        Button {
            pickerOpen.toggle()
        } label: {
            HStack(spacing: MFSpacing.xs) {
                Text(model.model.shortName)
                Image(systemName: "chevron.down").imageScale(.small)
            }
            .font(MFFont.chip)
            .foregroundStyle(MFColor.ink)
            .padding(.horizontal, MFSpacing.s + MFSpacing.xxs)
            .frame(height: ChatMetrics.controlHeight)
            .background(MFColor.fill.opacity(0.7), in: Capsule())
            .contentShape(Capsule())
        }
        .buttonStyle(.plain)
        .accessibilityLabel("Model, \(model.model.shortName)")
        .accessibilityHint("Choose the model for this chat")
    }

    @ViewBuilder
    private var sendButton: some View {
        if model.isGenerating {
            Button(action: model.stop) {
                Image(systemName: "stop.fill")
                    .font(MFFont.calloutEmphasis)
                    .foregroundStyle(MFColor.onInk)
                    .frame(width: ChatMetrics.sendSize, height: ChatMetrics.sendSize)
                    .background(MFColor.ink, in: Circle())
            }
            .buttonStyle(.plain)
            .keyboardShortcut(".", modifiers: .command)
            .accessibilityLabel("Stop")
            .accessibilityHint("Command period")
        } else {
            Button(action: model.send) {
                Image(systemName: "arrow.up")
                    .font(MFFont.bodyEmphasis)
                    .foregroundStyle(MFColor.onInk)
                    .frame(width: ChatMetrics.sendSize, height: ChatMetrics.sendSize)
                    .background(MFColor.ink.opacity(model.canSend ? 1 : 0.35), in: Circle())
            }
            .buttonStyle(.plain)
            .disabled(!model.canSend)
            .accessibilityLabel("Send")
        }
    }
}

/// Starters under the composer (mockup 02).
struct SuggestionChips: View {
    @Bindable var model: ChatViewModel
    let actions: ChatActions

    var body: some View {
        HStack(spacing: MFSpacing.xs + MFSpacing.xxs) {
            chip("Answer from the web") {
                model.setWebEnabled(true)
            }
            chip("Draft a reply") {
                model.draft = "Draft a reply to this message:\n"
            }
            chip("Plan my week") {
                model.draft = "Help me plan my week. Here is what I have on: "
            }
            chip("Scan a receipt") {
                actions.openSection(.scan)
            }
        }
    }

    private func chip(_ title: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Text(title)
                .font(MFFont.chip)
                .foregroundStyle(MFColor.secondary)
                .padding(.horizontal, MFSpacing.s + MFSpacing.xxs)
                .frame(height: ChatMetrics.controlHeight)
                .background(MFColor.surface.opacity(0.7), in: Capsule())
                .overlay(Capsule().strokeBorder(MFColor.hairline))
                .contentShape(Capsule())
        }
        .buttonStyle(.plain)
    }
}

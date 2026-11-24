import MferenceAppCore
import MferenceDesign
import SwiftUI
import UniformTypeIdentifiers

/// The message box: attached files, text, + (attach), Web on/off, model
/// button, send / stop.
struct Composer: View {
    @Bindable var model: ChatViewModel
    @Binding var pickerOpen: Bool
    @FocusState private var focused: Bool
    /// The open panel being shown: files or images.
    @State private var importing: ChatAttachment.Kind?
    @State private var dropTargeted = false

    var body: some View {
        VStack(alignment: .leading, spacing: MFSpacing.s) {
            if !model.pendingAttachments.isEmpty {
                attachmentRow
            }
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
                if model.offersFiles { filesChip }
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
        .overlay(RoundedRectangle(cornerRadius: MFRadius.xl)
            .strokeBorder(dropTargeted ? MFColor.accent : MFColor.hairlineStrong))
        .dropDestination(for: URL.self) { urls, _ in
            let files = urls.filter(\.isFileURL)
            guard !files.isEmpty else { return false }
            model.attach(files)
            return true
        } isTargeted: { dropTargeted = $0 }
        .fileImporter(isPresented: Binding(get: { importing != nil }, set: { if !$0 { importing = nil } }),
                      allowedContentTypes: Self.contentTypes(for: importing ?? .document),
                      allowsMultipleSelection: true) { result in
            if case .success(let urls) = result { model.attach(urls) }
        }
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

    /// "+": attach files or images to the next message.
    private var moreMenu: some View {
        Menu {
            Button {
                importing = .document
            } label: {
                Label("Add files…", systemImage: "doc")
            }
            Button {
                importing = .image
            } label: {
                Label("Add images…", systemImage: "photo")
            }
            Section {
                Text("PDFs, documents and text are read on this Mac. Images are read for the text in them.")
            }
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
        .disabled(!model.canAttach)
        .help("Attach files or images")
        .accessibilityLabel("Attach files")
    }

    /// The files waiting to be sent, one chip each.
    private var attachmentRow: some View {
        ScrollView(.horizontal) {
            HStack(spacing: MFSpacing.xs + MFSpacing.xxs) {
                ForEach(model.pendingAttachments) { pending in
                    AttachmentChip(pending: pending) { model.removeAttachment(pending.id) }
                }
            }
        }
        .scrollIndicators(.never)
        .accessibilityElement(children: .contain)
        .accessibilityLabel("Attached files")
    }

    /// The open panel's file types for files or images.
    static func contentTypes(for kind: ChatAttachment.Kind) -> [UTType] {
        let extensions = kind == .image ? ChatAttachmentReader.imageExtensions
                                        : ChatAttachmentReader.documentExtensions
        var seen = Set<UTType>()
        return extensions.compactMap { UTType(filenameExtension: $0) }.filter { seen.insert($0).inserted }
    }

    private var webChip: some View {
        let blocked = model.webBlockedReason
        let isOn = model.webEnabled && blocked == nil
        return Button {
            model.setWebEnabled(!model.webEnabled)
        } label: {
            HStack(spacing: MFSpacing.xs + 1) {
                Image(systemName: blocked == nil || model.webBlockedByModel ? "globe" : "wifi.slash").imageScale(.small)
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
                           : model.webBlockedByModel ? "Apple’s model answers without the web. Choose Quick or Deep to search"
                           : "Turn off offline mode in Settings to search the web")
    }

    /// Files: answer from this Mac's files with Find (per chat, like Web).
    private var filesChip: some View {
        let blocked = model.filesBlockedReason
        let isOn = model.filesEnabled && blocked == nil
        return Button {
            model.setFilesEnabled(!model.filesEnabled)
        } label: {
            HStack(spacing: MFSpacing.xs + 1) {
                Image(systemName: "folder").imageScale(.small)
                Text(blocked ?? (isOn ? "Files on" : "Files off"))
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
        .accessibilityLabel("Answer from your files")
        .accessibilityValue(blocked ?? (isOn ? "On" : "Off"))
        .accessibilityHint(blocked == nil ? "Searches the files on this Mac with Find; nothing leaves this Mac"
                           : "Turn on Find to answer from your files")
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

import MferenceAppCore
import MferenceDesign
import SwiftUI

/// Compose (C), docked at the bottom left of Mail. Not modal: the list and
/// the reader stay usable while it is open, and it folds down to a bar.
/// Quick can write the body from a prompt, rewrite it, fix its grammar and
/// suggest a subject; nothing is sent until Send.
struct MailComposeCard: View {
    @Bindable var composer: MailComposer
    @Environment(\.mfRendersStatically) private var staticRendering
    /// The prompt row; it always shows while the body is empty.
    @State private var promptOpen = false
    @State private var confirmsDiscard = false
    @FocusState private var focus: Field?

    enum Field: Hashable { case to, subject, body, prompt }

    var body: some View {
        Group {
            if composer.isMinimized { minimizedBar } else { expanded }
        }
        .background(RoundedRectangle(cornerRadius: MFRadius.l).fill(MFColor.surface))
        .clipShape(RoundedRectangle(cornerRadius: MFRadius.l))
        .overlay(RoundedRectangle(cornerRadius: MFRadius.l).strokeBorder(MFColor.hairlineStrong, lineWidth: MFLayout.hairline))
        .shadow(color: MFColor.ink.opacity(MailMetrics.composeShadowOpacity), radius: MailMetrics.composeShadowRadius,
                y: MailMetrics.composeShadowY)
        .confirmationDialog("Discard this message?", isPresented: $confirmsDiscard) {
            Button("Discard", role: .destructive) { composer.discard() }
        } message: {
            Text("The draft is not saved anywhere.")
        }
        .accessibilityElement(children: .contain)
        .accessibilityLabel("New message")
    }

    private var title: String {
        let subject = composer.subject.trimmingCharacters(in: .whitespacesAndNewlines)
        return subject.isEmpty ? "New message" : subject
    }

    // MARK: Minimized

    private var minimizedBar: some View {
        HStack(spacing: MFSpacing.s) {
            Button { composer.restore() } label: {
                HStack(spacing: MFSpacing.s) {
                    Image(systemName: "square.and.pencil").foregroundStyle(MFColor.accent)
                    Text(title)
                        .font(MFFont.rowTitle)
                        .foregroundStyle(MFColor.ink)
                        .lineLimit(1)
                    Spacer(minLength: MFSpacing.s)
                    Image(systemName: "chevron.up").foregroundStyle(MFColor.secondary)
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .accessibilityLabel("Show the message \(title)")
            closeButton
        }
        .padding(.horizontal, MFSpacing.m)
        .padding(.vertical, MFSpacing.s + MFSpacing.xxs)
        .frame(width: MailMetrics.composeMinimizedWidth)
    }

    // MARK: Expanded

    private var expanded: some View {
        VStack(spacing: 0) {
            header
            hairline
            addressRow("To", text: $composer.to, field: .to)
            hairline
            subjectRow
            hairline
            bodyEditor
            hairline
            assist
            hairline
            footer
        }
        .frame(width: MailMetrics.composeWidth)
        .frame(maxHeight: MailMetrics.composeMaxHeight)
        .onAppear { focus = composer.to.isEmpty ? .to : .body }
    }

    private var header: some View {
        HStack(spacing: MFSpacing.s) {
            Text(title)
                .font(MFFont.rowTitle)
                .foregroundStyle(MFColor.ink)
                .lineLimit(1)
                .accessibilityAddTraits(.isHeader)
            Spacer(minLength: MFSpacing.s)
            Button { composer.minimize() } label: { Image(systemName: "minus") }
                .buttonStyle(.appleQuiet)
                .help("Minimize")
                .accessibilityLabel("Minimize the message")
            closeButton
        }
        .padding(.leading, MFSpacing.l)
        .padding(.trailing, MFSpacing.s)
        .padding(.vertical, MFSpacing.xs)
        .background(MFColor.canvas)
    }

    private var closeButton: some View {
        Button { close() } label: { Image(systemName: "xmark") }
            .buttonStyle(.appleQuiet)
            .disabled(!composer.canDiscard)
            .help("Close")
            .accessibilityLabel("Close the message")
    }

    private func close() {
        if composer.hasContent { confirmsDiscard = true } else { composer.discard() }
    }

    private func addressRow(_ label: String, text: Binding<String>, field: Field) -> some View {
        HStack(spacing: MFSpacing.s) {
            fieldLabel(label)
            AppleTextField(title: "name@example.com", text: text)
                .font(MFFont.body)
                .focused($focus, equals: field)
                .disabled(composer.isLocked)
                .accessibilityLabel(label)
        }
        .padding(.horizontal, MFSpacing.l)
        .padding(.vertical, MFSpacing.s)
    }

    private var subjectRow: some View {
        HStack(spacing: MFSpacing.s) {
            fieldLabel("Subject")
            AppleTextField(title: "", text: $composer.subject)
                .font(MFFont.body)
                .focused($focus, equals: .subject)
                .disabled(composer.isLocked)
                .accessibilityLabel("Subject")
            Button { composer.run(.suggestSubject) } label: {
                Label("Suggest", systemImage: "sparkles")
            }
            .buttonStyle(.appleQuiet)
            .disabled(!composer.canRewrite)
            .help("Quick suggests a subject from the message")
        }
        .padding(.leading, MFSpacing.l)
        .padding(.trailing, MFSpacing.s)
        .padding(.vertical, MFSpacing.xs)
    }

    private func fieldLabel(_ text: String) -> some View {
        Text(text)
            .font(MFFont.caption)
            .foregroundStyle(MFColor.tertiary)
            .frame(width: MailMetrics.composeLabelWidth, alignment: .leading)
            .accessibilityHidden(true)
    }

    @ViewBuilder
    private var bodyEditor: some View {
        Group {
            if staticRendering {
                Text(composer.body.isEmpty ? Self.bodyPlaceholder : composer.body)
                    .foregroundStyle(composer.body.isEmpty ? MFColor.tertiary : MFColor.ink)
                    .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
            } else {
                TextEditor(text: $composer.body)
                    .scrollContentBackground(.hidden)
                    .focused($focus, equals: .body)
                    .disabled(composer.isLocked)
                    .overlay(alignment: .topLeading) {
                        if composer.body.isEmpty {
                            Text(Self.bodyPlaceholder)
                                .foregroundStyle(MFColor.tertiary)
                                .padding(.leading, MFSpacing.xs + MFSpacing.xxs)
                                .allowsHitTesting(false)
                                .accessibilityHidden(true)
                        }
                    }
                    .accessibilityLabel("Message")
            }
        }
        .font(MFFont.body)
        .lineSpacing(MFLineSpacing.normal)
        .frame(minHeight: MailMetrics.composeBodyMinHeight, maxHeight: .infinity)
        .padding(.horizontal, MFSpacing.m)
        .padding(.vertical, MFSpacing.s)
    }

    static let bodyPlaceholder = "Write your message, or ask Quick below."

    // MARK: Quick

    @ViewBuilder
    private var assist: some View {
        VStack(alignment: .leading, spacing: MFSpacing.s) {
            switch composer.assistState {
            case .writing:
                HStack(spacing: MFSpacing.s) {
                    ProgressView().controlSize(.small)
                    Text("Quick is writing on this Mac…")
                        .font(MFFont.callout)
                        .foregroundStyle(MFColor.secondary)
                    Spacer()
                    Button("Stop") { composer.cancelAssist() }
                        .buttonStyle(.appleSecondary)
                }
            case .failed(let message):
                HStack(alignment: .firstTextBaseline, spacing: MFSpacing.s) {
                    Text(message)
                        .font(MFFont.callout)
                        .foregroundStyle(MFColor.danger)
                        .fixedSize(horizontal: false, vertical: true)
                    Spacer()
                    Button("OK") { composer.dismissAssistError() }
                        .buttonStyle(.appleQuiet)
                }
            case .idle:
                EmptyView()
            }
            if showsPrompt { promptRow }
            tools
        }
        .padding(.horizontal, MFSpacing.l)
        .padding(.vertical, MFSpacing.s)
    }

    private var showsPrompt: Bool {
        promptOpen || composer.body.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    private var promptRow: some View {
        HStack(spacing: MFSpacing.s) {
            Image(systemName: "sparkles")
                .foregroundStyle(MFColor.accent)
                .accessibilityHidden(true)
            AppleTextField(title: composer.body.isEmpty ? "Describe the email for Quick…"
                                                        : "Tell Quick how to turn your notes into an email…",
                           text: $composer.prompt)
                .font(MFFont.callout)
                .focused($focus, equals: .prompt)
                .disabled(composer.isLocked)
                .onSubmit { composer.run(.draft) }
                .accessibilityLabel("What Quick should write")
            Button("Write") { composer.run(.draft) }
                .buttonStyle(.appleSecondary)
                .disabled(!composer.canDraft)
        }
        .padding(.horizontal, MFSpacing.s)
        .padding(.vertical, MFSpacing.xs)
        .background(RoundedRectangle(cornerRadius: MFRadius.s).fill(MFColor.accentFillFaint))
    }

    private var tools: some View {
        HStack(spacing: MFSpacing.xs) {
            FlowLayout(spacing: MFSpacing.xs) {
                if !showsPrompt {
                    Button { promptOpen = true; focus = .prompt } label: {
                        Label("Write with Quick", systemImage: "sparkles")
                    }
                    .buttonStyle(ChipButtonStyle())
                    .disabled(composer.isLocked)
                }
                Button("Shorter") { composer.run(.rewrite(.shorter)) }
                    .buttonStyle(ChipButtonStyle()).disabled(!composer.canRewrite)
                Button("Warmer") { composer.run(.rewrite(.warmer)) }
                    .buttonStyle(ChipButtonStyle()).disabled(!composer.canRewrite)
                Button("More formal") { composer.run(.rewrite(.formal)) }
                    .buttonStyle(ChipButtonStyle()).disabled(!composer.canRewrite)
                Button("Fix grammar") { composer.run(.fixGrammar) }
                    .buttonStyle(ChipButtonStyle()).disabled(!composer.canRewrite)
                if composer.canUndo {
                    Button { composer.undoAssist() } label: {
                        Label("Undo", systemImage: "arrow.uturn.backward")
                    }
                    .buttonStyle(ChipButtonStyle())
                }
            }
            Spacer(minLength: 0)
            if let label = composer.modelLabel {
                Text(label)
                    .font(MFFont.caption)
                    .foregroundStyle(MFColor.tertiary)
            }
        }
    }

    // MARK: Footer

    private var footer: some View {
        HStack(spacing: MFSpacing.m) {
            Text(status)
                .font(MFFont.callout)
                .foregroundStyle(sendFailed ? MFColor.danger : MFColor.secondary)
                .lineLimit(2)
                .fixedSize(horizontal: false, vertical: true)
            Spacer(minLength: MFSpacing.s)
            Button("Discard") { close() }
                .buttonStyle(.appleSecondary)
                .disabled(!composer.canDiscard)
            Button(composer.sendState == .sending ? "Sending…" : "Send") {
                Task { await composer.send() }
            }
            .buttonStyle(.applePrimary)
            .keyboardShortcut(.return, modifiers: .command)
            .disabled(!composer.canSend)
        }
        .padding(.horizontal, MFSpacing.l)
        .padding(.vertical, MFSpacing.s)
    }

    private var sendFailed: Bool {
        if case .failed = composer.sendState { return true }
        return false
    }

    private var status: String {
        switch composer.sendState {
        case .sending: "Sending through Mail…"
        case .failed(let message): message
        case .idle: "Not sent until you press Send."
        }
    }

    private var hairline: some View {
        Rectangle().fill(MFColor.hairline).frame(height: MFLayout.hairline)
    }
}

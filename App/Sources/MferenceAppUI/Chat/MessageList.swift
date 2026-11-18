import MferenceAppCore
import MferenceDesign
import SwiftUI

/// The scrolling conversation: user bubbles, answers, the answer being
/// written, and the error line.
struct MessageList: View {
    @Bindable var model: ChatViewModel
    let stepsExpanded: Bool

    private static let bottomID = "chat-bottom"

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                VStack(alignment: .leading, spacing: MFSpacing.l + MFSpacing.xxs) {
                    if model.messages.isEmpty, model.visibleStreaming == nil {
                        EmptyChat(model: model)
                    }
                    ForEach(model.messages) { message in
                        MessageView(message: message,
                                    isLastAnswer: message.id == lastAnswerID && !model.isGenerating,
                                    stepsExpanded: stepsExpanded,
                                    onRegenerate: { model.regenerate() })
                    }
                    if let streaming = model.visibleStreaming {
                        MessageView(message: streaming, isLastAnswer: false, stepsExpanded: false, onRegenerate: {})
                    }
                    if let error = model.errorMessage {
                        ErrorLine(message: error, retry: model.canRetryAfterError ? { model.regenerate() } : nil)
                    }
                    Color.clear.frame(height: ChatMetrics.hairlineWidth).id(Self.bottomID)
                }
                .frame(maxWidth: ChatMetrics.columnWidth, alignment: .leading)
                .padding(.horizontal, MFSpacing.xl)
                .padding(.vertical, MFSpacing.xl)
                .frame(maxWidth: .infinity)
            }
            .onChange(of: model.messages.count) { proxy.scrollTo(Self.bottomID, anchor: .bottom) }
            .onChange(of: model.visibleStreaming?.content.count) { proxy.scrollTo(Self.bottomID, anchor: .bottom) }
        }
    }

    private var lastAnswerID: UUID? {
        model.messages.last(where: { $0.role == .assistant })?.id
    }
}

struct EmptyChat: View {
    let model: ChatViewModel

    var body: some View {
        VStack(spacing: MFSpacing.s) {
            Text("What can I help with?")
                .font(MFFont.title)
                .foregroundStyle(MFColor.ink)
            Text(model.webEnabled
                 ? "\(model.model.shortName) answers on this Mac. Web search is on for this chat."
                 : "\(model.model.shortName) answers on this Mac. Nothing you type leaves it.")
                .font(MFFont.callout)
                .foregroundStyle(MFColor.secondary)
        }
        .frame(maxWidth: .infinity)
        .padding(.top, MFSpacing.xxl * 3)
    }
}

struct ErrorLine: View {
    let message: String
    /// "Try again", only when the error came from writing an answer.
    let retry: (() -> Void)?

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: MFSpacing.s) {
            Image(systemName: "exclamationmark.circle")
                .accessibilityHidden(true)
            Text(message)
                .fixedSize(horizontal: false, vertical: true)
            if let retry {
                Button("Try again", action: retry)
                    .buttonStyle(.plain)
                    .font(MFFont.calloutEmphasis)
                    .foregroundStyle(MFColor.accent)
            }
        }
        .font(MFFont.callout)
        .foregroundStyle(MFColor.danger)
        .accessibilityElement(children: .combine)
    }
}

/// One user bubble or one answer.
struct MessageView: View {
    let message: ChatMessageItem
    let isLastAnswer: Bool
    let stepsExpanded: Bool
    let onRegenerate: () -> Void

    var body: some View {
        switch message.role {
        case .user: userBubble
        case .assistant: answer
        }
    }

    private var userBubble: some View {
        HStack {
            Spacer(minLength: MFSpacing.xxl)
            Text(message.content)
                .font(MFFont.body)
                .lineSpacing(MFSpacing.xs)
                .foregroundStyle(MFColor.ink)
                .textSelection(.enabled)
                .padding(.horizontal, MFSpacing.m + MFSpacing.xxs)
                .padding(.vertical, MFSpacing.s + MFSpacing.xxs)
                .background(MFColor.fill, in: UnevenRoundedRectangle(
                    topLeadingRadius: MFRadius.l + MFSpacing.xxs, bottomLeadingRadius: MFRadius.l + MFSpacing.xxs,
                    bottomTrailingRadius: MFRadius.xs - 1, topTrailingRadius: MFRadius.l + MFSpacing.xxs))
                .frame(maxWidth: ChatMetrics.bubbleWidth, alignment: .trailing)
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel("You: \(message.content)")
    }

    private var answer: some View {
        VStack(alignment: .leading, spacing: MFSpacing.m) {
            if let title = message.stepsTitle ?? (message.isStreaming && message.content.isEmpty ? "Thinking…" : nil) {
                StepsDisclosure(title: title, steps: message.steps, reasoning: message.reasoning,
                                isExpanded: stepsExpanded)
            }
            if !message.content.isEmpty {
                MarkdownView(text: message.content, citations: message.citations)
            }
            if !message.citations.isEmpty, !message.isStreaming {
                SourcesRow(citations: message.citations)
            }
            footer
        }
        .accessibilityElement(children: .contain)
    }

    @ViewBuilder
    private var footer: some View {
        HStack(spacing: MFSpacing.s) {
            if message.isStreaming {
                ProgressView().controlSize(.small)
                    .accessibilityLabel("Writing")
            } else {
                if let note = finishNote {
                    Text(note)
                        .font(MFFont.caption)
                        .foregroundStyle(MFColor.preview)
                }
                QuietButton(title: "Copy", systemImage: "doc.on.doc") {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(message.content, forType: .string)
                }
                if isLastAnswer {
                    QuietButton(title: "Regenerate", systemImage: "arrow.clockwise", action: onRegenerate)
                        .keyboardShortcut("r", modifiers: .command)
                }
            }
            if let model = message.model {
                Text("\(model.shortName) · on this Mac")
                    .font(MFFont.caption)
                    .foregroundStyle(MFColor.tertiary)
                    .padding(.leading, MFSpacing.xs)
            }
        }
    }

    private var finishNote: String? {
        switch message.finish {
        case .cancelled: "Stopped"
        case .length: "Cut off at the length limit"
        case .failed: "Did not finish"
        default: nil
        }
    }
}

/// An outlined capsule button under an answer.
struct QuietButton: View {
    let title: String
    let systemImage: String
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Label(title, systemImage: systemImage)
                .font(MFFont.calloutEmphasis)
                .foregroundStyle(MFColor.ink)
                .padding(.horizontal, MFSpacing.m)
                .frame(height: ChatMetrics.controlHeight)
                .background(MFColor.surface, in: Capsule())
                .overlay(Capsule().strokeBorder(MFColor.hairlineStrong))
                .contentShape(Capsule())
        }
        .buttonStyle(.plain)
    }
}

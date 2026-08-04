import AppKit
import ChatTemplateCore
import ChatTemplateUI
import MferenceAppCore
import MferenceMacPresentation
import SwiftUI

struct OutputPaneView: View {
    let model: AppModel
    @State private var responseCopyFeedbackID: UUID?

    var body: some View {
        Group {
            if model.hasOutputTranscript {
                transcript
            } else {
                placeholder
            }
        }
        .task(id: responseCopyFeedbackID) {
            guard let feedbackID = responseCopyFeedbackID else { return }
            try? await Task.sleep(for: .seconds(1.2))
            guard !Task.isCancelled, responseCopyFeedbackID == feedbackID else { return }
            withAnimation(.easeOut(duration: 0.15)) {
                responseCopyFeedbackID = nil
            }
        }
        .contextMenu {
            Button("Copy response") {
                copyResponse()
            }
            .disabled(model.outputResponsePlainText.isEmpty)

            Button("Copy prompt") {
                copy(model.outputPromptText)
            }
            .disabled(model.outputPromptText.isEmpty)

            Button("Copy conversation") {
                copy(model.outputConversationPlainText)
            }
            .disabled(model.outputConversationPlainText.isEmpty)

            Divider()

            Button("Clear chat history") { model.clearOutput() }
                .disabled(model.isRunning || !model.hasOutputTranscript)
        }
    }

    private var placeholder: some View {
        EmptyConversationLayout(spacing: 8) {
            EmptyPlaceholderIcon(systemName: placeholderSymbol)
                .frame(width: 32, height: 32)

            emptyPlaceholderContent
        }
        .padding(.horizontal, 24)
        .padding(.vertical, 20)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var transcript: some View {
        TemplateTranscriptView(model: model)
            .id(model.selectedChatID)
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .overlay(alignment: .topTrailing) {
                if !model.isRunning && !model.outputResponsePlainText.isEmpty {
                    copyResponseButton
                        .padding(8)
                }
            }
            .padding(.horizontal, 24)
            .padding(.top, 20)
    }

    private var copyResponseButton: some View {
        Button {
            copyResponse()
        } label: {
            Image(systemName: responseCopyFeedbackID == nil
                  ? "doc.on.doc"
                  : "checkmark.circle.fill")
                .font(.callout.weight(.medium))
                .contentTransition(.symbolEffect(.replace))
                .foregroundStyle(responseCopyFeedbackID == nil
                                 ? Color.secondary
                                 : MferenceMacTheme.accentColor)
                .frame(width: 28, height: 28)
                .contentShape(Circle())
                .background(.regularMaterial, in: Circle())
                .overlay {
                    Circle().stroke(.separator.opacity(0.5), lineWidth: 0.5)
                }
        }
        .buttonStyle(.plain)
        .accessibilityLabel(responseCopyFeedbackID == nil
                            ? "Copy response"
                            : "Response copied")
        .accessibilityHint("Copies only the generated answer")
        .help(responseCopyFeedbackID == nil
              ? "Copy response"
              : "Response copied")
    }

    private var emptyPlaceholderContent: some View {
        VStack(spacing: 8) {
            if !needsModelLoad {
                Text("Chat")
                    .font(.title2.weight(.semibold))
                Text("Send a message to get started")
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
            }
            if isLoadingModel {
                LoadingModelText()
                    .font(.callout)
                    .foregroundStyle(.tertiary)
            } else if let placeholderHint {
                Text(placeholderHint)
                    .font(.callout)
                    .foregroundStyle(.tertiary)
            }
            if let detail = model.presentation.detail {
                Text(detail)
                    .font(.caption)
                    .foregroundStyle(model.presentation.severity == .error ? .red : .secondary)
                    .multilineTextAlignment(.center)
            }
            if model.canLoadModel {
                Button(model.loadState.isFailed ? "Retry Load" : "Load Model",
                       action: model.loadModel)
                    .buttonStyle(.borderedProminent)
                    .controlSize(.large)
            } else if isLoadingModel {
                Button("Load Model", action: {})
                    .buttonStyle(.borderedProminent)
                    .controlSize(.large)
                    .hidden()
                    .accessibilityHidden(true)
            } else if model.canReloadModel {
                Button("Reload Model", action: model.reloadModel)
                    .buttonStyle(.borderedProminent)
            }
        }
        .frame(maxWidth: .infinity)
    }

    private var needsModelLoad: Bool {
        !model.loadState.isReady
    }

    private var isLoadingModel: Bool {
        if case .loading = model.loadState { return true }
        return false
    }

    private var placeholderSymbol: String {
        "cube.transparent"
    }

    private var placeholderHint: String? {
        if model.loadState.isFailed { return "The model could not be loaded" }
        if model.hasStaleLoadedRuntime { return "Reload the model to use changed settings" }
        return needsModelLoad ? "Load the model to begin" : nil
    }

    private func copy(_ text: String) {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
    }

    private func copyResponse() {
        copy(model.outputResponsePlainText)
        withAnimation(.easeIn(duration: 0.15)) {
            responseCopyFeedbackID = UUID()
        }
    }
}

/// Chat-template conversation rendering over AppModel's transcript: user
/// bubbles, assistant markdown blocks, shimmer before the first token, and
/// a scroll-to-bottom button. Streaming text drains the generation mailbox
/// on the same 0.1 s cadence the previous NSTextView transcript used.
private struct TemplateTranscriptView: View {
    let model: AppModel

    @State private var streamedText = ""
    @State private var distanceFromBottom: CGFloat = 0

    /// While running, the mailbox is the live response; afterwards the
    /// committed output text is authoritative (same resolution the
    /// incremental transcript used).
    private var response: String {
        model.isRunning ? streamedText : model.outputText
    }

    private var baseMessages: [ChatMessage] {
        model.transcriptBaseMessages.map { message in
            ChatMessage(
                id: message.id,
                role: message.role == .user ? .user : .assistant,
                content: message.content)
        }
    }

    private var showsLiveResponseRow: Bool {
        model.isRunning || !response.isEmpty
    }

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 20) {
                    ForEach(baseMessages) { message in
                        MessageView(
                            message: message,
                            isStreamingSlot: false,
                            streamingText: "")
                    }
                    if showsLiveResponseRow {
                        MessageView(
                            message: ChatMessage(
                                role: .assistant,
                                content: model.isRunning ? "" : response),
                            isStreamingSlot: model.isRunning,
                            streamingText: response)
                        .id("live-response")
                    }
                    Color.clear
                        .frame(height: 8)
                        .id("bottom")
                }
                .padding(.top, 4)
                .padding(.bottom, 12)
                .frame(maxWidth: 760)
                .frame(maxWidth: .infinity)
            }
            .onScrollGeometryChange(for: CGFloat.self) { geometry in
                geometry.contentSize.height - geometry.visibleRect.maxY
            } action: { _, distance in
                distanceFromBottom = distance
            }
            .onChange(of: streamedText) {
                if distanceFromBottom < 120 {
                    proxy.scrollTo("bottom", anchor: .bottom)
                }
            }
            .onChange(of: baseMessages.count) {
                proxy.scrollTo("bottom", anchor: .bottom)
            }
            .overlay(alignment: .bottom) {
                if distanceFromBottom > 200 {
                    Button {
                        withAnimation { proxy.scrollTo("bottom", anchor: .bottom) }
                    } label: {
                        Image(systemName: "arrow.down")
                            .font(.system(size: 13, weight: .semibold))
                            .padding(9)
                            .background(.regularMaterial, in: Circle())
                            .overlay(Circle().strokeBorder(.separator, lineWidth: 0.5))
                    }
                    .buttonStyle(.plain)
                    .padding(.bottom, 12)
                    .transition(.opacity.combined(with: .scale))
                }
            }
            .onAppear {
                proxy.scrollTo("bottom", anchor: .bottom)
            }
        }
        .task(id: model.isRunning) {
            guard model.isRunning else { return }
            while !Task.isCancelled {
                if let mailbox = model.generationTranscriptMailbox {
                    let text = mailbox.completeText
                    if text != streamedText { streamedText = text }
                }
                try? await Task.sleep(for: .milliseconds(100))
            }
        }
        .accessibilityLabel("Conversation transcript")
    }
}

private struct EmptyPlaceholderIcon: View {
    let systemName: String

    var body: some View {
        Image(systemName: systemName)
            .font(.title2)
            .foregroundStyle(.quaternary)
            .accessibilityHidden(true)
    }
}

private struct EmptyConversationLayout: Layout {
    let spacing: CGFloat

    func sizeThatFits(
        proposal: ProposedViewSize,
        subviews: Subviews,
        cache: inout ()
    ) -> CGSize {
        proposal.replacingUnspecifiedDimensions()
    }

    func placeSubviews(
        in bounds: CGRect,
        proposal: ProposedViewSize,
        subviews: Subviews,
        cache: inout ()
    ) {
        guard subviews.count == 2 else { return }

        let iconSize = subviews[0].sizeThatFits(.unspecified)
        let iconCenter = CGPoint(x: bounds.midX, y: bounds.midY)
        subviews[0].place(
            at: iconCenter,
            anchor: .center,
            proposal: ProposedViewSize(
                width: iconSize.width,
                height: iconSize.height))

        subviews[1].place(
            at: CGPoint(
                x: bounds.midX,
                y: iconCenter.y + iconSize.height / 2 + spacing),
            anchor: .top,
            proposal: ProposedViewSize(width: bounds.width, height: nil))
    }
}

private struct LoadingModelText: View {
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var animationStart = Date()

    var body: some View {
        if reduceMotion {
            label(dotCount: 3)
        } else {
            TimelineView(.periodic(from: .now, by: 0.25)) { context in
                let elapsed = max(0, context.date.timeIntervalSince(animationStart))
                label(dotCount: Int(elapsed / 0.25) % 4)
            }
        }
    }

    private func label(dotCount: Int) -> some View {
        ZStack(alignment: .leading) {
            Text("Loading Model...").hidden()
            Text("Loading Model" + String(repeating: ".", count: dotCount))
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Loading Model")
    }
}

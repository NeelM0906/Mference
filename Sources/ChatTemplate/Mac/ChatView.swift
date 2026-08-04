import ChatTemplateCore
import SwiftUI

/// The conversation pane: scrolling transcript (or empty state), a
/// scroll-to-bottom button, an error banner, and the glass prompt composer.
struct ChatView: View {
    @Bindable var store: ChatStore

    @AppStorage(ChatSettings.useMockKey) private var useMock = true
    @AppStorage(ChatSettings.baseURLKey) private var baseURL = ChatSettings.defaultBaseURL
    @AppStorage(ChatSettings.modelKey) private var selectedModel = ModelCatalog.fallback[0].id

    @State private var input = ""
    @State private var distanceFromBottom: CGFloat = 0

    private var messages: [ChatMessage] {
        store.selectedChat?.messages ?? []
    }

    var body: some View {
        ZStack(alignment: .bottom) {
            if messages.isEmpty {
                emptyState
            } else {
                conversation
            }

            VStack(spacing: 8) {
                if let error = store.errorMessage {
                    errorBanner(error)
                }
                PromptComposer(
                    input: $input,
                    isGenerating: store.isGenerating,
                    onSend: send,
                    onStop: { store.stopGenerating() })
            }
            .padding(.horizontal, 16)
            .padding(.bottom, 12)
        }
        .background(Color(nsColor: .textBackgroundColor))
    }

    private var emptyState: some View {
        VStack(spacing: 6) {
            Text("Chat")
                .font(.title2.weight(.semibold))
            Text("Send a message to get started")
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var conversation: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 20) {
                    ForEach(messages) { message in
                        MessageView(
                            message: message,
                            isStreamingSlot: isStreamingSlot(message),
                            streamingText: store.streamingText)
                    }
                    Color.clear
                        .frame(height: 90)
                        .id("bottom")
                }
                .padding(.horizontal, 20)
                .padding(.top, 16)
                .frame(maxWidth: 760)
                .frame(maxWidth: .infinity)
            }
            .onScrollGeometryChange(for: CGFloat.self) { geometry in
                geometry.contentSize.height - geometry.visibleRect.maxY
            } action: { _, distance in
                distanceFromBottom = distance
            }
            .onChange(of: store.streamingText) {
                if store.generatingChatID == store.selectedChatID, distanceFromBottom < 120 {
                    proxy.scrollTo("bottom", anchor: .bottom)
                }
            }
            .onChange(of: messages.count) {
                withAnimation { proxy.scrollTo("bottom", anchor: .bottom) }
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
                    .padding(.bottom, 96)
                    .transition(.opacity.combined(with: .scale))
                }
            }
        }
    }

    private func errorBanner(_ message: String) -> some View {
        HStack(spacing: 8) {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundStyle(.red)
            Text(message)
                .lineLimit(2)
            Spacer()
            Button("Dismiss") { store.errorMessage = nil }
                .buttonStyle(.plain)
                .foregroundStyle(.secondary)
        }
        .font(.callout)
        .padding(10)
        .background(.red.opacity(0.1), in: RoundedRectangle(cornerRadius: 10))
        .frame(maxWidth: 760)
    }

    private func isStreamingSlot(_ message: ChatMessage) -> Bool {
        store.isGenerating
            && store.generatingChatID == store.selectedChatID
            && message.role == .assistant
            && message.id == messages.last?.id
    }

    private func send() {
        guard let backend = ChatSettings.backend(
            useMock: useMock, baseURL: baseURL, apiKey: KeychainStore.readAPIKey())
        else {
            store.errorMessage = "Invalid server base URL: \(baseURL)"
            return
        }
        let model = useMock ? ModelCatalog.mock.id : selectedModel
        store.send(input, using: backend, model: model)
        input = ""
    }
}

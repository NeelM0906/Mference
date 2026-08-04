import ChatTemplateCore
import SwiftUI

/// One transcript row: user messages as right-aligned bubbles, assistant
/// messages as plain markdown — with a shimmer placeholder before the first
/// streamed token arrives.
struct MessageView: View {
    let message: ChatMessage
    let isStreamingSlot: Bool
    let streamingText: String

    var body: some View {
        switch message.role {
        case .user:
            HStack {
                Spacer(minLength: 60)
                Text(message.content)
                    .textSelection(.enabled)
                    .padding(.horizontal, 14)
                    .padding(.vertical, 9)
                    .background(
                        Color(nsColor: .quaternarySystemFill),
                        in: RoundedRectangle(cornerRadius: 18))
            }
        case .assistant:
            if isStreamingSlot {
                if streamingText.isEmpty {
                    ShimmerPlaceholder()
                } else {
                    MarkdownView(text: streamingText)
                }
            } else {
                MarkdownView(text: message.content)
            }
        }
    }
}

/// Loading shimmer shown while waiting for the first token, mirroring the
/// template's shimmer state.
struct ShimmerPlaceholder: View {
    @State private var phase: CGFloat = -1

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            bar(width: 400)
            bar(width: 310)
            bar(width: 220)
        }
        .onAppear {
            withAnimation(.linear(duration: 1.2).repeatForever(autoreverses: false)) {
                phase = 2
            }
        }
    }

    private func bar(width: CGFloat) -> some View {
        RoundedRectangle(cornerRadius: 6)
            .fill(Color(nsColor: .quaternarySystemFill))
            .overlay(
                LinearGradient(
                    colors: [.clear, .primary.opacity(0.08), .clear],
                    startPoint: .leading,
                    endPoint: .trailing)
                .frame(width: width * 0.6)
                .offset(x: phase * width)
                .clipShape(RoundedRectangle(cornerRadius: 6)))
            .frame(width: width, height: 14)
    }
}

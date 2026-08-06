import SwiftUI

/// Glass prompt composer: plus action, multiline input, and a send button
/// that becomes stop while generating — mirroring the template's Liquid
/// Glass `PromptInput`. Return sends; Option-Return inserts a newline.
struct PromptComposer: View {
    @Binding var input: String
    let isGenerating: Bool
    let onSend: () -> Void
    let onStop: () -> Void

    @FocusState private var focused: Bool

    private var canSend: Bool {
        !input.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    var body: some View {
        HStack(alignment: .bottom, spacing: 10) {
            Menu {
                Button("Attach Photos", systemImage: "photo") {}
                    .disabled(true)
                Button("Attach Files", systemImage: "folder") {}
                    .disabled(true)
            } label: {
                Image(systemName: "plus")
                    .font(.system(size: 15, weight: .medium))
                    .foregroundStyle(.secondary)
                    .frame(width: 28, height: 28)
            }
            .menuStyle(.borderlessButton)
            .menuIndicator(.hidden)
            .fixedSize()
            .help("Attachments (not supported yet)")

            TextField("Message", text: $input, axis: .vertical)
                .textFieldStyle(.plain)
                .font(.body)
                .lineLimit(1...8)
                .focused($focused)
                .onSubmit {
                    if canSend, !isGenerating { onSend() }
                }
                .padding(.vertical, 5)

            Button {
                isGenerating ? onStop() : onSend()
            } label: {
                Image(systemName: isGenerating ? "stop.fill" : "arrow.up")
                    .font(.system(size: 13, weight: .bold))
                    .foregroundStyle(sendForeground)
                    .frame(width: 28, height: 28)
                    .background(sendBackground, in: Circle())
            }
            .buttonStyle(.plain)
            .disabled(!isGenerating && !canSend)
            .help(isGenerating ? "Stop generating (⌘.)" : "Send")
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 8)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 24))
        .overlay(
            RoundedRectangle(cornerRadius: 24)
                .strokeBorder(.separator, lineWidth: 0.5))
        .shadow(color: .black.opacity(0.12), radius: 14, y: 4)
        .frame(maxWidth: 760)
        .onAppear { focused = true }
    }

    private var sendForeground: Color {
        if isGenerating { return .white }
        return canSend ? .white : .secondary
    }

    private var sendBackground: Color {
        if isGenerating { return .red }
        return canSend ? .accentColor : Color(nsColor: .quaternarySystemFill)
    }
}

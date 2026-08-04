import ChatTemplateCore
import SwiftUI

/// Chat history sidebar: starred section plus recency buckets, with
/// star/rename/delete context menus — mirroring the template's chats screen
/// and web sidebar.
struct SidebarView: View {
    @Bindable var store: ChatStore

    @State private var renamingChat: Chat?
    @State private var renameText = ""

    private struct Bucket: Identifiable {
        let id: String
        let chats: [Chat]
    }

    private var buckets: [Bucket] {
        let unstarred = store.chats
            .filter { !$0.starred }
            .sorted { $0.lastActivity > $1.lastActivity }
        let now = Date()
        var grouped: [(String, [Chat])] = []
        for chat in unstarred {
            let days = Int(now.timeIntervalSince(chat.lastActivity) / 86_400)
            let label = switch days {
            case ..<1: "Today"
            case ..<7: "Previous 7 Days"
            case ..<30: "Previous 30 Days"
            default: "Older"
            }
            if grouped.last?.0 == label {
                grouped[grouped.count - 1].1.append(chat)
            } else {
                grouped.append((label, [chat]))
            }
        }
        return grouped.map { Bucket(id: $0.0, chats: $0.1) }
    }

    private var starred: [Chat] {
        store.chats
            .filter(\.starred)
            .sorted { $0.lastActivity > $1.lastActivity }
    }

    var body: some View {
        List(selection: $store.selectedChatID) {
            if !starred.isEmpty {
                Section("Starred") {
                    ForEach(starred) { chat in
                        row(chat)
                    }
                }
            }
            ForEach(buckets) { bucket in
                Section(bucket.id) {
                    ForEach(bucket.chats) { chat in
                        row(chat)
                    }
                }
            }
        }
        .listStyle(.sidebar)
        .navigationTitle("Chats")
        .alert("Rename Chat", isPresented: renameAlertShown) {
            TextField("Title", text: $renameText)
            Button("Rename") {
                if let chat = renamingChat {
                    store.renameChat(id: chat.id, title: renameText)
                }
                renamingChat = nil
            }
            Button("Cancel", role: .cancel) { renamingChat = nil }
        }
    }

    private var renameAlertShown: Binding<Bool> {
        Binding(
            get: { renamingChat != nil },
            set: { if !$0 { renamingChat = nil } })
    }

    private func row(_ chat: Chat) -> some View {
        HStack {
            Text(chat.title)
                .lineLimit(1)
            if chat.starred {
                Spacer()
                Image(systemName: "star.fill")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
        }
        .tag(chat.id)
        .contextMenu {
            Button(chat.starred ? "Unstar" : "Star", systemImage: "star") {
                store.toggleStar(id: chat.id)
            }
            Button("Rename", systemImage: "pencil") {
                renameText = chat.title
                renamingChat = chat
            }
            Divider()
            Button("Delete", systemImage: "trash", role: .destructive) {
                store.deleteChat(id: chat.id)
            }
        }
    }
}

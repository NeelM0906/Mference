import MferenceAppCore
import MferenceMacPresentation
import SwiftUI

/// Chat-template style sidebar: a native source list grouped by recency,
/// with rename/delete context menus. Replaces the custom ChatSidebarView
/// layout while keeping its behavior (selection and edits are blocked
/// while a generation runs; alerts confirm rename/delete).
struct TemplateSidebarView: View {
    let model: AppModel
    @AppStorage(AppAppearance.storageKey)
    private var appearanceRawValue = AppAppearance.system.rawValue

    @State private var chatBeingRenamed: AppChat?
    @State private var renameText = ""
    @State private var chatPendingDeletion: AppChat?

    private struct Bucket: Identifiable {
        let id: String
        let chats: [AppChat]
    }

    var body: some View {
        List(selection: selectionBinding) {
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
        .safeAreaInset(edge: .bottom, spacing: 0) {
            footer
        }
        .alert(
            "Rename chat",
            isPresented: renameAlertPresented,
            presenting: chatBeingRenamed
        ) { chat in
            TextField("Chat name", text: $renameText)
            Button("Cancel", role: .cancel) {}
            Button("Rename") {
                model.renameChat(id: chat.id, title: renameText)
            }
            .disabled(renameText.trimmingCharacters(
                in: .whitespacesAndNewlines).isEmpty)
        } message: { _ in
            Text("Choose a name that identifies this chat's context.")
        }
        .alert(
            "Delete chat?",
            isPresented: deletionAlertPresented,
            presenting: chatPendingDeletion
        ) { chat in
            Button("Cancel", role: .cancel) {}
            Button("Delete", role: .destructive) {
                model.deleteChat(id: chat.id)
            }
        } message: { chat in
            Text("“\(chat.title)” and its saved document context will be removed.")
        }
    }

    private func row(_ chat: AppChat) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack(spacing: 5) {
                Text(chat.title)
                    .lineLimit(1)
                if chat.contextSummary?.isEmpty == false {
                    Image(systemName: "brain")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .help("Older turns are kept in compressed memory")
                }
            }
            if !chat.preview.isEmpty {
                Text(chat.preview)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }
        }
        .tag(chat.id)
        .contextMenu {
            Button("Rename", systemImage: "pencil") {
                renameText = chat.title
                chatBeingRenamed = chat
            }
            .disabled(model.isRunning)
            Divider()
            Button("Delete", systemImage: "trash", role: .destructive) {
                chatPendingDeletion = chat
            }
            .disabled(model.isRunning)
        }
    }

    private var footer: some View {
        HStack(spacing: 7) {
            Image(systemName: "internaldrive")
            Text("\(model.chats.count) local \(model.chats.count == 1 ? "chat" : "chats")")
            Spacer()
            appearanceMenu
        }
        .font(.caption)
        .foregroundStyle(.secondary)
        .padding(.horizontal, 16)
        .frame(height: 38)
        .background(.bar)
    }

    private var appearanceMenu: some View {
        let appearance = AppAppearance.resolve(appearanceRawValue)
        return Menu {
            Picker("Appearance", selection: $appearanceRawValue) {
                ForEach(AppAppearance.allCases) { option in
                    Label(option.label, systemImage: option.systemImage)
                        .tag(option.rawValue)
                }
            }
        } label: {
            Label("Appearance", systemImage: appearance.systemImage)
                .labelStyle(.iconOnly)
                .frame(width: 26, height: 26)
                .contentShape(Circle())
        }
        .menuStyle(.borderlessButton)
        .menuIndicator(.hidden)
        .fixedSize()
        .help("Appearance: \(appearance.label)")
        .accessibilityLabel("Appearance")
        .accessibilityValue(appearance.label)
    }

    private var selectionBinding: Binding<AppChat.ID?> {
        Binding(
            get: { model.selectedChatID },
            set: { id in
                if let id { model.selectChat(id: id) }
            })
    }

    private var buckets: [Bucket] {
        let sorted = model.chats.sorted { lhs, rhs in
            if lhs.updatedAt == rhs.updatedAt {
                return lhs.createdAt > rhs.createdAt
            }
            return lhs.updatedAt > rhs.updatedAt
        }
        let now = Date()
        var grouped: [(String, [AppChat])] = []
        for chat in sorted {
            let days = Int(now.timeIntervalSince(chat.updatedAt) / 86_400)
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

    private var renameAlertPresented: Binding<Bool> {
        Binding(
            get: { chatBeingRenamed != nil },
            set: { if !$0 { chatBeingRenamed = nil } })
    }

    private var deletionAlertPresented: Binding<Bool> {
        Binding(
            get: { chatPendingDeletion != nil },
            set: { if !$0 { chatPendingDeletion = nil } })
    }
}

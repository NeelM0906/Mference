import ChatTemplateCore
import SwiftUI

struct RootView: View {
    @Bindable var store: ChatStore

    var body: some View {
        NavigationSplitView {
            SidebarView(store: store)
                .navigationSplitViewColumnWidth(min: 220, ideal: 260, max: 340)
        } detail: {
            ChatView(store: store)
                .navigationTitle("")
                .toolbar {
                    ToolbarItem(placement: .principal) {
                        ModelPickerMenu()
                    }
                    ToolbarItem(placement: .primaryAction) {
                        Button("New Chat", systemImage: "square.and.pencil") {
                            store.newChat()
                        }
                        .help("New chat (⌘N)")
                    }
                }
        }
    }
}

struct ModelPickerMenu: View {
    @AppStorage(ChatSettings.useMockKey) private var useMock = true
    @AppStorage(ChatSettings.baseURLKey) private var baseURL = ChatSettings.defaultBaseURL
    @AppStorage(ChatSettings.modelKey) private var selectedModel = ModelCatalog.fallback[0].id
    @AppStorage(ChatSettings.extendedThinkingKey) private var extendedThinking = true

    @State private var serverModels: [ChatModelInfo] = []

    private var models: [ChatModelInfo] {
        if useMock { return [ModelCatalog.mock] }
        return serverModels.isEmpty ? ModelCatalog.fallback : serverModels
    }

    private var selected: ChatModelInfo {
        models.first { $0.id == selectedModel } ?? models[0]
    }

    var body: some View {
        Menu {
            Picker("Model", selection: $selectedModel) {
                ForEach(models) { model in
                    if let subtitle = model.subtitle {
                        Text("\(model.label) — \(subtitle)").tag(model.id)
                    } else {
                        Text(model.label).tag(model.id)
                    }
                }
            }
            .pickerStyle(.inline)

            Divider()

            Toggle("Extended Thinking", systemImage: "sparkles", isOn: $extendedThinking)
        } label: {
            HStack(spacing: 4) {
                Text(selected.label)
                    .fontWeight(.semibold)
                Image(systemName: "chevron.down")
                    .font(.caption2.weight(.bold))
                    .foregroundStyle(.secondary)
            }
        }
        .menuStyle(.borderlessButton)
        .fixedSize()
        .task(id: "\(useMock)|\(baseURL)") {
            guard !useMock, let url = URL(string: baseURL) else { return }
            let apiKey = KeychainStore.readAPIKey()
            let backend = OpenAICompatBackend(
                baseURL: url,
                apiKey: apiKey.isEmpty ? nil : apiKey)
            guard let fetched = try? await backend.listModels(), !fetched.isEmpty else { return }
            serverModels = fetched
            if !fetched.contains(where: { $0.id == selectedModel }) {
                selectedModel = fetched[0].id
            }
        }
    }
}

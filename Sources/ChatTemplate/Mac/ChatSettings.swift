import ChatTemplateCore
import Foundation
import SwiftUI

/// @AppStorage keys and backend construction from persisted settings.
/// Mock mode is the default so the app streams out of the box, like the
/// template's `EXPO_PUBLIC_MOCK_AI` path.
enum ChatSettings {
    static let useMockKey = "chat.useMock"
    static let baseURLKey = "chat.baseURL"
    static let modelKey = "chat.model"
    static let extendedThinkingKey = "chat.extendedThinking"

    static let defaultBaseURL = "http://127.0.0.1:8080/v1"

    /// Returns nil when mock mode is off but the base URL cannot be parsed;
    /// callers surface that as an error rather than silently mocking.
    static func backend(useMock: Bool, baseURL: String, apiKey: String) -> (any ChatBackend)? {
        if useMock { return sharedMockBackend }
        guard let url = URL(string: baseURL), url.scheme != nil else { return nil }
        return OpenAICompatBackend(
            baseURL: url,
            apiKey: apiKey.isEmpty ? nil : apiKey)
    }

    /// One instance so the canned responses cycle across sends.
    private static let sharedMockBackend = MockChatBackend()
}

struct SettingsView: View {
    @AppStorage(ChatSettings.useMockKey) private var useMock = true
    @AppStorage(ChatSettings.baseURLKey) private var baseURL = ChatSettings.defaultBaseURL
    @State private var apiKey = KeychainStore.readAPIKey()

    var body: some View {
        Form {
            Toggle("Use mock responses", isOn: $useMock)
            Text("Streams canned answers without a server, like the template's EXPO_PUBLIC_MOCK_AI mode.")
                .font(.caption)
                .foregroundStyle(.secondary)

            Section("OpenAI-compatible server") {
                TextField("Base URL", text: $baseURL, prompt: Text(ChatSettings.defaultBaseURL))
                    .disabled(useMock)
                SecureField("API key (optional)", text: $apiKey)
                    .disabled(useMock)
                    .onChange(of: apiKey) {
                        KeychainStore.writeAPIKey(apiKey)
                    }
                Text("Defaults to the local MferenceServer. Any /v1 chat-completions endpoint works. The key is stored in your login keychain.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .formStyle(.grouped)
        .frame(width: 460)
        .fixedSize()
    }
}

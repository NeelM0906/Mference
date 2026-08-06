import Foundation
import Security

/// The optional server API key lives in the login keychain, not
/// UserDefaults, since the endpoint field accepts non-local servers.
enum KeychainStore {
    private static let service = "ChatTemplateMac"
    private static let account = "openai-compat-api-key"

    static func readAPIKey() -> String {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne,
        ]
        var item: CFTypeRef?
        guard SecItemCopyMatching(query as CFDictionary, &item) == errSecSuccess,
              let data = item as? Data,
              let key = String(data: data, encoding: .utf8)
        else { return "" }
        return key
    }

    static func writeAPIKey(_ key: String) {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
        ]
        SecItemDelete(query as CFDictionary)
        guard !key.isEmpty, let data = key.data(using: .utf8) else { return }
        let attributes = query.merging([
            kSecValueData as String: data,
        ]) { _, new in new }
        SecItemAdd(attributes as CFDictionary, nil)
    }
}

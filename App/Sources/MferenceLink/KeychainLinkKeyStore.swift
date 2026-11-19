import Foundation
import Security

/// The link's keys in the Keychain: one generic-password item for this
/// device's identity key and one per paired device (its identity key, name
/// and pairing secret). Items never sync (no iCloud Keychain) and, on the
/// iPhone, stay on this device and readable after the first unlock so a
/// hand-off result can arrive while the phone is locked.
///
/// "Forget this Mac" / "Forget this iPhone" is `deletePeer(id:)`.
public struct KeychainLinkKeyStore: LinkKeyStore {
    /// The Keychain service all items share ("ai.mference.link").
    public let service: String

    private static let identityAccount = "identity"
    private static let peerPrefix = "peer."

    public init(service: String = "ai.mference.link") {
        self.service = service
    }

    public func loadIdentity() throws -> LinkKeyPair? {
        guard let data = try read(account: Self.identityAccount) else { return nil }
        do {
            return try LinkKeyPair(privateKeyBytes: data)
        } catch {
            throw LinkFailure.keyStore("The saved link key is damaged.")
        }
    }

    public func saveIdentity(_ identity: LinkKeyPair) throws {
        try write(identity.privateKeyBytes, account: Self.identityAccount)
    }

    public func peer(id: String) throws -> LinkPeer? {
        guard let data = try read(account: Self.peerPrefix + id) else { return nil }
        return try decode(data)
    }

    public func peers() throws -> [LinkPeer] {
        var query = baseQuery()
        query[kSecMatchLimit as String] = kSecMatchLimitAll
        query[kSecReturnAttributes as String] = true
        query[kSecReturnData as String] = true
        var result: CFTypeRef?
        let status = SecItemCopyMatching(query as CFDictionary, &result)
        if status == errSecItemNotFound { return [] }
        try check(status)
        let items = result as? [[String: Any]] ?? []
        return try items.compactMap { item -> LinkPeer? in
            guard let account = item[kSecAttrAccount as String] as? String,
                  account.hasPrefix(Self.peerPrefix),
                  let data = item[kSecValueData as String] as? Data
            else { return nil }
            return try decode(data)
        }
        .sorted { $0.pairedAt < $1.pairedAt }
    }

    public func savePeer(_ peer: LinkPeer) throws {
        try write(try JSONEncoder().encode(peer), account: Self.peerPrefix + peer.id)
    }

    public func deletePeer(id: String) throws {
        var query = baseQuery()
        query[kSecAttrAccount as String] = Self.peerPrefix + id
        let status = SecItemDelete(query as CFDictionary)
        if status == errSecItemNotFound { return }
        try check(status)
    }

    // MARK: - Keychain calls

    private func baseQuery() -> [String: Any] {
        [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrSynchronizable as String: false,
        ]
    }

    private func read(account: String) throws -> Data? {
        var query = baseQuery()
        query[kSecAttrAccount as String] = account
        query[kSecMatchLimit as String] = kSecMatchLimitOne
        query[kSecReturnData as String] = true
        var result: CFTypeRef?
        let status = SecItemCopyMatching(query as CFDictionary, &result)
        if status == errSecItemNotFound { return nil }
        try check(status)
        return result as? Data
    }

    private func write(_ data: Data, account: String) throws {
        var query = baseQuery()
        query[kSecAttrAccount as String] = account
        var attributes: [String: Any] = [kSecValueData as String: data]
        #if os(iOS)
        attributes[kSecAttrAccessible as String] = kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly
        #endif
        let status = SecItemUpdate(query as CFDictionary, attributes as CFDictionary)
        if status == errSecItemNotFound {
            query.merge(attributes) { _, new in new }
            try check(SecItemAdd(query as CFDictionary, nil))
        } else {
            try check(status)
        }
    }

    private func decode(_ data: Data) throws -> LinkPeer {
        do {
            return try JSONDecoder().decode(LinkPeer.self, from: data)
        } catch {
            throw LinkFailure.keyStore("A saved paired device is damaged.")
        }
    }

    private func check(_ status: OSStatus) throws {
        guard status == errSecSuccess else {
            let text = SecCopyErrorMessageString(status, nil) as String? ?? "OSStatus \(status)"
            throw LinkFailure.keyStore(text)
        }
    }
}

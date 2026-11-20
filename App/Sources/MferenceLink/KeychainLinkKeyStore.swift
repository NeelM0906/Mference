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

    public func addIdentity(_ identity: LinkKeyPair) throws -> LinkKeyPair {
        var query = baseQuery()
        query[kSecAttrAccount as String] = Self.identityAccount
        query[kSecValueData as String] = identity.privateKeyBytes
        #if os(iOS)
        query[kSecAttrAccessible as String] = kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly
        #endif
        let status = SecItemAdd(query as CFDictionary, nil)
        if status == errSecDuplicateItem, let existing = try loadIdentity() { return existing }
        try check(status)
        return identity
    }

    public func peer(id: String) throws -> LinkPeer? {
        guard let data = try read(account: Self.peerPrefix + id) else { return nil }
        return try decode(data)
    }

    public func peers() throws -> [LinkPeer] {
        // List the accounts first, then read each item: the file-based
        // keychain on macOS does not return data for a match-all query.
        var query = baseQuery()
        query[kSecMatchLimit as String] = kSecMatchLimitAll
        query[kSecReturnAttributes as String] = true
        var result: CFTypeRef?
        let status = SecItemCopyMatching(query as CFDictionary, &result)
        if status == errSecItemNotFound { return [] }
        try check(status)
        let accounts = (result as? [[String: Any]] ?? [])
            .compactMap { $0[kSecAttrAccount as String] as? String }
            .filter { $0.hasPrefix(Self.peerPrefix) }
        return try accounts.compactMap { account -> LinkPeer? in
            guard let data = try read(account: account) else { return nil }
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
        guard status == errSecItemNotFound else { return try check(status) }
        var add = query
        add.merge(attributes) { _, new in new }
        let added = SecItemAdd(add as CFDictionary, nil)
        // Another writer added it between the two calls: update that one.
        if added == errSecDuplicateItem {
            return try check(SecItemUpdate(query as CFDictionary, attributes as CFDictionary))
        }
        try check(added)
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

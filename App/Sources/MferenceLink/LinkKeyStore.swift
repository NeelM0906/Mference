import Foundation

/// Why a link connection ended. Every one of these closes the connection; the
/// link never keeps going after a frame it could not trust.
public enum LinkFailure: Error, Equatable, Sendable {
    /// A frame or message larger than the protocol allows.
    case tooLarge
    /// A frame that does not parse, or a message that does not decode.
    case malformedFrame
    /// A sealed frame whose tag does not check: tampered with, or sealed with
    /// other keys (a peer that is not the paired device).
    case authenticationFailed
    /// A sequence number this side has already received.
    case replayedFrame
    /// A sequence number that skips ahead.
    case outOfOrderFrame
    /// A message this side does not accept at this point of the handshake,
    /// or one sent in the clear that must be sealed.
    case unexpectedMessage
    /// A public key of the wrong length or one that gives no shared secret.
    case badKey
    /// No common protocol version.
    case unsupportedVersion
    /// The other device is not paired with this one.
    case notPaired
    /// Pairing is off on the Mac.
    case pairingOff
    /// The Mac's pairing nonce does not match its commitment.
    case commitmentMismatch
    /// The person (on either device) said the codes do not match.
    case pairingRejected
    /// The other device forgot this one.
    case forgotten
    /// The other device sent a `LinkError` and closed.
    case peer(LinkError)
    /// The key store failed (the Keychain refused a read or a write).
    case keyStore(String)
    /// The connection was closed on this side.
    case closed
}

/// A device this one is paired with.
public struct LinkPeer: Codable, Equatable, Sendable, Identifiable {
    /// Hex of the first 16 bytes of SHA-256 of `identityKey`.
    public var id: String
    public var role: LinkRole
    /// The name the device gave when it paired ("MacBook Pro").
    public var name: String
    /// Its long-term X25519 public key.
    public var identityKey: Data
    /// The 32-byte secret both devices derived when they paired. Secret: it
    /// lives only in the key store.
    public var pairingSecret: Data
    public var pairedAt: Date

    public init(role: LinkRole, name: String, identityKey: Data, pairingSecret: Data, pairedAt: Date) {
        self.id = LinkPeer.id(for: identityKey)
        self.role = role
        self.name = name
        self.identityKey = identityKey
        self.pairingSecret = pairingSecret
        self.pairedAt = pairedAt
    }

    public static func id(for identityKey: Data) -> String {
        LinkKeySchedule.transcriptHash("peer id", [identityKey]).prefix(16).linkHex
    }
}

/// Where a device keeps its link identity and its paired devices. The apps
/// use `KeychainLinkKeyStore`; tests use `InMemoryLinkKeyStore`.
public protocol LinkKeyStore: Sendable {
    /// This device's identity key, or nil before the first one is made.
    func loadIdentity() throws -> LinkKeyPair?
    func saveIdentity(_ identity: LinkKeyPair) throws
    func peer(id: String) throws -> LinkPeer?
    func peers() throws -> [LinkPeer]
    /// Adds the peer or replaces the one with the same id.
    func savePeer(_ peer: LinkPeer) throws
    /// Deletes the peer's keys; nothing happens when it is not there.
    func deletePeer(id: String) throws
}

extension LinkKeyStore {
    /// The identity key, made and saved on first use.
    public func identity(using random: any LinkRandomSource = SystemLinkRandom()) throws -> LinkKeyPair {
        if let existing = try loadIdentity() { return existing }
        let identity = try LinkKeyPair.generate(using: random)
        try saveIdentity(identity)
        return identity
    }
}

/// A key store that lives in memory, for tests and previews.
public final class InMemoryLinkKeyStore: LinkKeyStore, @unchecked Sendable {
    private let lock = NSLock()
    private var identityKey: LinkKeyPair?
    private var stored: [String: LinkPeer] = [:]

    public init(identity: LinkKeyPair? = nil, peers: [LinkPeer] = []) {
        identityKey = identity
        for peer in peers { stored[peer.id] = peer }
    }

    public func loadIdentity() throws -> LinkKeyPair? {
        lock.withLock { identityKey }
    }

    public func saveIdentity(_ identity: LinkKeyPair) throws {
        lock.withLock { identityKey = identity }
    }

    public func peer(id: String) throws -> LinkPeer? {
        lock.withLock { stored[id] }
    }

    public func peers() throws -> [LinkPeer] {
        lock.withLock { stored.values.sorted { $0.pairedAt < $1.pairedAt } }
    }

    public func savePeer(_ peer: LinkPeer) throws {
        lock.withLock { stored[peer.id] = peer }
    }

    public func deletePeer(id: String) throws {
        _ = lock.withLock { stored.removeValue(forKey: id) }
    }
}

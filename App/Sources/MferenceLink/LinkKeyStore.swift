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
    /// Longest name kept for a peer, in characters.
    public static let maxNameLength = 64

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

    /// A name a peer sent, made safe to show: control and invisible format
    /// characters (bidi overrides, zero-width) removed, whitespace trimmed, at
    /// most `maxNameLength` characters, and a plain fallback when nothing is
    /// left. The code comparison, not the name, is what proves the device.
    public static func cleanName(_ name: String, role: LinkRole) -> String {
        let kept = name.unicodeScalars.filter {
            switch $0.properties.generalCategory {
            case .control, .format, .lineSeparator, .paragraphSeparator, .surrogate, .privateUse, .unassigned:
                false
            default:
                true
            }
        }
        let cleaned = String(String.UnicodeScalarView(kept))
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let short = String(cleaned.prefix(maxNameLength))
        return short.isEmpty ? (role == .mac ? "Mac" : "iPhone") : short
    }
}

/// Printing a peer (or an event that carries one) never shows its pairing
/// secret.
extension LinkPeer: CustomStringConvertible, CustomDebugStringConvertible, CustomReflectable {
    public var description: String { "LinkPeer(\(role) \"\(name)\" \(id))" }
    public var debugDescription: String { description }
    public var customMirror: Mirror {
        Mirror(self, children: ["id": id, "role": role, "name": name, "pairedAt": pairedAt])
    }
}

/// Where a device keeps its link identity and its paired devices. The apps
/// use `KeychainLinkKeyStore`; tests use `InMemoryLinkKeyStore`.
public protocol LinkKeyStore: Sendable {
    /// This device's identity key, or nil before the first one is made.
    func loadIdentity() throws -> LinkKeyPair?
    /// Saves `identity` unless one is already saved, and returns the saved
    /// one. Two first launches racing end with one identity, never two.
    func addIdentity(_ identity: LinkKeyPair) throws -> LinkKeyPair
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
        return try addIdentity(try LinkKeyPair.generate(using: random))
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

    public func addIdentity(_ identity: LinkKeyPair) throws -> LinkKeyPair {
        lock.withLock {
            if let identityKey { return identityKey }
            identityKey = identity
            return identity
        }
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

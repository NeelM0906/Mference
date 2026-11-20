import CryptoKit
import Foundation

// Key material and key schedule for the iPhone to Mac link (mobile spec 5.3).
//
// Every device has one long-term X25519 identity key (kept by a
// `LinkKeyStore`, the Keychain in the apps). Every connection adds a fresh
// ephemeral X25519 key on each side. A paired connection's keys come from
// three Diffie-Hellman results (ephemeral-ephemeral and each identity with
// the other side's ephemeral) plus the 32-byte pairing secret both devices
// saved when they paired, so a connection needs both devices' identity keys
// and the pairing secret, and old traffic stays safe if an identity key leaks
// later. Pairing itself is described above `LinkPairRequest`.

/// Where the link gets its random bytes. The apps use `SystemLinkRandom`;
/// tests pass a seeded source so every key and nonce is reproducible.
public protocol LinkRandomSource: Sendable {
    func bytes(_ count: Int) -> Data
}

/// Cryptographically secure random bytes from the system generator.
public struct SystemLinkRandom: LinkRandomSource {
    public init() {}

    public func bytes(_ count: Int) -> Data {
        var generator = SystemRandomNumberGenerator()
        return Data((0..<count).map { _ in UInt8.random(in: .min ... .max, using: &generator) })
    }
}

/// An X25519 key pair. The private half never leaves the process except into
/// a `LinkKeyStore`.
public struct LinkKeyPair: Sendable, Equatable {
    /// Raw 32-byte private key.
    let privateKeyBytes: Data
    /// Raw 32-byte public key, as sent in `LinkHello`.
    public let publicKey: Data

    /// Restores a key pair from its raw private key (what a store saved).
    public init(privateKeyBytes: Data) throws {
        let key = try Curve25519.KeyAgreement.PrivateKey(rawRepresentation: privateKeyBytes)
        self.privateKeyBytes = key.rawRepresentation
        self.publicKey = key.publicKey.rawRepresentation
    }

    public static func generate(using random: any LinkRandomSource) throws -> LinkKeyPair {
        try LinkKeyPair(privateKeyBytes: random.bytes(LinkProtocol.keyBytes))
    }

    /// X25519 with the peer's raw public key. Throws `LinkFailure.badKey` for a
    /// key of the wrong length or one that yields the all-zero secret.
    func sharedSecret(with peerPublicKey: Data) throws -> Data {
        guard peerPublicKey.count == LinkProtocol.keyBytes,
              let peer = try? Curve25519.KeyAgreement.PublicKey(rawRepresentation: peerPublicKey),
              let key = try? Curve25519.KeyAgreement.PrivateKey(rawRepresentation: privateKeyBytes),
              let secret = try? key.sharedSecretFromKeyAgreement(with: peer)
        else { throw LinkFailure.badKey }
        let bytes = secret.withUnsafeBytes { Data($0) }
        guard bytes.contains(where: { $0 != 0 }) else { throw LinkFailure.badKey }
        return bytes
    }

    public static func == (lhs: LinkKeyPair, rhs: LinkKeyPair) -> Bool {
        lhs.publicKey == rhs.publicKey
    }
}

/// The 6-digit code both screens show while pairing.
///
/// Six digits: a device in the middle gets one try per pairing attempt (the
/// commitment stops it from choosing keys that make the codes match), so it
/// succeeds with probability one in a million, and each try needs the person
/// to start pairing and approve a code. Eight digits would add little against
/// that and be harder to compare at a glance; four (1 in 10,000) is too weak.
public struct LinkPairingCode: Sendable, Equatable, CustomStringConvertible {
    public static let digits = 6

    /// The code as a number in `0..<1_000_000`.
    public let value: Int

    init(value: Int) {
        self.value = value
    }

    /// Six digits with leading zeros ("042917").
    public var digitsString: String {
        let raw = String(value)
        return String(repeating: "0", count: Self.digits - raw.count) + raw
    }

    /// What the screens show: two groups of three ("042 917").
    public var description: String {
        let s = digitsString
        return "\(s.prefix(3)) \(s.suffix(3))"
    }
}

/// The keys one connection seals and opens frames with.
struct LinkSessionKeys: Sendable {
    let phoneToMac: SymmetricKey
    let macToPhone: SymmetricKey

    func sending(as role: LinkRole) -> SymmetricKey { role == .phone ? phoneToMac : macToPhone }
    func receiving(as role: LinkRole) -> SymmetricKey { role == .phone ? macToPhone : phoneToMac }
}

/// The derivations, kept in one place so both roles compute them identically.
enum LinkKeySchedule {
    static let label = Data("mference-link/1".utf8)

    /// SHA-256 over a domain label and length-prefixed parts, so no two
    /// different part lists hash the same bytes.
    static func transcriptHash(_ label: String, _ parts: [Data]) -> Data {
        var hasher = SHA256()
        hasher.update(data: Self.label)
        hasher.update(data: Data(label.utf8))
        for part in parts {
            var length = UInt32(part.count).bigEndian
            hasher.update(data: Data(bytes: &length, count: 4))
            hasher.update(data: part)
        }
        return Data(hasher.finalize())
    }

    // MARK: Transcript fields
    //
    // The transcript is built from the decoded field values in a fixed order,
    // never from the JSON bytes they arrived in: whitespace, key order and
    // unknown fields are free for anyone on the path to change, so hashing
    // them would let a device in the middle re-roll the code without
    // touching a single committed value.

    static func field(_ string: String) -> Data { Data(string.utf8) }

    static func field(_ value: Int) -> Data {
        var big = Int64(value).bigEndian
        return Data(bytes: &big, count: 8)
    }

    /// Fixed 8 bytes per entry, so the list needs no inner separators.
    static func field(_ values: [Int]) -> Data { values.reduce(into: Data()) { $0 += field($1) } }

    /// Every field of a hello, in declaration order.
    static func helloFields(_ hello: LinkHello) -> [Data] {
        [field(hello.role.rawValue), field(hello.deviceName), field(hello.protocolVersions),
         field(hello.appVersion), hello.identityKey, hello.ephemeralKey]
    }

    /// What every transcript starts with: each field of the phone's hello
    /// (so its identity and ephemeral keys), then the Mac's, then the version
    /// in use. One part per field, each length-prefixed by `transcriptHash`.
    static func helloParts(phone: LinkHello, mac: LinkHello, version: Int) -> [Data] {
        helloFields(phone) + helloFields(mac) + [field(version)]
    }

    /// The commitment's context: everything sent before the commitment.
    static func commitContext(phone: LinkHello, mac: LinkHello, version: Int, phoneName: String) -> Data {
        transcriptHash("commit context", helloParts(phone: phone, mac: mac, version: version) + [field(phoneName)])
    }

    /// A paired connection's transcript.
    static func sessionTranscript(phone: LinkHello, mac: LinkHello, version: Int) -> Data {
        transcriptHash("session", helloParts(phone: phone, mac: mac, version: version))
    }

    /// Every value the pairing exchanged, in one fixed order on both sides.
    /// The Mac's name and nonce are fixed by the commitment before the phone's
    /// nonce is known, so nothing here is left to choose after it.
    static func pairingTranscript(phone: LinkHello, mac: LinkHello, version: Int, phoneName: String,
                                  macName: String, commitment: Data, phoneNonce: Data, macNonce: Data) -> Data {
        transcriptHash("pairing", helloParts(phone: phone, mac: mac, version: version) + [
            field(phoneName), field(macName), commitment, phoneNonce, macNonce,
        ])
    }

    /// The three Diffie-Hellman results, in the same order on both sides:
    /// ephemeral-ephemeral, phone identity with Mac ephemeral, phone ephemeral
    /// with Mac identity.
    static func handshakeSecrets(
        role: LinkRole, identity: LinkKeyPair, ephemeral: LinkKeyPair,
        peerIdentityKey: Data, peerEphemeralKey: Data
    ) throws -> Data {
        let ee = try ephemeral.sharedSecret(with: peerEphemeralKey)
        switch role {
        case .phone:
            let phoneIdentityMacEphemeral = try identity.sharedSecret(with: peerEphemeralKey)
            let phoneEphemeralMacIdentity = try ephemeral.sharedSecret(with: peerIdentityKey)
            return ee + phoneIdentityMacEphemeral + phoneEphemeralMacIdentity
        case .mac:
            let phoneIdentityMacEphemeral = try ephemeral.sharedSecret(with: peerIdentityKey)
            let phoneEphemeralMacIdentity = try identity.sharedSecret(with: peerEphemeralKey)
            return ee + phoneIdentityMacEphemeral + phoneEphemeralMacIdentity
        }
    }

    static func expand(_ ikm: Data, salt: Data, info: String, count: Int) -> Data {
        let key = HKDF<SHA256>.deriveKey(
            inputKeyMaterial: SymmetricKey(data: ikm), salt: salt,
            info: label + Data(" \(info)".utf8), outputByteCount: count)
        return key.withUnsafeBytes { Data($0) }
    }

    static func sessionKeys(secrets: Data, pairingSecret: Data, transcript: Data) -> LinkSessionKeys {
        let ikm = secrets + pairingSecret
        return LinkSessionKeys(
            phoneToMac: SymmetricKey(data: expand(ikm, salt: transcript, info: "key phone->mac", count: 32)),
            macToPhone: SymmetricKey(data: expand(ikm, salt: transcript, info: "key mac->phone", count: 32)))
    }

    static func pairingSecret(secrets: Data, transcript: Data) -> Data {
        expand(secrets, salt: transcript, info: "pairing secret", count: 32)
    }

    static func pairingCode(secrets: Data, transcript: Data) -> LinkPairingCode {
        let bytes = expand(secrets, salt: transcript, info: "pairing code", count: 8)
        let number = bytes.reduce(UInt64(0)) { $0 << 8 | UInt64($1) }
        return LinkPairingCode(value: Int(number % 1_000_000))
    }

    /// The Mac's commitment: H(Mac ephemeral ‖ phone ephemeral ‖ Mac nonce),
    /// also bound to `context` (the hash of everything sent before it).
    static func commitment(macEphemeral: Data, phoneEphemeral: Data, macNonce: Data, context: Data) -> Data {
        transcriptHash("commit", [macEphemeral, phoneEphemeral, macNonce, context])
    }
}

extension Data {
    /// Lower-case hex.
    var linkHex: String { map { String(format: "%02x", $0) }.joined() }
}

import CryptoKit
import Foundation

/// One end of one link connection, without any I/O: feed it the bytes that
/// arrive (`receive(_:)`), write out every `.send` it returns, and act on the
/// other events. The transport (Network.framework on both apps, an in-memory
/// pipe in tests) owns a `LinkConnection` per socket.
///
/// The connection runs this sequence:
///
/// 1. Both sides send `hello` in the clear (identity key, ephemeral key,
///    versions). The highest common version wins.
/// 2. Already paired: both derive this connection's keys from the hellos and
///    the saved pairing secret. The phone sends its `hello` again, sealed;
///    when it opens on the Mac, the connection is established there and the
///    Mac app must answer with `capabilities` (the Mac's first sealed frame is
///    what confirms the keys on the phone). A phone the Mac does not know gets
///    `LinkError.notPaired`; a device without the right keys fails its first
///    sealed frame (`authenticationFailed`).
/// 3. Pairing (the phone asked to pair, and the Mac has pairing on): the
///    exchange above `LinkPairRequest`, then both show `pairingCode`, the
///    person confirms on each device, both save the other device, and the
///    connection goes on as in 2 with the new keys.
///
/// Any frame it cannot trust closes the connection (`.closed(LinkFailure)`);
/// after that it ignores input.
public struct LinkConnection {
    /// What the phone wants from this connection.
    public enum Intent: Sendable, Equatable {
        /// Use an existing pairing.
        case connect
        /// Pair (again) with this Mac.
        case pair
    }

    public struct Configuration: Sendable {
        public var role: LinkRole
        /// This device's name as the other side shows it.
        public var deviceName: String
        public var appVersion: String
        public var protocolVersions: [Int]
        /// Phone only.
        public var intent: Intent
        /// Mac only: pairing is turned on (Settings › iPhone).
        public var pairingAllowed: Bool

        public init(role: LinkRole, deviceName: String, appVersion: String,
                    protocolVersions: [Int] = [LinkProtocol.version],
                    intent: Intent = .connect, pairingAllowed: Bool = false) {
            self.role = role
            self.deviceName = deviceName
            self.appVersion = appVersion
            self.protocolVersions = protocolVersions
            self.intent = intent
            self.pairingAllowed = pairingAllowed
        }
    }

    public enum Event: Equatable, Sendable {
        /// Bytes to write to the socket, in order.
        case send(Data)
        /// Show this code; call `confirmPairing(accepted:)` with the person's answer.
        case pairingCode(LinkPairingCode, peerName: String)
        /// Both people confirmed; the peer is saved in the key store.
        case paired(LinkPeer)
        /// The connection is authenticated and ready for app messages.
        case established(LinkPeer, version: Int)
        /// An app message from the peer.
        case received(LinkMessage)
        /// The peer forgot this device; its keys are deleted here too.
        case forgotten(LinkPeer)
        /// The connection is over; close the socket.
        case closed(LinkFailure)
    }

    enum State: Equatable {
        case notStarted
        case awaitingHello
        /// Mac: the phone either sends its sealed hello (known phone) or
        /// starts pairing.
        case macAwaitingPhone
        /// Mac, after pairing: waiting for the phone's sealed hello.
        case macAwaitingSealedHello
        /// Phone: sealed hello sent, waiting for the Mac's first sealed frame.
        case phoneAwaitingMac
        case phoneAwaitingPairResponse
        case macAwaitingPhoneNonce
        case phoneAwaitingMacNonce
        case comparingCodes(localAccepted: Bool, peerAccepted: Bool)
        case established
        case closed(LinkFailure)
    }

    public let configuration: Configuration
    private let store: any LinkKeyStore
    private let random: any LinkRandomSource
    private let identity: LinkKeyPair
    private let ephemeral: LinkKeyPair

    private(set) var state: State = .notStarted
    private var reader = LinkFrameReader()
    private var outbound = LinkDirection()
    private var inbound = LinkDirection()
    private var version: Int?

    // Handshake transcript pieces, kept as the decoded values (see
    // `LinkKeySchedule.field`): the JSON bytes they came in are not hashed.
    private var localHello: LinkHello?
    private var peerHello: LinkHello?
    /// The phone's name exactly as sent in `pairRequest`.
    private var pairRequestName = ""
    /// The Mac's name exactly as sent in `pairResponse`.
    private var pairResponseName = ""
    private var commitment = Data()
    private var phoneNonce = Data()
    private var macNonce = Data()
    private var peerName = ""

    /// Mac: keys for a known phone, used once its sealed hello opens with them.
    private var candidateKeys: LinkSessionKeys?
    /// The peer this connection authenticated (or is pairing with).
    public private(set) var peer: LinkPeer?

    public init(configuration: Configuration, store: any LinkKeyStore,
                random: any LinkRandomSource = SystemLinkRandom()) throws {
        self.configuration = configuration
        self.store = store
        self.random = random
        self.identity = try store.identity(using: random)
        self.ephemeral = try LinkKeyPair.generate(using: random)
    }

    private var role: LinkRole { configuration.role }

    /// This device's identity key (what the other side pairs with).
    public var identityKey: Data { identity.publicKey }

    public var isEstablished: Bool { state == .established }

    // MARK: - Public operations

    /// Sends this side's hello. Call once, when the socket is ready.
    public mutating func start() -> [Event] {
        guard state == .notStarted else { return [] }
        state = .awaitingHello
        let hello = LinkHello(
            role: role, deviceName: configuration.deviceName,
            protocolVersions: configuration.protocolVersions, appVersion: configuration.appVersion,
            identityKey: identity.publicKey, ephemeralKey: ephemeral.publicKey)
        var events: [Event] = []
        do {
            try sendPlain(.hello(hello), into: &events)
            localHello = hello
        } catch {
            fail(error, into: &events)
        }
        return events
    }

    /// Handles bytes that arrived from the peer.
    public mutating func receive(_ bytes: Data) -> [Event] {
        if case .closed = state { return [] }
        reader.append(bytes)
        var events: [Event] = []
        do {
            while true {
                if case .closed = state { break }
                let limit = state == .established ? LinkFraming.maxSealedBodyBytes : LinkFraming.maxHandshakeBodyBytes
                guard let body = try reader.nextBody(maxBodyBytes: limit) else { break }
                try handle(body, into: &events)
            }
        } catch {
            fail(error, into: &events)
        }
        return events
    }

    /// The person's answer to the pairing code.
    public mutating func confirmPairing(accepted: Bool) -> [Event] {
        guard case .comparingCodes(false, let peerAccepted) = state else { return [] }
        var events: [Event] = []
        do {
            try sendSealed(.pairConfirm(LinkPairConfirm(accepted: accepted)), into: &events)
            guard accepted else { throw LinkFailure.pairingRejected }
            state = .comparingCodes(localAccepted: true, peerAccepted: peerAccepted)
            if peerAccepted { try completePairing(into: &events) }
        } catch {
            fail(error, into: &events)
        }
        return events
    }

    /// Sends an app message. Only once established; handshake messages
    /// (`hello`, `pair*`, `forget`) are the connection's own.
    public mutating func send(_ message: LinkMessage) throws -> [Event] {
        guard state == .established else { throw LinkFailure.unexpectedMessage }
        switch message {
        case .hello, .pairRequest, .pairResponse, .pairNonce, .pairConfirm, .forget:
            throw LinkFailure.unexpectedMessage
        default:
            var events: [Event] = []
            try sendSealed(message, into: &events)
            return events
        }
    }

    /// Forgets the peer: tells it, deletes its keys here, closes.
    public mutating func forget() -> [Event] {
        guard state == .established, let peer else { return [] }
        var events: [Event] = []
        do {
            try sendSealed(.forget, into: &events)
            try deletePeer(peer)
            throw LinkFailure.closed
        } catch {
            fail(error, into: &events)
        }
        return events
    }

    /// Closes from this side.
    public mutating func close() -> [Event] {
        if case .closed = state { return [] }
        state = .closed(.closed)
        return [.closed(.closed)]
    }

    // MARK: - Frames

    /// Closes the connection after a thrown error; anything not a
    /// `LinkFailure` (an encoder error) counts as a malformed frame.
    private mutating func fail(_ error: any Error, into events: inout [Event]) {
        let failure = (error as? LinkFailure) ?? .malformedFrame
        state = .closed(failure)
        events.append(.closed(failure))
    }

    // Both senders use a sequence number only once the frame is built, so a
    // message that fails to encode or seal leaves the next number free.

    private mutating func sendPlain(_ message: LinkMessage, into events: inout [Event]) throws {
        let sequence = try outbound.peek()
        let json = try LinkFrameCodec.encode(
            LinkEnvelope(version: version ?? LinkProtocol.version, sequence: sequence, message: message))
        let frame = LinkFraming.frame(LinkFrameCodec.plainBody(json))
        outbound.advance()
        events.append(.send(frame))
    }

    private mutating func sendSealed(_ message: LinkMessage, into events: inout [Event]) throws {
        guard let key = outbound.key, let version else { throw LinkFailure.unexpectedMessage }
        let sequence = try outbound.peek()
        let json = try LinkFrameCodec.encode(LinkEnvelope(version: version, sequence: sequence, message: message))
        let frame = LinkFraming.frame(try LinkFrameCodec.sealedBody(json, sequence: sequence, key: key))
        outbound.advance()
        events.append(.send(frame))
    }

    private mutating func handle(_ body: Data, into events: inout [Event]) throws {
        switch try LinkFrameCodec.kind(of: body) {
        case .plain:
            // Nothing arrives in the clear once established. Between the key
            // derivation and that point only a refusal may (the Mac turning
            // away a phone it does not know, the phone ending a pairing whose
            // commitment failed).
            guard state != .established else { throw LinkFailure.unexpectedMessage }
            let json = Data(body.dropFirst())
            let envelope = try LinkFrameCodec.decode(json)
            if inbound.key != nil {
                guard case .error = envelope.message else { throw LinkFailure.unexpectedMessage }
            }
            try inbound.accept(envelope.sequence)
            if let version, envelope.version != version { throw LinkFailure.unsupportedVersion }
            try handlePlain(envelope.message, into: &events)

        case .sealed:
            var adoptingCandidate = false
            let key: SymmetricKey
            if let current = inbound.key {
                key = current
            } else if state == .macAwaitingPhone, let candidateKeys {
                key = candidateKeys.receiving(as: role)
                adoptingCandidate = true
            } else if state == .macAwaitingPhone {
                // A phone this Mac does not know, speaking as if paired.
                try sendPlain(.error(LinkError(code: .notPaired, message: "This iPhone isn't paired with this Mac.")),
                              into: &events)
                throw LinkFailure.notPaired
            } else {
                throw LinkFailure.unexpectedMessage
            }
            let sequence = try LinkFrameCodec.sealedSequence(body)
            try inbound.accept(sequence)
            let json = try LinkFrameCodec.open(body, sequence: sequence, key: key)
            let envelope = try LinkFrameCodec.decode(json)
            guard envelope.sequence == sequence else { throw LinkFailure.malformedFrame }
            guard envelope.version == version else { throw LinkFailure.unsupportedVersion }
            if adoptingCandidate, let candidateKeys {
                inbound.key = candidateKeys.receiving(as: role)
                outbound.key = candidateKeys.sending(as: role)
                self.candidateKeys = nil
            }
            try handleSealed(envelope.message, into: &events)
        }
    }

    // MARK: - Plain messages (handshake)

    private mutating func handlePlain(_ message: LinkMessage, into events: inout [Event]) throws {
        if case .error(let error) = message {
            switch error.code {
            case .notPaired: throw LinkFailure.notPaired
            case .unsupportedVersion: throw LinkFailure.unsupportedVersion
            default: throw LinkFailure.peer(error)
            }
        }
        switch (state, message) {
        case (.awaitingHello, .hello(let hello)):
            try receiveHello(hello, into: &events)

        case (.macAwaitingPhone, .pairRequest(let request)):
            guard configuration.pairingAllowed else {
                try sendPlain(.error(LinkError(
                    code: .pairingRejected,
                    message: "Pairing is off on this Mac. Turn it on in Settings › iPhone.")), into: &events)
                throw LinkFailure.pairingOff
            }
            candidateKeys = nil
            pairRequestName = request.deviceName
            peerName = LinkPeer.cleanName(request.deviceName, role: .phone)
            macNonce = random.bytes(LinkProtocol.keyBytes)
            commitment = try expectedCommitment(macNonce: macNonce)
            pairResponseName = configuration.deviceName
            try sendPlain(.pairResponse(LinkPairResponse(macName: pairResponseName, commitment: commitment)),
                          into: &events)
            state = .macAwaitingPhoneNonce

        case (.phoneAwaitingPairResponse, .pairResponse(let response)):
            guard response.commitment.count == LinkProtocol.keyBytes else { throw LinkFailure.malformedFrame }
            pairResponseName = response.macName
            commitment = response.commitment
            peerName = LinkPeer.cleanName(response.macName, role: .mac)
            phoneNonce = random.bytes(LinkProtocol.keyBytes)
            try sendPlain(.pairNonce(LinkPairNonce(nonce: phoneNonce)), into: &events)
            state = .phoneAwaitingMacNonce

        case (.macAwaitingPhoneNonce, .pairNonce(let nonce)):
            guard nonce.nonce.count == LinkProtocol.keyBytes else { throw LinkFailure.malformedFrame }
            phoneNonce = nonce.nonce
            try sendPlain(.pairNonce(LinkPairNonce(nonce: macNonce)), into: &events)
            try deriveFromPairing(into: &events)

        case (.phoneAwaitingMacNonce, .pairNonce(let nonce)):
            guard nonce.nonce.count == LinkProtocol.keyBytes else { throw LinkFailure.malformedFrame }
            guard try expectedCommitment(macNonce: nonce.nonce) == commitment else {
                try sendPlain(.error(LinkError(
                    code: .pairingRejected,
                    message: "Pairing didn't check out, so nothing was saved. Try again.")), into: &events)
                throw LinkFailure.commitmentMismatch
            }
            macNonce = nonce.nonce
            try deriveFromPairing(into: &events)

        default:
            throw LinkFailure.unexpectedMessage
        }
    }

    private mutating func receiveHello(_ hello: LinkHello, into events: inout [Event]) throws {
        guard hello.role != role else { throw LinkFailure.unexpectedMessage }
        guard hello.identityKey.count == LinkProtocol.keyBytes, hello.ephemeralKey.count == LinkProtocol.keyBytes,
              hello.identityKey != identity.publicKey
        else { throw LinkFailure.badKey }
        guard let common = Set(hello.protocolVersions).intersection(configuration.protocolVersions).max() else {
            try sendPlain(.error(LinkError(code: .unsupportedVersion,
                                           message: "Update Mference on both devices to connect.")), into: &events)
            throw LinkFailure.unsupportedVersion
        }
        version = common
        peerHello = hello

        let known = try storedPeer(identityKey: hello.identityKey)
        switch role {
        case .phone:
            if configuration.intent == .pair {
                pairRequestName = configuration.deviceName
                try sendPlain(.pairRequest(LinkPairRequest(deviceName: pairRequestName)), into: &events)
                state = .phoneAwaitingPairResponse
                return
            }
            guard let known else { throw LinkFailure.notPaired }
            peer = known
            let keys = try sessionKeys(pairingSecret: known.pairingSecret)
            outbound.key = keys.sending(as: role)
            inbound.key = keys.receiving(as: role)
            try sendSealedHello(into: &events)
            state = .phoneAwaitingMac
        case .mac:
            if let known {
                peer = known
                candidateKeys = try sessionKeys(pairingSecret: known.pairingSecret)
            }
            state = .macAwaitingPhone
        }
    }

    // MARK: - Sealed messages

    private mutating func handleSealed(_ message: LinkMessage, into events: inout [Event]) throws {
        switch state {
        case .macAwaitingPhone, .macAwaitingSealedHello:
            guard case .hello(let hello) = message, hello == peerHello, let peer else {
                throw LinkFailure.unexpectedMessage
            }
            state = .established
            events.append(.established(peer, version: version ?? LinkProtocol.version))

        case .phoneAwaitingMac:
            // The Mac's first sealed frame opened: the keys are confirmed.
            // Check the message before reporting the connection as up.
            guard let peer else { throw LinkFailure.unexpectedMessage }
            state = .established
            var first: [Event] = []
            try handleEstablished(message, into: &first)
            events.append(.established(peer, version: version ?? LinkProtocol.version))
            events += first

        case .comparingCodes(let localAccepted, false):
            guard case .pairConfirm(let confirm) = message else { throw LinkFailure.unexpectedMessage }
            guard confirm.accepted else { throw LinkFailure.pairingRejected }
            state = .comparingCodes(localAccepted: localAccepted, peerAccepted: true)
            if localAccepted { try completePairing(into: &events) }

        case .established:
            try handleEstablished(message, into: &events)

        default:
            throw LinkFailure.unexpectedMessage
        }
    }

    private mutating func handleEstablished(_ message: LinkMessage, into events: inout [Event]) throws {
        switch message {
        case .hello, .pairRequest, .pairResponse, .pairNonce, .pairConfirm:
            throw LinkFailure.unexpectedMessage
        case .forget:
            guard let peer else { throw LinkFailure.unexpectedMessage }
            try deletePeer(peer)
            events.append(.forgotten(peer))
            throw LinkFailure.forgotten
        default:
            events.append(.received(message))
        }
    }

    // MARK: - Keys

    /// The fields both sides agree on once the hellos are in: every field of
    /// the phone's hello, then the Mac's, then the version in use. Each later
    /// transcript starts with these.
    private func helloTranscript() throws -> [Data] {
        guard let localHello, let peerHello, let version else { throw LinkFailure.unexpectedMessage }
        let (phone, mac) = role == .phone ? (localHello, peerHello) : (peerHello, localHello)
        return LinkKeySchedule.helloFields(phone) + LinkKeySchedule.helloFields(mac)
            + [LinkKeySchedule.field(version)]
    }

    /// SHA-256 over the Mac's ephemeral key, the phone's ephemeral key and the
    /// Mac's nonce, bound to everything sent before the commitment (both
    /// hellos, so both identity keys, and the pair request).
    private func expectedCommitment(macNonce: Data) throws -> Data {
        guard let peerHello else { throw LinkFailure.unexpectedMessage }
        let macEphemeral = role == .mac ? ephemeral.publicKey : peerHello.ephemeralKey
        let phoneEphemeral = role == .phone ? ephemeral.publicKey : peerHello.ephemeralKey
        let context = LinkKeySchedule.transcriptHash(
            "commit context", try helloTranscript() + [LinkKeySchedule.field(pairRequestName)])
        return LinkKeySchedule.commitment(macEphemeral: macEphemeral, phoneEphemeral: phoneEphemeral,
                                          macNonce: macNonce, context: context)
    }

    private func handshakeSecrets() throws -> Data {
        guard let peerHello else { throw LinkFailure.unexpectedMessage }
        return try LinkKeySchedule.handshakeSecrets(
            role: role, identity: identity, ephemeral: ephemeral,
            peerIdentityKey: peerHello.identityKey, peerEphemeralKey: peerHello.ephemeralKey)
    }

    private func sessionKeys(pairingSecret: Data) throws -> LinkSessionKeys {
        guard pairingSecret.count == LinkProtocol.keyBytes else { throw LinkFailure.badKey }
        let transcript = LinkKeySchedule.transcriptHash("session", try helloTranscript())
        return LinkKeySchedule.sessionKeys(secrets: try handshakeSecrets(), pairingSecret: pairingSecret,
                                           transcript: transcript)
    }

    private mutating func deriveFromPairing(into events: inout [Event]) throws {
        guard let peerHello else { throw LinkFailure.unexpectedMessage }
        let secrets = try handshakeSecrets()
        // Every value the pairing exchanged, in one fixed order on both sides.
        // The Mac's name and nonce were fixed by the commitment before the
        // phone's nonce was known, so nothing here is left to choose after it.
        let transcript = LinkKeySchedule.transcriptHash("pairing", try helloTranscript() + [
            LinkKeySchedule.field(pairRequestName), LinkKeySchedule.field(pairResponseName), commitment,
            phoneNonce, macNonce,
        ])
        let pairingSecret = LinkKeySchedule.pairingSecret(secrets: secrets, transcript: transcript)
        let keys = LinkKeySchedule.sessionKeys(secrets: secrets, pairingSecret: pairingSecret, transcript: transcript)
        outbound.key = keys.sending(as: role)
        inbound.key = keys.receiving(as: role)
        peer = LinkPeer(role: peerHello.role, name: peerName, identityKey: peerHello.identityKey,
                        pairingSecret: pairingSecret, pairedAt: Date())
        state = .comparingCodes(localAccepted: false, peerAccepted: false)
        events.append(.pairingCode(LinkKeySchedule.pairingCode(secrets: secrets, transcript: transcript),
                                   peerName: peerName))
    }

    private mutating func completePairing(into events: inout [Event]) throws {
        guard let peer else { throw LinkFailure.unexpectedMessage }
        do {
            try store.savePeer(peer)
        } catch {
            throw (error as? LinkFailure) ?? .keyStore("\(error)")
        }
        events.append(.paired(peer))
        switch role {
        case .phone:
            try sendSealedHello(into: &events)
            state = .phoneAwaitingMac
        case .mac:
            state = .macAwaitingSealedHello
        }
    }

    private mutating func sendSealedHello(into events: inout [Event]) throws {
        guard let localHello else { throw LinkFailure.unexpectedMessage }
        try sendSealed(.hello(localHello), into: &events)
    }

    private func storedPeer(identityKey: Data) throws -> LinkPeer? {
        do {
            guard let peer = try store.peer(id: LinkPeer.id(for: identityKey)),
                  peer.identityKey == identityKey, peer.role != role
            else { return nil }
            return peer
        } catch {
            throw (error as? LinkFailure) ?? .keyStore("\(error)")
        }
    }

    private func deletePeer(_ peer: LinkPeer) throws {
        do {
            try store.deletePeer(id: peer.id)
        } catch {
            throw (error as? LinkFailure) ?? .keyStore("\(error)")
        }
    }
}

import Foundation
@testable import MferenceLink

/// Reproducible random bytes (SplitMix64), so keys, nonces and codes are the
/// same on every run.
final class SeededLinkRandom: LinkRandomSource, @unchecked Sendable {
    private let lock = NSLock()
    private var state: UInt64

    init(seed: UInt64) {
        state = seed
    }

    func bytes(_ count: Int) -> Data {
        lock.withLock {
            var out = Data(capacity: count)
            while out.count < count {
                state &+= 0x9E37_79B9_7F4A_7C15
                var z = state
                z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
                z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
                z ^= z >> 31
                withUnsafeBytes(of: z) { out.append(contentsOf: $0.prefix(count - out.count)) }
            }
            return out
        }
    }
}

typealias Event = LinkConnection.Event

extension Array where Element == LinkConnection.Event {
    var frames: [Data] {
        compactMap { if case .send(let data) = $0 { data } else { nil } }
    }

    var nonSend: [LinkConnection.Event] {
        filter { if case .send = $0 { false } else { true } }
    }

    var code: LinkPairingCode? {
        for event in self { if case .pairingCode(let code, _) = event { return code } }
        return nil
    }

    var failure: LinkFailure? {
        for event in self { if case .closed(let failure) = event { return failure } }
        return nil
    }

    var established: LinkPeer? {
        for event in self { if case .established(let peer, _) = event { return peer } }
        return nil
    }

    var paired: LinkPeer? {
        for event in self { if case .paired(let peer) = event { return peer } }
        return nil
    }

    var received: [LinkMessage] {
        compactMap { if case .received(let message) = $0 { message } else { nil } }
    }
}

/// Two connections joined by an in-memory pipe. `deliver` can rewrite or drop
/// frames on their way (an attacker on the network).
struct LinkPipe {
    var phone: LinkConnection
    var mac: LinkConnection
    var phoneEvents: [Event] = []
    var macEvents: [Event] = []
    private var toMac: [Data] = []
    private var toPhone: [Data] = []
    /// Called for each frame; returns the frames to deliver instead.
    var intercept: (_ from: LinkRole, _ frame: Data) -> [Data] = { _, frame in [frame] }

    init(phone: LinkConnection, mac: LinkConnection) {
        self.phone = phone
        self.mac = mac
    }

    mutating func start() {
        route(phone.start(), from: .phone)
        route(mac.start(), from: .mac)
        pump()
    }

    mutating func route(_ events: [Event], from role: LinkRole) {
        for event in events {
            if case .send(let frame) = event {
                if role == .phone { toMac.append(frame) } else { toPhone.append(frame) }
            }
        }
        if role == .phone { phoneEvents += events.nonSend } else { macEvents += events.nonSend }
    }

    mutating func pump() {
        while !toMac.isEmpty || !toPhone.isEmpty {
            if !toMac.isEmpty {
                let frame = toMac.removeFirst()
                for delivered in intercept(.phone, frame) { route(mac.receive(delivered), from: .mac) }
            }
            if !toPhone.isEmpty {
                let frame = toPhone.removeFirst()
                for delivered in intercept(.mac, frame) { route(phone.receive(delivered), from: .phone) }
            }
        }
    }

    mutating func confirm(phone phoneAccepts: Bool?, mac macAccepts: Bool?) {
        if let phoneAccepts { route(phone.confirmPairing(accepted: phoneAccepts), from: .phone) }
        pump()
        if let macAccepts { route(mac.confirmPairing(accepted: macAccepts), from: .mac) }
        pump()
    }

    mutating func sendFromMac(_ message: LinkMessage) throws {
        route(try mac.send(message), from: .mac)
        pump()
    }

    mutating func sendFromPhone(_ message: LinkMessage) throws {
        route(try phone.send(message), from: .phone)
        pump()
    }
}

enum LinkFixture {
    static let capabilities = LinkMessage.capabilities(LinkCapabilities(
        macName: "MacBook Pro",
        models: [LinkModel(id: "deep", title: "Deep 27B", isReady: true, isDefault: true)],
        webSearch: true, modelTransfer: true))

    static func phone(_ store: any LinkKeyStore, intent: LinkConnection.Intent, seed: UInt64,
                      versions: [Int] = [LinkProtocol.version]) throws -> LinkConnection {
        try LinkConnection(
            configuration: .init(role: .phone, deviceName: "iPhone", appVersion: "1.0",
                                 protocolVersions: versions, intent: intent),
            store: store, random: SeededLinkRandom(seed: seed))
    }

    static func mac(_ store: any LinkKeyStore, pairing: Bool, seed: UInt64,
                    versions: [Int] = [LinkProtocol.version]) throws -> LinkConnection {
        try LinkConnection(
            configuration: .init(role: .mac, deviceName: "MacBook Pro", appVersion: "1.0",
                                 protocolVersions: versions, pairingAllowed: pairing),
            store: store, random: SeededLinkRandom(seed: seed))
    }

    /// Pairs a fresh phone and Mac; returns their stores.
    static func pairedStores(seed: UInt64 = 1) throws -> (phone: InMemoryLinkKeyStore, mac: InMemoryLinkKeyStore) {
        let phoneStore = InMemoryLinkKeyStore()
        let macStore = InMemoryLinkKeyStore()
        var pipe = LinkPipe(phone: try phone(phoneStore, intent: .pair, seed: seed),
                            mac: try mac(macStore, pairing: true, seed: seed + 1000))
        pipe.start()
        pipe.confirm(phone: true, mac: true)
        return (phoneStore, macStore)
    }

    /// An established connection between two paired stores.
    static func established(seed: UInt64 = 7) throws -> LinkPipe {
        let stores = try pairedStores()
        var pipe = LinkPipe(phone: try phone(stores.phone, intent: .connect, seed: seed),
                            mac: try mac(stores.mac, pairing: false, seed: seed + 1000))
        pipe.start()
        try pipe.sendFromMac(capabilities)
        return pipe
    }

    /// Rewrites the envelope inside a plain frame (the attacker can, since
    /// plain frames carry no tag).
    static func rewritePlain(_ frame: Data, _ change: (inout LinkEnvelope) -> Void) throws -> Data {
        let body = frame.dropFirst(LinkFraming.lengthBytes)
        guard body.first == 0 else { return frame }
        var envelope = try JSONDecoder().decode(LinkEnvelope.self, from: Data(body.dropFirst()))
        change(&envelope)
        return LinkFraming.frame(Data([0]) + (try JSONEncoder().encode(envelope)))
    }

    static func plainMessage(_ frame: Data) -> LinkMessage? {
        let body = frame.dropFirst(LinkFraming.lengthBytes)
        guard body.first == 0 else { return nil }
        return try? JSONDecoder().decode(LinkEnvelope.self, from: Data(body.dropFirst())).message
    }
}

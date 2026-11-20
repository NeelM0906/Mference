import Foundation
import Testing
@testable import MferenceLink

/// Gate G-M8: the link refuses a wrong code, a tampered, replayed or
/// out-of-order frame, an unpaired peer, an oversized frame and a non-local
/// address. Everything runs in memory; nothing touches the network.
@Suite("Link security (G-M8)")
struct LinkSecurityTests {
    // MARK: - Pairing

    @Test func pairingShowsTheSameCodeAndSavesBothSides() throws {
        let phoneStore = InMemoryLinkKeyStore()
        let macStore = InMemoryLinkKeyStore()
        var pipe = LinkPipe(phone: try LinkFixture.phone(phoneStore, intent: .pair, seed: 1),
                            mac: try LinkFixture.mac(macStore, pairing: true, seed: 2))
        pipe.start()

        let phoneCode = try #require(pipe.phoneEvents.code)
        let macCode = try #require(pipe.macEvents.code)
        #expect(phoneCode == macCode)
        #expect(phoneCode.digitsString.count == LinkPairingCode.digits)
        #expect(pipe.phoneEvents.contains(.pairingCode(phoneCode, peerName: "MacBook Pro")))
        #expect(pipe.macEvents.contains(.pairingCode(macCode, peerName: "iPhone")))
        // Nothing is saved before both people confirm.
        #expect(try phoneStore.peers().isEmpty)
        #expect(try macStore.peers().isEmpty)

        pipe.confirm(phone: true, mac: true)
        let macOnPhone = try #require(pipe.phoneEvents.paired)
        let phoneOnMac = try #require(pipe.macEvents.paired)
        #expect(macOnPhone.role == .mac && macOnPhone.name == "MacBook Pro")
        #expect(phoneOnMac.role == .phone && phoneOnMac.name == "iPhone")
        #expect(macOnPhone.pairingSecret == phoneOnMac.pairingSecret)
        #expect(macOnPhone.identityKey == pipe.mac.identityKey)
        #expect(phoneOnMac.identityKey == pipe.phone.identityKey)
        #expect(try phoneStore.peers() == [macOnPhone])
        #expect(try macStore.peers() == [phoneOnMac])

        // The pairing connection goes straight on, sealed.
        #expect(pipe.macEvents.established == phoneOnMac)
        try pipe.sendFromMac(LinkFixture.capabilities)
        #expect(pipe.phoneEvents.established == macOnPhone)
        #expect(pipe.phoneEvents.received == [LinkFixture.capabilities])
    }

    @Test func eitherSideMayConfirmFirst() throws {
        var pipe = LinkPipe(phone: try LinkFixture.phone(InMemoryLinkKeyStore(), intent: .pair, seed: 3),
                            mac: try LinkFixture.mac(InMemoryLinkKeyStore(), pairing: true, seed: 4))
        pipe.start()
        pipe.confirm(phone: nil, mac: true)
        #expect(pipe.macEvents.paired == nil)
        pipe.confirm(phone: true, mac: nil)
        #expect(pipe.phoneEvents.paired != nil)
        #expect(pipe.macEvents.paired != nil)
        #expect(pipe.macEvents.established != nil)
    }

    @Test(arguments: [(false, true), (true, false), (false, false)])
    func wrongCodeSavesNothing(phoneAccepts: Bool, macAccepts: Bool) throws {
        let phoneStore = InMemoryLinkKeyStore()
        let macStore = InMemoryLinkKeyStore()
        var pipe = LinkPipe(phone: try LinkFixture.phone(phoneStore, intent: .pair, seed: 5),
                            mac: try LinkFixture.mac(macStore, pairing: true, seed: 6))
        pipe.start()
        pipe.confirm(phone: phoneAccepts, mac: macAccepts)
        #expect(pipe.phoneEvents.failure == .pairingRejected)
        #expect(pipe.macEvents.failure == .pairingRejected)
        #expect(pipe.phoneEvents.paired == nil && pipe.macEvents.paired == nil)
        #expect(try phoneStore.peers().isEmpty)
        #expect(try macStore.peers().isEmpty)
    }

    /// A device in the middle pairs with each side separately. The two codes
    /// it causes differ, so the person sees the mismatch and rejects.
    @Test(arguments: 0..<16)
    func manInTheMiddleGetsDifferentCodes(seed: UInt64) throws {
        let base = seed * 10 + 100
        var phoneSide = LinkPipe(phone: try LinkFixture.phone(InMemoryLinkKeyStore(), intent: .pair, seed: base),
                                 mac: try LinkFixture.mac(InMemoryLinkKeyStore(), pairing: true, seed: base + 1))
        var macSide = LinkPipe(phone: try LinkFixture.phone(InMemoryLinkKeyStore(), intent: .pair, seed: base + 2),
                               mac: try LinkFixture.mac(InMemoryLinkKeyStore(), pairing: true, seed: base + 3))
        phoneSide.start()
        macSide.start()
        let phoneCode = try #require(phoneSide.phoneEvents.code)
        let macCode = try #require(macSide.macEvents.code)
        #expect(phoneCode != macCode)
    }

    /// The Mac must reveal the nonce it committed to; a changed nonce (an
    /// attacker trying to steer the code) ends pairing on the phone.
    @Test func changedMacNonceFailsTheCommitment() throws {
        var pipe = LinkPipe(phone: try LinkFixture.phone(InMemoryLinkKeyStore(), intent: .pair, seed: 8),
                            mac: try LinkFixture.mac(InMemoryLinkKeyStore(), pairing: true, seed: 9))
        pipe.intercept = { from, frame in
            guard from == .mac, case .pairNonce = LinkFixture.plainMessage(frame) else { return [frame] }
            let changed = try? LinkFixture.rewritePlain(frame) {
                $0.message = .pairNonce(LinkPairNonce(nonce: Data(repeating: 0xAB, count: 32)))
            }
            return [changed ?? frame]
        }
        pipe.start()
        #expect(pipe.phoneEvents.failure == .commitmentMismatch)
        #expect(pipe.phoneEvents.code == nil)
        // The phone tells the Mac in the clear; nothing is saved anywhere.
        guard case .peer(let error) = pipe.macEvents.failure else {
            Issue.record("expected the phone's refusal, got \(String(describing: pipe.macEvents.failure))")
            return
        }
        #expect(error.code == .pairingRejected)
        #expect(pipe.phoneEvents.paired == nil && pipe.macEvents.paired == nil)
    }

    /// The code and keys come from the decoded field values, not the JSON
    /// bytes they travel in. A device on the path that adds unknown fields,
    /// reorders keys and changes spacing in every handshake frame (which it
    /// could repeat freely to re-roll the code) changes nothing.
    @Test func reshapedHandshakeJSONChangesNeitherCodeNorKeys() throws {
        final class Count: @unchecked Sendable { var changed = 0 }
        func pairing(reshape: Bool) throws -> (pipe: LinkPipe, changed: Int) {
            let count = Count()
            var pipe = LinkPipe(phone: try LinkFixture.phone(InMemoryLinkKeyStore(), intent: .pair, seed: 40),
                                mac: try LinkFixture.mac(InMemoryLinkKeyStore(), pairing: true, seed: 41))
            if reshape {
                pipe.intercept = { _, frame in
                    let reshaped = LinkFixture.reshapePlain(frame)
                    if reshaped != frame { count.changed += 1 }
                    return [reshaped]
                }
            }
            pipe.start()
            pipe.confirm(phone: true, mac: true)
            return (pipe, count.changed)
        }
        let untouched = try pairing(reshape: false).pipe
        let result = try pairing(reshape: true)
        var reshaped = result.pipe
        // Both hellos, the request, the response and both nonces were rewritten.
        #expect(result.changed == 6)

        let code = try #require(untouched.phoneEvents.code)
        #expect(reshaped.phoneEvents.code == code)
        #expect(reshaped.macEvents.code == code)
        let secret = try #require(untouched.phoneEvents.paired?.pairingSecret)
        #expect(reshaped.phoneEvents.paired?.pairingSecret == secret)
        #expect(reshaped.macEvents.paired?.pairingSecret == secret)
        // The session keys agree too: sealed frames open both ways.
        try reshaped.sendFromMac(LinkFixture.capabilities)
        #expect(reshaped.phoneEvents.established != nil)
        #expect(reshaped.phoneEvents.received == [LinkFixture.capabilities])
        #expect(reshaped.phoneEvents.failure == nil && reshaped.macEvents.failure == nil)
    }

    /// The same holds for a paired connection: reshaped hellos still give
    /// both sides the same session keys.
    @Test func reshapedHellosStillConnect() throws {
        let stores = try LinkFixture.pairedStores(seed: 42)
        var pipe = LinkPipe(phone: try LinkFixture.phone(stores.phone, intent: .connect, seed: 43),
                            mac: try LinkFixture.mac(stores.mac, pairing: false, seed: 44))
        pipe.intercept = { _, frame in [LinkFixture.reshapePlain(frame)] }
        pipe.start()
        try pipe.sendFromMac(LinkFixture.capabilities)
        #expect(pipe.macEvents.established != nil && pipe.phoneEvents.established != nil)
        #expect(pipe.phoneEvents.received == [LinkFixture.capabilities])
    }

    /// The commitment covers both ephemeral keys and the Mac's nonce: a
    /// commitment made for another key or nonce does not check.
    @Test func commitmentBindsBothEphemeralsAndTheNonce() {
        let mac = Data(repeating: 1, count: 32), phone = Data(repeating: 2, count: 32)
        let nonce = Data(repeating: 3, count: 32), context = Data(repeating: 4, count: 32)
        let other = Data(repeating: 9, count: 32)
        let base = LinkKeySchedule.commitment(macEphemeral: mac, phoneEphemeral: phone, macNonce: nonce, context: context)
        #expect(base.count == 32)
        #expect(base != LinkKeySchedule.commitment(macEphemeral: other, phoneEphemeral: phone, macNonce: nonce, context: context))
        #expect(base != LinkKeySchedule.commitment(macEphemeral: mac, phoneEphemeral: other, macNonce: nonce, context: context))
        #expect(base != LinkKeySchedule.commitment(macEphemeral: mac, phoneEphemeral: phone, macNonce: other, context: context))
        #expect(base != LinkKeySchedule.commitment(macEphemeral: mac, phoneEphemeral: phone, macNonce: nonce, context: other))
        // Swapping the two keys is a different commitment too.
        #expect(base != LinkKeySchedule.commitment(macEphemeral: phone, phoneEphemeral: mac, macNonce: nonce, context: context))
    }

    /// The code depends on the shared secret and on every transcript part,
    /// so changing any key, name or nonce after the fact changes the code.
    @Test func codeBindsTheWholeTranscript() {
        let secrets = Data(repeating: 7, count: 96)
        let parts = (0..<6).map { Data(repeating: UInt8($0 + 1), count: 40) }
        func code(_ secrets: Data, _ parts: [Data]) -> LinkPairingCode {
            LinkKeySchedule.pairingCode(secrets: secrets,
                                        transcript: LinkKeySchedule.transcriptHash("pairing", parts))
        }
        let base = code(secrets, parts)
        var changedSecrets = secrets
        changedSecrets[0] ^= 1
        var differing = code(changedSecrets, parts) != base ? 1 : 0
        for index in parts.indices {
            var changed = parts
            changed[index][changed[index].startIndex] ^= 1
            if code(secrets, changed) != base { differing += 1 }
        }
        // 7 independent changes; each could collide with probability 1e-6.
        #expect(differing == 7)
    }

    /// The transcript behind the code carries both identity keys and both
    /// ephemeral keys, each as its own length-prefixed part: changing any one
    /// of them, or swapping two, changes the transcript and the code.
    @Test func transcriptCarriesAllFourKeys() {
        func key(_ byte: UInt8) -> Data { Data(repeating: byte, count: 32) }
        let phone = LinkHello(role: .phone, deviceName: "iPhone", appVersion: "1.0",
                              identityKey: key(1), ephemeralKey: key(2))
        let mac = LinkHello(role: .mac, deviceName: "MacBook Pro", appVersion: "1.0",
                            identityKey: key(3), ephemeralKey: key(4))
        let parts = LinkKeySchedule.helloParts(phone: phone, mac: mac, version: 1)
        for key in [phone.identityKey, phone.ephemeralKey, mac.identityKey, mac.ephemeralKey] {
            #expect(parts.contains(key))
        }
        // Phone hello fields, then the Mac's, then the version.
        #expect(parts.count == 2 * LinkKeySchedule.helloFields(phone).count + 1)
        #expect(Array(parts[0..<6]) == LinkKeySchedule.helloFields(phone))

        let secrets = Data(repeating: 7, count: 96)
        func transcript(_ phone: LinkHello, _ mac: LinkHello) -> Data {
            LinkKeySchedule.pairingTranscript(phone: phone, mac: mac, version: 1, phoneName: "iPhone",
                                              macName: "MacBook Pro", commitment: key(5),
                                              phoneNonce: key(6), macNonce: key(8))
        }
        let base = transcript(phone, mac)
        let baseCode = LinkKeySchedule.pairingCode(secrets: secrets, transcript: base)
        var variants: [(LinkHello, LinkHello)] = []
        var changed = phone; changed.identityKey = key(9); variants.append((changed, mac))
        changed = phone; changed.ephemeralKey = key(9); variants.append((changed, mac))
        changed = mac; changed.identityKey = key(9); variants.append((phone, changed))
        changed = mac; changed.ephemeralKey = key(9); variants.append((phone, changed))
        // Swapped roles of two keys: the order is fixed, so this differs too.
        changed = phone; changed.identityKey = key(2); changed.ephemeralKey = key(1); variants.append((changed, mac))
        var other = mac; other.identityKey = key(1); changed = phone; changed.identityKey = key(3)
        variants.append((changed, other))
        for (phoneHello, macHello) in variants {
            let changedTranscript = transcript(phoneHello, macHello)
            #expect(changedTranscript != base)
            #expect(LinkKeySchedule.pairingCode(secrets: secrets, transcript: changedTranscript) != baseCode)
            #expect(LinkKeySchedule.sessionTranscript(phone: phoneHello, mac: macHello, version: 1)
                != LinkKeySchedule.sessionTranscript(phone: phone, mac: mac, version: 1))
            #expect(LinkKeySchedule.commitContext(phone: phoneHello, mac: macHello, version: 1, phoneName: "iPhone")
                != LinkKeySchedule.commitContext(phone: phone, mac: mac, version: 1, phoneName: "iPhone"))
        }
    }

    /// An attacker posing as the Mac picks its key after it has seen the
    /// phone's key, aiming at the code the real Mac shows. It must commit
    /// before the phone reveals its nonce, so no choice of key helps: over
    /// many re-picked keys (each against a fresh phone nonce) it never hits.
    @Test func attackerRepickingItsKeyCannotSteerTheCode() throws {
        var realSide = LinkPipe(phone: try LinkFixture.phone(InMemoryLinkKeyStore(), intent: .pair, seed: 900),
                                mac: try LinkFixture.mac(InMemoryLinkKeyStore(), pairing: true, seed: 901))
        realSide.start()
        let target = try #require(realSide.macEvents.code)
        var hits = 0
        for attempt in UInt64(0)..<300 {
            var phone = try LinkFixture.phone(InMemoryLinkKeyStore(), intent: .pair, seed: 10_000 + attempt)
            let phoneHello = phone.start().frames
            // The attacker sees the phone's hello first, then picks its key.
            var attacker = try LinkFixture.mac(InMemoryLinkKeyStore(), pairing: true, seed: 20_000 + attempt)
            var toMac = phoneHello
            var toPhone = attacker.start().frames
            var phoneEvents: [Event] = []
            while !toMac.isEmpty || !toPhone.isEmpty {
                let m = toMac; toMac = []
                for f in m { toPhone += attacker.receive(f).frames }
                let p = toPhone; toPhone = []
                for f in p { let e = phone.receive(f); phoneEvents += e; toMac += e.frames }
            }
            if phoneEvents.code == target { hits += 1 }
        }
        #expect(hits == 0)
    }

    /// Re-picking the key mid-pairing (a second hello after the commitment)
    /// is refused outright.
    @Test func secondHelloDuringPairingIsRefused() throws {
        var pipe = LinkPipe(phone: try LinkFixture.phone(InMemoryLinkKeyStore(), intent: .pair, seed: 910),
                            mac: try LinkFixture.mac(InMemoryLinkKeyStore(), pairing: true, seed: 911))
        var macHello: Data?
        var injected = false
        pipe.intercept = { from, frame in
            guard from == .mac else { return [frame] }
            if case .hello = LinkFixture.plainMessage(frame) { macHello = frame }
            guard !injected, case .pairResponse = LinkFixture.plainMessage(frame), let macHello else { return [frame] }
            injected = true
            // Send the response, then a fresh hello with another key, as the
            // next frame in sequence.
            let rekeyed = try? LinkFixture.rewritePlain(macHello) {
                guard case .hello(var hello) = $0.message else { return }
                hello.ephemeralKey = (try? LinkKeyPair.generate(using: SeededLinkRandom(seed: 912)).publicKey) ?? Data()
                $0.message = .hello(hello)
                $0.sequence = 3
            }
            return [frame, rekeyed ?? frame]
        }
        pipe.start()
        #expect(pipe.phoneEvents.failure == .unexpectedMessage)
        #expect(pipe.phoneEvents.code == nil)
    }

    @Test func pairingOffOnTheMacRefusesAndSavesNothing() throws {
        let macStore = InMemoryLinkKeyStore()
        var pipe = LinkPipe(phone: try LinkFixture.phone(InMemoryLinkKeyStore(), intent: .pair, seed: 10),
                            mac: try LinkFixture.mac(macStore, pairing: false, seed: 11))
        pipe.start()
        #expect(pipe.macEvents.failure == .pairingOff)
        guard case .peer(let error) = pipe.phoneEvents.failure else {
            Issue.record("expected the Mac's refusal, got \(String(describing: pipe.phoneEvents.failure))")
            return
        }
        #expect(error.code == .pairingRejected)
        #expect(try macStore.peers().isEmpty)
    }

    @Test func codeFormatting() {
        #expect(LinkPairingCode(value: 42_917).digitsString == "042917")
        #expect(LinkPairingCode(value: 42_917).description == "042 917")
        #expect(LinkPairingCode(value: 0).description == "000 000")
    }

    // MARK: - Paired connections

    @Test func savedKeysConnectAndCarryMessagesBothWays() throws {
        var pipe = try LinkFixture.established()
        #expect(pipe.mac.isEstablished && pipe.phone.isEstablished)
        #expect(pipe.phoneEvents.received == [LinkFixture.capabilities])

        let search = LinkMessage.webSearch(LinkWebSearchRequest(query: "tides today"))
        try pipe.sendFromPhone(search)
        #expect(pipe.macEvents.received == [search])
        let chunk = LinkMessage.modelChunk(LinkModelChunk(transferID: UUID(), offset: 0,
                                                          data: Data(repeating: 1, count: 1_000_000)))
        try pipe.sendFromMac(chunk)
        #expect(pipe.phoneEvents.received.last == chunk)
        #expect(pipe.phoneEvents.failure == nil && pipe.macEvents.failure == nil)
    }

    @Test func framesSplitIntoSingleBytesStillWork() throws {
        let stores = try LinkFixture.pairedStores()
        var phone = try LinkFixture.phone(stores.phone, intent: .connect, seed: 20)
        var mac = try LinkFixture.mac(stores.mac, pairing: false, seed: 21)
        // Deliver every frame one byte at a time.
        var toMac = phone.start().frames
        var toPhone = mac.start().frames
        var macEvents: [Event] = []
        var phoneEvents: [Event] = []
        while !toMac.isEmpty || !toPhone.isEmpty {
            for frame in toMac { for byte in frame { let e = mac.receive(Data([byte])); macEvents += e; toPhone += e.frames } }
            toMac = []
            for frame in toPhone { for byte in frame { let e = phone.receive(Data([byte])); phoneEvents += e; toMac += e.frames } }
            toPhone = []
        }
        #expect(macEvents.established != nil)
        #expect(mac.isEstablished)
        #expect(phoneEvents.failure == nil)
    }

    @Test func eachConnectionHasFreshKeys() throws {
        let stores = try LinkFixture.pairedStores()
        var first = LinkPipe(phone: try LinkFixture.phone(stores.phone, intent: .connect, seed: 30),
                             mac: try LinkFixture.mac(stores.mac, pairing: false, seed: 31))
        var second = LinkPipe(phone: try LinkFixture.phone(stores.phone, intent: .connect, seed: 32),
                              mac: try LinkFixture.mac(stores.mac, pairing: false, seed: 33))
        first.start()
        second.start()
        // A frame from one connection does not open on another.
        let frame = try #require(try first.mac.send(LinkFixture.capabilities).frames.first)
        let events = second.phone.receive(frame)
        #expect(events.failure == .authenticationFailed)
    }

    // MARK: - Tampered, replayed, out-of-order frames

    @Test(arguments: [LinkFraming.lengthBytes + 9, -20, -1])
    func tamperedFrameIsRefused(offset: Int) throws {
        var pipe = try LinkFixture.established()
        var frame = try #require(try pipe.mac.send(.jobCancel(LinkJobCancel(jobID: UUID()))).frames.first)
        // A byte of ciphertext, of the tag, the last byte of the tag.
        let index = offset >= 0 ? offset : frame.count + offset
        frame[index] ^= 0x01
        let events = pipe.phone.receive(frame)
        #expect(events.failure == .authenticationFailed)
        #expect(events.received.isEmpty)
        // Closed: nothing more is accepted, even a genuine frame.
        let next = try? pipe.mac.send(.jobCancel(LinkJobCancel(jobID: UUID()))).frames.first
        #expect(pipe.phone.receive(try #require(next)).isEmpty)
    }

    @Test func tamperedSequenceIsRefused() throws {
        var pipe = try LinkFixture.established()
        var frame = try #require(try pipe.mac.send(.jobCancel(LinkJobCancel(jobID: UUID()))).frames.first)
        // Change the sequence to a later one: the counter refuses it before
        // decrypting, and the tag would not check either.
        frame[LinkFraming.lengthBytes + 8] ^= 0x04
        #expect(pipe.phone.receive(frame).failure == .outOfOrderFrame)
    }

    @Test func replayedFrameIsRefused() throws {
        var pipe = try LinkFixture.established()
        let message = LinkMessage.jobCancel(LinkJobCancel(jobID: UUID()))
        let frame = try #require(try pipe.mac.send(message).frames.first)
        #expect(pipe.phone.receive(frame).received == [message])
        let replay = pipe.phone.receive(frame)
        #expect(replay.failure == .replayedFrame)
        #expect(replay.received.isEmpty)
    }

    @Test func replayedHandshakeFromAnEarlierConnectionIsRefused() throws {
        // Record every frame the phone sent on one connection, then play them
        // to the Mac on a new one: the Mac's fresh ephemeral key gives new
        // keys, so the recorded sealed hello does not open.
        let stores = try LinkFixture.pairedStores()
        var recorded: [Data] = []
        var first = LinkPipe(phone: try LinkFixture.phone(stores.phone, intent: .connect, seed: 40),
                             mac: try LinkFixture.mac(stores.mac, pairing: false, seed: 41))
        first.intercept = { from, frame in
            if from == .phone { recorded.append(frame) }
            return [frame]
        }
        first.start()
        #expect(first.macEvents.established != nil)

        var mac = try LinkFixture.mac(stores.mac, pairing: false, seed: 42)
        _ = mac.start()
        var events: [Event] = []
        for frame in recorded { events += mac.receive(frame) }
        #expect(events.established == nil)
        #expect(events.failure == .authenticationFailed)
    }

    @Test func outOfOrderFrameIsRefused() throws {
        var pipe = try LinkFixture.established()
        // The first frame is lost on the way; the second skips ahead.
        _ = try pipe.mac.send(.jobCancel(LinkJobCancel(jobID: UUID())))
        let second = try #require(try pipe.mac.send(.jobCancel(LinkJobCancel(jobID: UUID()))).frames.first)
        let events = pipe.phone.receive(second)
        #expect(events.failure == .outOfOrderFrame)
        #expect(events.received.isEmpty)
    }

    @Test func plainFrameAfterTheHandshakeIsRefused() throws {
        var pipe = try LinkFixture.established()
        let envelope = LinkEnvelope(sequence: 99, message: .forget)
        let frame = LinkFraming.frame(Data([0]) + (try JSONEncoder().encode(envelope)))
        #expect(pipe.phone.receive(frame).failure == .unexpectedMessage)
    }

    @Test func handshakeMessagesCannotBeSentAsAppMessages() throws {
        var pipe = try LinkFixture.established()
        #expect(throws: LinkFailure.unexpectedMessage) {
            _ = try pipe.phone.send(.pairConfirm(LinkPairConfirm(accepted: true)))
        }
        #expect(throws: LinkFailure.unexpectedMessage) { _ = try pipe.phone.send(.forget) }
    }

    // MARK: - Unpaired peers

    @Test func macRefusesAPhoneItDoesNotKnow() throws {
        let stores = try LinkFixture.pairedStores()
        // The Mac forgot the phone while it was away.
        for peer in try stores.mac.peers() { try stores.mac.deletePeer(id: peer.id) }
        var pipe = LinkPipe(phone: try LinkFixture.phone(stores.phone, intent: .connect, seed: 50),
                            mac: try LinkFixture.mac(stores.mac, pairing: false, seed: 51))
        pipe.start()
        #expect(pipe.macEvents.failure == .notPaired)
        #expect(pipe.phoneEvents.failure == .notPaired)
        #expect(pipe.macEvents.established == nil && pipe.phoneEvents.established == nil)
    }

    @Test func phoneWithoutAPairingDoesNotConnect() throws {
        var pipe = LinkPipe(phone: try LinkFixture.phone(InMemoryLinkKeyStore(), intent: .connect, seed: 52),
                            mac: try LinkFixture.mac(InMemoryLinkKeyStore(), pairing: true, seed: 53))
        pipe.start()
        #expect(pipe.phoneEvents.failure == .notPaired)
        #expect(pipe.macEvents.established == nil)
    }

    /// A device that copies the phone's identity key but holds the wrong
    /// pairing secret (or the right secret but another key) gets nothing.
    @Test func wrongPairingSecretFailsAuthentication() throws {
        let stores = try LinkFixture.pairedStores()
        var macRecord = try #require(try stores.phone.peers().first)
        macRecord.pairingSecret = Data(repeating: 0x5A, count: 32)
        try stores.phone.savePeer(macRecord)
        var pipe = LinkPipe(phone: try LinkFixture.phone(stores.phone, intent: .connect, seed: 54),
                            mac: try LinkFixture.mac(stores.mac, pairing: false, seed: 55))
        pipe.start()
        #expect(pipe.macEvents.failure == .authenticationFailed)
        #expect(pipe.macEvents.established == nil)
    }

    @Test func impostorWithAStolenPairingSecretButNotTheKeyFails() throws {
        let stores = try LinkFixture.pairedStores()
        // Same Mac record (and pairing secret), different identity key.
        let impostor = InMemoryLinkKeyStore(peers: try stores.phone.peers())
        var pipe = LinkPipe(phone: try LinkFixture.phone(impostor, intent: .connect, seed: 56),
                            mac: try LinkFixture.mac(stores.mac, pairing: false, seed: 57))
        pipe.start()
        #expect(pipe.macEvents.failure == .notPaired)
        #expect(pipe.macEvents.established == nil)
    }

    @Test func forgetDeletesKeysOnBothSides() throws {
        let stores = try LinkFixture.pairedStores()
        var pipe = LinkPipe(phone: try LinkFixture.phone(stores.phone, intent: .connect, seed: 63),
                            mac: try LinkFixture.mac(stores.mac, pairing: false, seed: 64))
        pipe.start()
        try pipe.sendFromMac(LinkFixture.capabilities)
        let phonePeer = try #require(pipe.macEvents.established)
        pipe.route(pipe.phone.forget(), from: .phone)
        pipe.pump()
        #expect(pipe.macEvents.contains(.forgotten(phonePeer)))
        #expect(pipe.macEvents.failure == .forgotten)
        #expect(pipe.phoneEvents.failure == .closed)
        #expect(try stores.phone.peers().isEmpty)
        #expect(try stores.mac.peers().isEmpty)
    }

    @Test func noCommonVersionIsRefused() throws {
        let stores = try LinkFixture.pairedStores()
        var pipe = LinkPipe(phone: try LinkFixture.phone(stores.phone, intent: .connect, seed: 58, versions: [2]),
                            mac: try LinkFixture.mac(stores.mac, pairing: false, seed: 59))
        pipe.start()
        #expect(pipe.macEvents.failure == .unsupportedVersion)
        #expect(pipe.phoneEvents.failure == .unsupportedVersion)
    }

    @Test func badHelloKeysAreRefused() throws {
        var mac = try LinkFixture.mac(InMemoryLinkKeyStore(), pairing: true, seed: 60)
        _ = mac.start()
        let hello = LinkHello(role: .phone, deviceName: "x", appVersion: "1",
                              identityKey: Data(repeating: 1, count: 31), ephemeralKey: Data(repeating: 2, count: 32))
        let json = try JSONEncoder().encode(LinkEnvelope(sequence: 1, message: .hello(hello)))
        #expect(mac.receive(LinkFraming.frame(Data([0]) + json)).failure == .badKey)
    }

    /// Every key and nonce in the handshake must be exactly 32 bytes; a
    /// shorter or longer one ends the connection on the side that receives
    /// it, which shows no code, and nothing is saved on either side.
    @Test(arguments: LinkKeyField.allCases, [0, 31, 33, 64])
    func wrongSizedKeyMaterialIsRefused(field: LinkKeyField, size: Int) throws {
        let phoneStore = InMemoryLinkKeyStore()
        let macStore = InMemoryLinkKeyStore()
        var pipe = LinkPipe(phone: try LinkFixture.phone(phoneStore, intent: .pair, seed: 63),
                            mac: try LinkFixture.mac(macStore, pairing: true, seed: 64))
        let bad = Data(repeating: 0x5A, count: size)
        pipe.intercept = { from, frame in
            guard from == field.sender, let message = LinkFixture.plainMessage(frame) else { return [frame] }
            let rewritten = try? LinkFixture.rewritePlain(frame) { envelope in
                switch (field, message) {
                case (.phoneIdentityKey, .hello(var hello)), (.macIdentityKey, .hello(var hello)):
                    hello.identityKey = bad
                    envelope.message = .hello(hello)
                case (.phoneEphemeralKey, .hello(var hello)), (.macEphemeralKey, .hello(var hello)):
                    hello.ephemeralKey = bad
                    envelope.message = .hello(hello)
                case (.commitment, .pairResponse(var response)):
                    response.commitment = bad
                    envelope.message = .pairResponse(response)
                case (.phoneNonce, .pairNonce), (.macNonce, .pairNonce):
                    envelope.message = .pairNonce(LinkPairNonce(nonce: bad))
                default:
                    break
                }
            }
            return [rewritten ?? frame]
        }
        pipe.start()
        let receiverEvents = field.sender == .phone ? pipe.macEvents : pipe.phoneEvents
        #expect(receiverEvents.failure == field.expectedFailure)
        // The receiver never shows a code. (A Mac whose nonce was tampered on
        // the way has shown its own; the phone never answers it.)
        #expect(receiverEvents.code == nil)
        pipe.confirm(phone: true, mac: true)
        #expect(pipe.phoneEvents.paired == nil && pipe.macEvents.paired == nil)
        #expect(try phoneStore.peers().isEmpty && macStore.peers().isEmpty)
    }

    @Test func allZeroPublicKeyIsRefused() throws {
        var mac = try LinkFixture.mac(InMemoryLinkKeyStore(), pairing: true, seed: 61)
        var phone = try LinkFixture.phone(InMemoryLinkKeyStore(), intent: .pair, seed: 62)
        _ = mac.start()
        var helloFrame = try #require(phone.start().frames.first)
        helloFrame = try LinkFixture.rewritePlain(helloFrame) {
            guard case .hello(var hello) = $0.message else { return }
            hello.ephemeralKey = Data(count: 32)
            $0.message = .hello(hello)
        }
        var events = mac.receive(helloFrame)
        let request = LinkFraming.frame(Data([0]) + (try JSONEncoder().encode(
            LinkEnvelope(sequence: 2, message: .pairRequest(LinkPairRequest(deviceName: "iPhone"))))))
        events += mac.receive(request)
        let nonce = LinkFraming.frame(Data([0]) + (try JSONEncoder().encode(
            LinkEnvelope(sequence: 3, message: .pairNonce(LinkPairNonce(nonce: Data(repeating: 3, count: 32)))))))
        events += mac.receive(nonce)
        #expect(events.failure == .badKey)
        #expect(events.code == nil)
    }

    // MARK: - Oversized frames

    @Test func oversizedHandshakeFrameIsRefusedBeforeItsBody() throws {
        var mac = try LinkFixture.mac(InMemoryLinkKeyStore(), pairing: true, seed: 70)
        _ = mac.start()
        var length = UInt32(LinkFraming.maxHandshakeBodyBytes + 1).bigEndian
        #expect(mac.receive(Data(bytes: &length, count: 4)).failure == .tooLarge)
    }

    @Test func oversizedSealedFrameIsRefusedBeforeItsBody() throws {
        var pipe = try LinkFixture.established()
        var length = UInt32(LinkFraming.maxSealedBodyBytes + 1).bigEndian
        #expect(pipe.phone.receive(Data(bytes: &length, count: 4)).failure == .tooLarge)
    }

    @Test func oversizedMessageIsNotSent() throws {
        var pipe = try LinkFixture.established()
        let huge = LinkMessage.modelChunk(LinkModelChunk(
            transferID: UUID(), offset: 0, data: Data(count: LinkProtocol.maxEnvelopeBytes)))
        #expect(throws: LinkFailure.tooLarge) { _ = try pipe.mac.send(huge) }
    }

    /// A send that fails (here: too large) uses no sequence number, so the
    /// next message still arrives in order on the peer, in both directions.
    @Test func failedSendKeepsTheSequence() throws {
        var pipe = try LinkFixture.established()
        func chunk(_ count: Int, _ byte: UInt8) -> LinkMessage {
            .modelChunk(LinkModelChunk(transferID: UUID(), offset: 0, data: Data(repeating: byte, count: count)))
        }
        #expect(throws: LinkFailure.tooLarge) { _ = try pipe.mac.send(chunk(LinkProtocol.maxEnvelopeBytes, 1)) }
        #expect(throws: LinkFailure.tooLarge) { _ = try pipe.phone.send(chunk(LinkProtocol.maxEnvelopeBytes, 2)) }
        #expect(pipe.mac.isEstablished && pipe.phone.isEstablished)

        let toPhone = chunk(16, 3)
        try pipe.sendFromMac(toPhone)
        #expect(pipe.phoneEvents.received.last == toPhone)
        let toMac = chunk(16, 4)
        try pipe.sendFromPhone(toMac)
        #expect(pipe.macEvents.received.last == toMac)
        #expect(pipe.phoneEvents.failure == nil && pipe.macEvents.failure == nil)
    }

    @Test func emptyFrameIsMalformed() throws {
        var mac = try LinkFixture.mac(InMemoryLinkKeyStore(), pairing: true, seed: 71)
        _ = mac.start()
        #expect(mac.receive(Data([0, 0, 0, 0])).failure == .malformedFrame)
    }

    @Test func largestChunkFitsOneFrame() throws {
        var pipe = try LinkFixture.established()
        let chunk = LinkMessage.modelChunk(LinkModelChunk(
            transferID: UUID(), offset: 0, data: Data(repeating: 0xEE, count: LinkProtocol.maxChunkBytes)))
        try pipe.sendFromMac(chunk)
        #expect(pipe.phoneEvents.received.last == chunk)
    }

    // MARK: - Addresses

    @Test(arguments: [
        "192.168.1.20", "10.0.0.5", "172.16.0.1", "172.31.255.255", "169.254.10.2",
        "fe80::1", "fe80::1c2a:3bff:fe4d:5e6f%en0", "fd12:3456:789a::1", "::ffff:192.168.1.2",
    ])
    func localAddressesAreAllowed(host: String) {
        #expect(LinkAddressPolicy().isAllowed(host: host))
    }

    @Test(arguments: [
        "8.8.8.8", "172.32.0.1", "172.15.255.255", "100.64.0.1", "192.169.0.1", "2001:4860:4860::8888",
        "::ffff:8.8.8.8", "127.0.0.1", "::1", "example.local", "", "999.1.1.1", "0.0.0.0", "::",
    ])
    func nonLocalAddressesAreRefused(host: String) {
        #expect(!LinkAddressPolicy().isAllowed(host: host))
    }

    @Test func loopbackOnlyWhenAllowed() {
        let policy = LinkAddressPolicy(allowLoopback: true)
        #expect(policy.isAllowed(host: "127.0.0.1"))
        #expect(policy.isAllowed(host: "::1"))
        #expect(!policy.isAllowed(host: "8.8.8.8"))
    }

    // MARK: - Key store

    @Test func identityIsMadeOnceAndKept() throws {
        let store = InMemoryLinkKeyStore()
        let first = try store.identity(using: SeededLinkRandom(seed: 80))
        let second = try store.identity(using: SeededLinkRandom(seed: 81))
        #expect(first == second)
        #expect(first.publicKey.count == 32)
    }

    @Test func addIdentityKeepsTheFirstOne() throws {
        let store = InMemoryLinkKeyStore()
        let first = try LinkKeyPair.generate(using: SeededLinkRandom(seed: 82))
        let second = try LinkKeyPair.generate(using: SeededLinkRandom(seed: 83))
        #expect(try store.addIdentity(first) == first)
        #expect(try store.addIdentity(second) == first)
        #expect(try store.loadIdentity() == first)
    }

    @Test func peerNamesAreCleaned() {
        #expect(LinkPeer.cleanName("  Zoe's iPhone \n", role: .phone) == "Zoe's iPhone")
        // Bidi override and zero-width characters are dropped.
        #expect(LinkPeer.cleanName("Mac\u{202E}koob\u{200B}", role: .mac) == "Mackoob")
        #expect(LinkPeer.cleanName(String(repeating: "x", count: 500), role: .mac).count == LinkPeer.maxNameLength)
        #expect(LinkPeer.cleanName("\u{0007}\u{200D}", role: .mac) == "Mac")
        #expect(LinkPeer.cleanName("", role: .phone) == "iPhone")
    }

    @Test func pairingSavesTheCleanedName() throws {
        let macStore = InMemoryLinkKeyStore()
        let phone = try LinkConnection(
            configuration: .init(role: .phone, deviceName: "Ph\u{202E}one\u{0000}", appVersion: "1", intent: .pair),
            store: InMemoryLinkKeyStore(), random: SeededLinkRandom(seed: 84))
        var pipe = LinkPipe(phone: phone, mac: try LinkFixture.mac(macStore, pairing: true, seed: 85))
        pipe.start()
        #expect(pipe.macEvents.contains { if case .pairingCode(_, "Phone") = $0 { true } else { false } })
        pipe.confirm(phone: true, mac: true)
        #expect(try macStore.peers().first?.name == "Phone")
    }

    @Test func printingAPeerNeverShowsItsSecret() throws {
        let secret = Data(repeating: 0xA5, count: 32)
        let peer = LinkPeer(role: .mac, name: "MacBook Pro", identityKey: Data(repeating: 4, count: 32),
                            pairingSecret: secret, pairedAt: Date(timeIntervalSince1970: 1))
        let event = LinkConnection.Event.paired(peer)
        for text in [String(describing: peer), String(reflecting: peer), String(describing: event),
                     String(reflecting: event), "\(peer)"] {
            #expect(!text.contains("165"))  // 0xA5
            #expect(!text.contains("a5a5"))
            #expect(!text.lowercased().contains("pairingsecret"))
        }
        var dumped = ""
        dump(event, to: &dumped)
        #expect(!dumped.contains("165") && !dumped.lowercased().contains("pairingsecret"))
    }

    /// The real Keychain, only when asked (it may prompt outside a signed app).
    @Test(.enabled(if: ProcessInfo.processInfo.environment["MFERENCE_KEYCHAIN_LIVE"] == "1"))
    func keychainStoreRoundTrips() throws {
        let store = KeychainLinkKeyStore(service: "ai.mference.link.test.\(UUID().uuidString)")
        defer {
            for peer in (try? store.peers()) ?? [] { try? store.deletePeer(id: peer.id) }
        }
        #expect(try store.loadIdentity() == nil)
        let identity = try store.identity()
        #expect(try store.loadIdentity() == identity)
        let peer = LinkPeer(role: .mac, name: "MacBook Pro", identityKey: Data(repeating: 4, count: 32),
                            pairingSecret: Data(repeating: 5, count: 32), pairedAt: Date(timeIntervalSince1970: 1))
        try store.savePeer(peer)
        #expect(try store.peer(id: peer.id) == peer)
        #expect(try store.peers() == [peer])
        try store.deletePeer(id: peer.id)
        #expect(try store.peer(id: peer.id) == nil)
    }
}

/// The handshake's 32-byte values, by the side that sends them.
enum LinkKeyField: String, CaseIterable, Sendable {
    case phoneIdentityKey, phoneEphemeralKey, macIdentityKey, macEphemeralKey, commitment, phoneNonce, macNonce

    var sender: LinkRole {
        switch self {
        case .phoneIdentityKey, .phoneEphemeralKey, .phoneNonce: .phone
        case .macIdentityKey, .macEphemeralKey, .commitment, .macNonce: .mac
        }
    }

    var expectedFailure: LinkFailure {
        switch self {
        case .phoneIdentityKey, .phoneEphemeralKey, .macIdentityKey, .macEphemeralKey: .badKey
        case .commitment, .phoneNonce, .macNonce: .malformedFrame
        }
    }
}

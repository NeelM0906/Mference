import CryptoKit
import Foundation

// Frames on the wire. Each frame is a 4-byte big-endian body length, then the
// body:
//
//   plain   0x00 | JSON LinkEnvelope                       (handshake only)
//   sealed  0x01 | sequence (8 bytes, big-endian) | ChaCha20-Poly1305
//           ciphertext of the JSON LinkEnvelope | 16-byte tag
//
// A sealed frame's nonce is 4 zero bytes and the sequence; its additional
// data is the kind byte and the sequence, so neither can be changed without
// the tag failing. Each direction has its own key, so a nonce never repeats
// under one key. The sequence starts at 1 for each direction and counts every
// frame, plain and sealed; the envelope inside must carry the same number.

enum LinkFrameKind: UInt8 {
    case plain = 0
    case sealed = 1
}

public enum LinkFraming {
    /// Bytes of the length prefix.
    public static let lengthBytes = 4
    /// Largest body before the handshake is done. An unauthenticated peer can
    /// make this side hold at most this much.
    public static let maxHandshakeBodyBytes = 64 * 1024
    /// Largest sealed body: kind, sequence, the largest envelope, the tag.
    public static let maxSealedBodyBytes = 1 + 8 + LinkProtocol.maxEnvelopeBytes + 16

    static func frame(_ body: Data) -> Data {
        var length = UInt32(body.count).bigEndian
        return Data(bytes: &length, count: lengthBytes) + body
    }
}

/// Collects bytes as they arrive and cuts them into frame bodies. It refuses
/// a frame whose announced length is over the limit before it holds the
/// body, so it never keeps more than one partial frame within the limit plus
/// the last chunk the transport read (the transport reads bounded chunks).
struct LinkFrameReader {
    private var buffer = Data()

    mutating func append(_ bytes: Data) {
        buffer.append(bytes)
    }

    /// The next complete body, or nil while it is still arriving.
    mutating func nextBody(maxBodyBytes: Int) throws -> Data? {
        guard buffer.count >= LinkFraming.lengthBytes else { return nil }
        let start = buffer.startIndex
        let length = buffer[start..<start + LinkFraming.lengthBytes].reduce(0) { $0 << 8 | Int($1) }
        guard length > 0 else { throw LinkFailure.malformedFrame }
        guard length <= maxBodyBytes else { throw LinkFailure.tooLarge }
        let total = LinkFraming.lengthBytes + length
        guard buffer.count >= total else { return nil }
        let body = Data(buffer[(start + LinkFraming.lengthBytes)..<(start + total)])
        buffer.removeSubrange(start..<(start + total))
        return body
    }
}

/// One direction's sequence counter and, once the handshake is done, its key.
struct LinkDirection {
    /// The sequence the next frame in this direction must carry.
    private(set) var next: UInt64 = 1
    var key: SymmetricKey?

    /// Checks `sequence` is exactly the next one and moves on.
    mutating func accept(_ sequence: UInt64) throws {
        if sequence < next { throw LinkFailure.replayedFrame }
        if sequence > next { throw LinkFailure.outOfOrderFrame }
        guard next < UInt64.max else { throw LinkFailure.closed }
        next += 1
    }

    /// Takes the next sequence number for an outgoing frame.
    mutating func take() throws -> UInt64 {
        guard next < UInt64.max else { throw LinkFailure.closed }
        defer { next += 1 }
        return next
    }
}

enum LinkFrameCodec {
    static let encoder = JSONEncoder()
    static let decoder = JSONDecoder()

    static func header(_ kind: LinkFrameKind, _ sequence: UInt64) -> Data {
        var seq = sequence.bigEndian
        return Data([kind.rawValue]) + Data(bytes: &seq, count: 8)
    }

    static func nonce(_ sequence: UInt64) throws -> ChaChaPoly.Nonce {
        var seq = sequence.bigEndian
        return try ChaChaPoly.Nonce(data: Data(count: 4) + Data(bytes: &seq, count: 8))
    }

    static func encode(_ envelope: LinkEnvelope) throws -> Data {
        let json = try encoder.encode(envelope)
        guard json.count <= LinkProtocol.maxEnvelopeBytes else { throw LinkFailure.tooLarge }
        return json
    }

    static func decode(_ json: Data) throws -> LinkEnvelope {
        guard json.count <= LinkProtocol.maxEnvelopeBytes else { throw LinkFailure.tooLarge }
        do {
            return try decoder.decode(LinkEnvelope.self, from: json)
        } catch {
            throw LinkFailure.malformedFrame
        }
    }

    /// A plain body: the kind byte and the JSON.
    static func plainBody(_ json: Data) -> Data {
        Data([LinkFrameKind.plain.rawValue]) + json
    }

    /// A sealed body for `json` at `sequence`.
    static func sealedBody(_ json: Data, sequence: UInt64, key: SymmetricKey) throws -> Data {
        let header = header(.sealed, sequence)
        let box = try ChaChaPoly.seal(json, using: key, nonce: nonce(sequence), authenticating: header)
        return header + box.ciphertext + box.tag
    }

    /// The kind of a body, or `malformedFrame`.
    static func kind(of body: Data) throws -> LinkFrameKind {
        guard let first = body.first, let kind = LinkFrameKind(rawValue: first) else {
            throw LinkFailure.malformedFrame
        }
        return kind
    }

    /// The sequence in a sealed body's header (not yet authenticated).
    static func sealedSequence(_ body: Data) throws -> UInt64 {
        guard body.count >= 1 + 8 + 16 else { throw LinkFailure.malformedFrame }
        let start = body.startIndex + 1
        return body[start..<start + 8].reduce(UInt64(0)) { $0 << 8 | UInt64($1) }
    }

    /// Opens a sealed body; `authenticationFailed` when anything in it changed
    /// or it was sealed with another key.
    static func open(_ body: Data, sequence: UInt64, key: SymmetricKey) throws -> Data {
        let start = body.startIndex
        let header = Data(body[start..<start + 9])
        let ciphertext = Data(body[(start + 9)..<(body.endIndex - 16)])
        let tag = Data(body[(body.endIndex - 16)...])
        do {
            let box = try ChaChaPoly.SealedBox(nonce: nonce(sequence), ciphertext: ciphertext, tag: tag)
            return try ChaChaPoly.open(box, using: key, authenticating: header)
        } catch {
            throw LinkFailure.authenticationFailed
        }
    }
}

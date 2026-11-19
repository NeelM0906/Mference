import Foundation

// The iPhone to Mac link, protocol v1: the messages only (mobile program spec
// v1.0, section 5.3). Built by both apps: the Mac app's MferenceAppCore
// (listener, pairing, job runner, search proxy, model transfer server) and the
// phone's MferenceMobileCore (browse, pair, web via Mac, hand-off, model copy).
//
// This file is a contract: changing a case, a field or the wire shape is a
// "contract: …" PR the mobile coordinator approves, and it bumps
// `LinkProtocol.version` when an old peer could misread the new shape.
// Discovery, key agreement, encryption and framing live with the link
// implementation (team mobile-link), not here: every `LinkEnvelope` travels
// authenticated and encrypted, and its `sequence` makes a replay fail.

/// Protocol constants shared by both sides.
public enum LinkProtocol {
    /// The wire version both peers announce in `LinkHello`. A peer refuses a
    /// version it does not support with `LinkError.Code.unsupportedVersion`.
    public static let version = 1

    /// Bonjour service type the Mac advertises and the phone browses
    /// (`NSBonjourServices` on the phone lists it).
    public static let bonjourServiceType = "_mference._tcp"

    // Size limits. A receiver checks a frame's length before decoding it and
    // answers anything larger with `LinkError.Code.tooLarge`; both apps keep
    // memory bounded by these, never by what a peer claims.

    /// Largest encoded `LinkEnvelope` (after decryption) either side accepts.
    public static let maxEnvelopeBytes = 16 * 1_048_576
    /// Largest `LinkModelChunk.data` and `LinkModelChunkRequest.length`.
    public static let maxChunkBytes = 4 * 1_048_576
    /// Largest total of `LinkJobSubmit.attachments` data.
    public static let maxAttachmentBytes = 8 * 1_048_576
    /// Largest `LinkWebSearchRequest.maxResults`.
    public static let maxSearchResults = 20
}

/// One message on the link. `sequence` starts at 1 for each direction of a
/// connection and grows by one per envelope; a receiver drops the connection
/// on a sequence it has already seen or one that skips ahead.
/// The link layer checks `version` against the version the hello exchange
/// settled on and drops the connection on a mismatch.
public struct LinkEnvelope: Codable, Equatable, Sendable {
    public var version: Int
    public var sequence: UInt64
    public var message: LinkMessage

    public init(version: Int = LinkProtocol.version, sequence: UInt64, message: LinkMessage) {
        self.version = version
        self.sequence = sequence
        self.message = message
    }
}

/// Every message either side can send. Wire shape:
/// `{"type": "<case name>", "body": {…}}` (`body` omitted for `forget`).
public enum LinkMessage: Equatable, Sendable {
    // Session
    case hello(LinkHello)
    case capabilities(LinkCapabilities)
    case error(LinkError)

    // Pairing (X25519; both screens show the code derived from the secret)
    case pairRequest(LinkPairRequest)
    case pairResponse(LinkPairResponse)
    case pairConfirm(LinkPairConfirm)
    case forget

    // Web via Mac (the phone runs the tool loop; only these calls cross)
    case webSearch(LinkWebSearchRequest)
    case webSearchResults(LinkWebSearchResults)
    case readPage(LinkReadPageRequest)
    case pageText(LinkPageText)

    // Hand-off jobs (run on the Mac through EngineService)
    case jobSubmit(LinkJobSubmit)
    case jobEvent(LinkJobEvent)
    case jobCancel(LinkJobCancel)

    // Model transfer (Quick copied from the Mac; resumable, SHA-256 checked)
    case modelOffer(LinkModelOffer)
    case modelChunkRequest(LinkModelChunkRequest)
    case modelChunk(LinkModelChunk)

    /// The `type` string on the wire.
    public var type: String {
        switch self {
        case .hello: "hello"
        case .capabilities: "capabilities"
        case .error: "error"
        case .pairRequest: "pairRequest"
        case .pairResponse: "pairResponse"
        case .pairConfirm: "pairConfirm"
        case .forget: "forget"
        case .webSearch: "webSearch"
        case .webSearchResults: "webSearchResults"
        case .readPage: "readPage"
        case .pageText: "pageText"
        case .jobSubmit: "jobSubmit"
        case .jobEvent: "jobEvent"
        case .jobCancel: "jobCancel"
        case .modelOffer: "modelOffer"
        case .modelChunkRequest: "modelChunkRequest"
        case .modelChunk: "modelChunk"
        }
    }
}

extension LinkMessage: Codable {
    private enum CodingKeys: String, CodingKey { case type, body }

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let type = try c.decode(String.self, forKey: .type)
        switch type {
        case "hello": self = .hello(try c.decode(LinkHello.self, forKey: .body))
        case "capabilities": self = .capabilities(try c.decode(LinkCapabilities.self, forKey: .body))
        case "error": self = .error(try c.decode(LinkError.self, forKey: .body))
        case "pairRequest": self = .pairRequest(try c.decode(LinkPairRequest.self, forKey: .body))
        case "pairResponse": self = .pairResponse(try c.decode(LinkPairResponse.self, forKey: .body))
        case "pairConfirm": self = .pairConfirm(try c.decode(LinkPairConfirm.self, forKey: .body))
        case "forget": self = .forget
        case "webSearch": self = .webSearch(try c.decode(LinkWebSearchRequest.self, forKey: .body))
        case "webSearchResults": self = .webSearchResults(try c.decode(LinkWebSearchResults.self, forKey: .body))
        case "readPage": self = .readPage(try c.decode(LinkReadPageRequest.self, forKey: .body))
        case "pageText": self = .pageText(try c.decode(LinkPageText.self, forKey: .body))
        case "jobSubmit": self = .jobSubmit(try c.decode(LinkJobSubmit.self, forKey: .body))
        case "jobEvent": self = .jobEvent(try c.decode(LinkJobEvent.self, forKey: .body))
        case "jobCancel": self = .jobCancel(try c.decode(LinkJobCancel.self, forKey: .body))
        case "modelOffer": self = .modelOffer(try c.decode(LinkModelOffer.self, forKey: .body))
        case "modelChunkRequest": self = .modelChunkRequest(try c.decode(LinkModelChunkRequest.self, forKey: .body))
        case "modelChunk": self = .modelChunk(try c.decode(LinkModelChunk.self, forKey: .body))
        default:
            throw DecodingError.dataCorruptedError(
                forKey: .type, in: c, debugDescription: "Unknown link message type \(type)")
        }
    }

    public func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(type, forKey: .type)
        switch self {
        case .hello(let body): try c.encode(body, forKey: .body)
        case .capabilities(let body): try c.encode(body, forKey: .body)
        case .error(let body): try c.encode(body, forKey: .body)
        case .pairRequest(let body): try c.encode(body, forKey: .body)
        case .pairResponse(let body): try c.encode(body, forKey: .body)
        case .pairConfirm(let body): try c.encode(body, forKey: .body)
        case .forget: break
        case .webSearch(let body): try c.encode(body, forKey: .body)
        case .webSearchResults(let body): try c.encode(body, forKey: .body)
        case .readPage(let body): try c.encode(body, forKey: .body)
        case .pageText(let body): try c.encode(body, forKey: .body)
        case .jobSubmit(let body): try c.encode(body, forKey: .body)
        case .jobEvent(let body): try c.encode(body, forKey: .body)
        case .jobCancel(let body): try c.encode(body, forKey: .body)
        case .modelOffer(let body): try c.encode(body, forKey: .body)
        case .modelChunkRequest(let body): try c.encode(body, forKey: .body)
        case .modelChunk(let body): try c.encode(body, forKey: .body)
        }
    }
}

// MARK: - Session

/// Which app is speaking.
public enum LinkRole: String, Codable, Sendable {
    case mac, phone
}

/// First message from each side after the connection opens.
public struct LinkHello: Codable, Equatable, Sendable {
    public var role: LinkRole
    /// The name the other side shows ("MacBook Pro", "iPhone").
    public var deviceName: String
    /// Every protocol version this peer speaks; the highest common one wins.
    public var protocolVersions: [Int]
    public var appVersion: String

    public init(role: LinkRole, deviceName: String,
                protocolVersions: [Int] = [LinkProtocol.version], appVersion: String) {
        self.role = role
        self.deviceName = deviceName
        self.protocolVersions = protocolVersions
        self.appVersion = appVersion
    }
}

/// A model the Mac can run for a hand-off job.
public struct LinkModel: Codable, Equatable, Sendable, Identifiable {
    /// The Mac engine catalog's model ID.
    public var id: String
    /// What the phone shows ("Deep 27B").
    public var title: String
    /// Ready to run now (installed and fits).
    public var isReady: Bool
    /// The Mac's default for big jobs (Deep when installed).
    public var isDefault: Bool

    public init(id: String, title: String, isReady: Bool, isDefault: Bool) {
        self.id = id
        self.title = title
        self.isReady = isReady
        self.isDefault = isDefault
    }
}

/// The Mac's answer to a paired phone's hello.
public struct LinkCapabilities: Codable, Equatable, Sendable {
    public var macName: String
    public var models: [LinkModel]
    /// SearXNG on the Mac is running and web search may be proxied.
    public var webSearch: Bool
    /// The Mac has the pinned Quick GGUF and can send it.
    public var modelTransfer: Bool

    public init(macName: String, models: [LinkModel], webSearch: Bool, modelTransfer: Bool) {
        self.macName = macName
        self.models = models
        self.webSearch = webSearch
        self.modelTransfer = modelTransfer
    }
}

/// A refusal or failure, either direction. `requestID` ties it to a request
/// when there is one.
public struct LinkError: Codable, Equatable, Sendable, Error {
    public enum Code: String, Codable, Sendable {
        case unsupportedVersion
        case notPaired
        case pairingRejected
        case busy
        case modelUnavailable
        case webSearchOff
        case notFound
        case cancelled
        case tooLarge
        case `internal`
    }

    public var code: Code
    /// Short, plain, shown to the person as is.
    public var message: String
    public var requestID: UUID?

    public init(code: Code, message: String, requestID: UUID? = nil) {
        self.code = code
        self.message = message
        self.requestID = requestID
    }
}

// MARK: - Pairing

/// Phone to Mac: start pairing with this ephemeral X25519 public key.
public struct LinkPairRequest: Codable, Equatable, Sendable {
    public var deviceName: String
    /// Raw 32-byte X25519 public key (base64 on the wire).
    public var publicKey: Data

    public init(deviceName: String, publicKey: Data) {
        self.deviceName = deviceName
        self.publicKey = publicKey
    }
}

/// Mac to phone: its X25519 public key. Both sides now derive the shared
/// secret and show the same short code.
public struct LinkPairResponse: Codable, Equatable, Sendable {
    public var macName: String
    public var publicKey: Data

    public init(macName: String, publicKey: Data) {
        self.macName = macName
        self.publicKey = publicKey
    }
}

/// Either side: the person confirmed (or rejected) that both screens show the
/// same code. Pairing completes only when both sides sent `accepted: true`.
public struct LinkPairConfirm: Codable, Equatable, Sendable {
    public var accepted: Bool

    public init(accepted: Bool) {
        self.accepted = accepted
    }
}

// MARK: - Web via Mac

/// Phone to Mac: run one search through the Mac's SearXNG.
public struct LinkWebSearchRequest: Codable, Equatable, Sendable {
    public var requestID: UUID
    public var query: String
    public var maxResults: Int

    public init(requestID: UUID = UUID(), query: String, maxResults: Int = 8) {
        self.requestID = requestID
        self.query = query
        self.maxResults = maxResults
    }
}

/// One search hit, in the order SearXNG ranked it.
public struct LinkSearchResult: Codable, Equatable, Sendable {
    public var title: String
    public var url: URL
    public var snippet: String

    public init(title: String, url: URL, snippet: String) {
        self.title = title
        self.url = url
        self.snippet = snippet
    }
}

public struct LinkWebSearchResults: Codable, Equatable, Sendable {
    public var requestID: UUID
    public var results: [LinkSearchResult]

    public init(requestID: UUID, results: [LinkSearchResult]) {
        self.requestID = requestID
        self.results = results
    }
}

/// Phone to Mac: fetch one page through the Mac's page reader. The Mac fetches
/// only `http`/`https` URLs on public hosts: it refuses other schemes and
/// loopback, link-local and private-network targets with a plain `LinkError`.
public struct LinkReadPageRequest: Codable, Equatable, Sendable {
    public var requestID: UUID
    public var url: URL

    public init(requestID: UUID = UUID(), url: URL) {
        self.requestID = requestID
        self.url = url
    }
}

/// Reader text of a page (already trimmed by the Mac's reader).
public struct LinkPageText: Codable, Equatable, Sendable {
    public var requestID: UUID
    public var url: URL
    public var title: String
    public var text: String

    public init(requestID: UUID, url: URL, title: String, text: String) {
        self.requestID = requestID
        self.url = url
        self.title = title
        self.text = text
    }
}

// MARK: - Hand-off jobs

public enum LinkChatRole: String, Codable, Sendable {
    case system, user, assistant
}

public struct LinkChatMessage: Codable, Equatable, Sendable {
    public var role: LinkChatRole
    public var text: String

    public init(role: LinkChatRole, text: String) {
        self.role = role
        self.text = text
    }
}

/// A file the person attached to a hand-off job.
public struct LinkAttachment: Codable, Equatable, Sendable {
    public var name: String
    /// A UTI or MIME type ("public.pdf", "application/pdf").
    public var contentType: String
    public var data: Data

    public init(name: String, contentType: String, data: Data) {
        self.name = name
        self.contentType = contentType
        self.data = data
    }
}

/// Phone to Mac: run this conversation on a Mac model.
public struct LinkJobSubmit: Codable, Equatable, Sendable {
    public var jobID: UUID
    /// A `LinkModel.id` from the Mac's capabilities.
    public var modelID: String
    public var messages: [LinkChatMessage]
    public var attachments: [LinkAttachment]

    public init(jobID: UUID = UUID(), modelID: String, messages: [LinkChatMessage],
                attachments: [LinkAttachment] = []) {
        self.jobID = jobID
        self.modelID = modelID
        self.messages = messages
        self.attachments = attachments
    }
}

/// A finished job's answer.
public struct LinkJobResult: Codable, Equatable, Sendable {
    public var text: String
    /// What ran it, as the phone labels the answer ("Deep 27B").
    public var modelTitle: String
    public var promptTokens: Int
    public var completionTokens: Int

    public init(text: String, modelTitle: String, promptTokens: Int, completionTokens: Int) {
        self.text = text
        self.modelTitle = modelTitle
        self.promptTokens = promptTokens
        self.completionTokens = completionTokens
    }
}

/// Mac to phone: progress of a running job, streamed in order.
public struct LinkJobEvent: Codable, Equatable, Sendable {
    public enum Kind: Codable, Equatable, Sendable {
        /// The job started on the Mac (it may have waited behind another).
        case started
        /// The next piece of answer text.
        case token(text: String)
        /// A step line ("Read 40 PDFs").
        case step(text: String)
        case finished(result: LinkJobResult)
        case failed(error: LinkError)
        case cancelled
    }

    public var jobID: UUID
    public var kind: Kind

    public init(jobID: UUID, kind: Kind) {
        self.jobID = jobID
        self.kind = kind
    }
}

public struct LinkJobCancel: Codable, Equatable, Sendable {
    public var jobID: UUID

    public init(jobID: UUID) {
        self.jobID = jobID
    }
}

// MARK: - Model transfer

/// Mac to phone: the pinned file it can send.
public struct LinkModelOffer: Codable, Equatable, Sendable {
    public var transferID: UUID
    /// The engine catalog's model ID ("quick").
    public var modelID: String
    /// Display only. The phone never builds a path from it: it stores the file
    /// where its model storage puts `modelID`, and refuses an offer whose
    /// `modelID`, `byteCount` and `sha256` do not match the pinned Quick file.
    public var fileName: String
    /// Positive; the phone checks free space before the first chunk.
    public var byteCount: Int64
    /// 64 lower-case hex characters: SHA-256 of the whole file, checked at the end.
    public var sha256: String

    public init(transferID: UUID = UUID(), modelID: String, fileName: String,
                byteCount: Int64, sha256: String) {
        self.transferID = transferID
        self.modelID = modelID
        self.fileName = fileName
        self.byteCount = byteCount
        self.sha256 = sha256
    }
}

/// Phone to Mac: send `length` bytes starting at `offset` (resume = ask from
/// the first byte not yet on disk). `offset` is in `0..<byteCount`; `length`
/// is in `1...LinkProtocol.maxChunkBytes`; the Mac refuses anything else.
public struct LinkModelChunkRequest: Codable, Equatable, Sendable {
    public var transferID: UUID
    public var offset: Int64
    public var length: Int

    public init(transferID: UUID, offset: Int64, length: Int) {
        self.transferID = transferID
        self.offset = offset
        self.length = length
    }
}

public struct LinkModelChunk: Codable, Equatable, Sendable {
    public var transferID: UUID
    public var offset: Int64
    public var data: Data

    public init(transferID: UUID, offset: Int64, data: Data) {
        self.transferID = transferID
        self.offset = offset
        self.data = data
    }
}

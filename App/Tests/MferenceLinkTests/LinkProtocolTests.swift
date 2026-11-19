import Foundation
import Testing
@testable import MferenceLink

@Suite("Link protocol v1")
struct LinkProtocolTests {
    private static let id = UUID(uuidString: "00000000-0000-0000-0000-0000000000A1")!

    /// One sample of every message, so a new case without a round trip fails
    /// `everyTypeHasASample`.
    static let samples: [LinkMessage] = [
        .hello(LinkHello(role: .phone, deviceName: "iPhone", appVersion: "0.1")),
        .capabilities(LinkCapabilities(
            macName: "MacBook Pro",
            models: [LinkModel(id: "deep", title: "Deep 27B", isReady: true, isDefault: true)],
            webSearch: true, modelTransfer: true)),
        .error(LinkError(code: .notPaired, message: "Pair this iPhone first.", requestID: id)),
        .pairRequest(LinkPairRequest(deviceName: "iPhone", publicKey: Data(repeating: 7, count: 32))),
        .pairResponse(LinkPairResponse(macName: "MacBook Pro", publicKey: Data(repeating: 9, count: 32))),
        .pairConfirm(LinkPairConfirm(accepted: true)),
        .forget,
        .webSearch(LinkWebSearchRequest(requestID: id, query: "swift testing", maxResults: 5)),
        .webSearchResults(LinkWebSearchResults(requestID: id, results: [
            LinkSearchResult(title: "Swift", url: URL(string: "https://www.swift.org")!, snippet: "A language"),
        ])),
        .readPage(LinkReadPageRequest(requestID: id, url: URL(string: "https://www.swift.org")!)),
        .pageText(LinkPageText(requestID: id, url: URL(string: "https://www.swift.org")!,
                               title: "Swift", text: "Body")),
        .jobSubmit(LinkJobSubmit(jobID: id, modelID: "deep",
                                 messages: [LinkChatMessage(role: .user, text: "Make a table")],
                                 attachments: [LinkAttachment(name: "a.pdf", contentType: "com.adobe.pdf",
                                                              data: Data([1, 2, 3]))])),
        .jobEvent(LinkJobEvent(jobID: id, kind: .started)),
        .jobEvent(LinkJobEvent(jobID: id, kind: .token(text: "Hel"))),
        .jobEvent(LinkJobEvent(jobID: id, kind: .step(text: "Read 40 PDFs"))),
        .jobEvent(LinkJobEvent(jobID: id, kind: .finished(result: LinkJobResult(
            text: "| a |", modelTitle: "Deep 27B", promptTokens: 10, completionTokens: 3)))),
        .jobEvent(LinkJobEvent(jobID: id, kind: .failed(error: LinkError(code: .busy, message: "Busy")))),
        .jobEvent(LinkJobEvent(jobID: id, kind: .cancelled)),
        .jobCancel(LinkJobCancel(jobID: id)),
        .modelOffer(LinkModelOffer(transferID: id, modelID: "quick", fileName: "Qwen3.5-4B-UD-Q4_K_XL.gguf",
                                   byteCount: 2_900_000_000, sha256: String(repeating: "a", count: 64))),
        .modelChunkRequest(LinkModelChunkRequest(transferID: id, offset: 4_194_304, length: 4_194_304)),
        .modelChunk(LinkModelChunk(transferID: id, offset: 0, data: Data([0xDE, 0xAD]))),
    ]

    @Test func versionAndServiceAreFixed() {
        #expect(LinkProtocol.version == 1)
        #expect(LinkProtocol.bonjourServiceType == "_mference._tcp")
        #expect(LinkHello(role: .mac, deviceName: "Mac", appVersion: "1").protocolVersions == [1])
    }

    @Test(arguments: samples)
    func everyMessageRoundTrips(_ message: LinkMessage) throws {
        let envelope = LinkEnvelope(sequence: 42, message: message)
        let data = try JSONEncoder().encode(envelope)
        let decoded = try JSONDecoder().decode(LinkEnvelope.self, from: data)
        #expect(decoded == envelope)
        #expect(decoded.version == LinkProtocol.version)
    }

    /// Every wire type of v1. Adding a case means adding it here and a sample
    /// above; the switch in `LinkMessage.type` fails to compile without it.
    static let allTypes: Set<String> = [
        "hello", "capabilities", "error", "pairRequest", "pairResponse", "pairConfirm", "forget",
        "webSearch", "webSearchResults", "readPage", "pageText", "jobSubmit", "jobEvent", "jobCancel",
        "modelOffer", "modelChunkRequest", "modelChunk",
    ]

    @Test func everyTypeHasASample() {
        #expect(Set(Self.samples.map(\.type)) == Self.allTypes)
    }

    /// `Data` travels as base64 (4/3 larger); a full chunk or a full set of
    /// attachments plus 1 MiB of headroom must still fit one envelope.
    @Test func limitsStayInsideTheEnvelope() {
        let headroom = 1_048_576
        #expect(LinkProtocol.maxChunkBytes * 4 / 3 + headroom <= LinkProtocol.maxEnvelopeBytes)
        #expect(LinkProtocol.maxAttachmentBytes * 4 / 3 + headroom <= LinkProtocol.maxEnvelopeBytes)
        #expect(LinkProtocol.maxSearchResults >= LinkWebSearchRequest(query: "q").maxResults)
    }

    @Test func wireShapeIsTypeAndBody() throws {
        let data = try JSONEncoder().encode(LinkMessage.jobCancel(LinkJobCancel(jobID: Self.id)))
        let object = try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
        #expect(object["type"] as? String == "jobCancel")
        let body = try #require(object["body"] as? [String: Any])
        #expect(body["jobID"] as? String == Self.id.uuidString)

        let forget = try JSONEncoder().encode(LinkMessage.forget)
        let forgetObject = try #require(try JSONSerialization.jsonObject(with: forget) as? [String: Any])
        #expect(forgetObject.keys.sorted() == ["type"])
    }

    @Test func unknownTypeFailsToDecode() {
        let data = Data(#"{"type":"teleport","body":{}}"#.utf8)
        #expect(throws: DecodingError.self) {
            try JSONDecoder().decode(LinkMessage.self, from: data)
        }
    }
}

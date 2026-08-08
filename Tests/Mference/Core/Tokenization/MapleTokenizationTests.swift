import Foundation
import Testing
@testable import Mference

@Suite("Maple tokenization")
struct MapleTokenizationTests {
    private typealias Message = MFTokenizer.Message

    private static func fixtureFolder() throws -> URL {
        try #require(Bundle.module.url(
            forResource: "MapleChatMLTokenizer",
            withExtension: nil,
            subdirectory: "Fixtures"))
    }

    private static func modelDirectory(withTokenizer: Bool) throws -> URL {
        let model = try ManifestReaderTests.writeToyManifest(
            archOverrides: ["family": ModelFamily.maple.rawValue],
            config: .maplePreview).0
        if withTokenizer {
            try FileManager.default.copyItem(
                at: fixtureFolder(),
                to: model.appendingPathComponent("tokenizer", isDirectory: true))
        }
        return model
    }

    private static func loadMapleTokenizer() async throws -> (MFTokenizer, URL) {
        let model = try modelDirectory(withTokenizer: true)
        do {
            return (try await MFTokenizer.load(
                forModelDirectory: model,
                environment: [:]), model)
        } catch {
            try? FileManager.default.removeItem(at: model)
            throw error
        }
    }

    @Test("Manifest family selects exact Maple framing and token IDs")
    func exactPromptAndContinuation() async throws {
        let (tokenizer, model) = try await Self.loadMapleTokenizer()
        defer { try? FileManager.default.removeItem(at: model) }

        #expect(tokenizer.dialect == .chatml)
        #expect(tokenizer.bosID == 151_643)
        #expect(tokenizer.eosID == 151_645)
        #expect(tokenizer.padID == 151_643)
        #expect(tokenizer.endOfTurnID == 151_645)
        #expect(tokenizer.toolCallStartID == 151_657)
        #expect(tokenizer.toolCallEndID == 151_658)
        #expect(tokenizer.toolResponseID == 151_665)
        #expect(tokenizer.toolResponseEndID == 151_666)
        #expect(tokenizer.thinkStartID == 151_667)
        #expect(tokenizer.thinkEndID == 151_668)
        #expect(tokenizer.stopTokenIDs == [151_643, 151_645])
        #expect(tokenizer.vocabSize == 151_936)
        #expect(tokenizer.generationPromptStartsInThinking)

        let prompt = try tokenizer.applyChatTemplate([
            Message(role: .user, content: " Next \n"),
        ])
        let expectedPrompt = "<|im_start|>user\n Next \n<|im_end|>\n"
            + "<|im_start|>assistant\n<think>\n"
        let expectedPromptIDs: [Int32] = [
            151_644, 872, 198, 9_295, 715, 151_645, 198,
            151_644, 77_091, 198, 151_667, 198,
        ]
        #expect(prompt == expectedPrompt)
        #expect(tokenizer.encode(prompt, addBOS: false) == expectedPromptIDs)
        #expect(tokenizer.decode(expectedPromptIDs, skipSpecialTokens: false) == prompt)
        #expect(!prompt.contains("</think>"))

        let text = " Next \n"
        let textIDs: [Int32] = [9_295, 715]
        #expect(tokenizer.encode(text, addBOS: false) == textIDs)
        #expect(tokenizer.encode(text, addBOS: true) == textIDs)

        let continuation = tokenizer.encodeTextContinuation(userContent: text)
        let expectedContinuationIDs: [Int32] = [
            151_645, 198, 151_644, 872, 198, 9_295, 715,
            151_645, 198, 151_644, 77_091, 198, 151_667, 198,
        ]
        let expectedContinuation = "<|im_end|>\n<|im_start|>user\n Next \n"
            + "<|im_end|>\n<|im_start|>assistant\n<think>\n"
        #expect(continuation == expectedContinuationIDs)
        #expect(tokenizer.decode(continuation, skipSpecialTokens: false)
                == expectedContinuation)
    }

    @Test("Pinned Maple Jinja leaves tool generation inside think")
    func toolTemplateSuffix() async throws {
        let (tokenizer, model) = try await Self.loadMapleTokenizer()
        defer { try? FileManager.default.removeItem(at: model) }

        let ids = try tokenizer.encodeToolChat(
            messages: [Message(role: .user, content: "Use lookup.")],
            tools: [
                .init(
                    name: "lookup",
                    description: "Look up a value",
                    parameters: .object([
                        "type": .string("object"),
                        "properties": .object([:]),
                    ])),
            ])
        let rendered = tokenizer.decode(ids, skipSpecialTokens: false)
        #expect(rendered.contains("# Tools"))
        #expect(rendered.contains("lookup"))
        #expect(rendered.contains("<|im_start|>user\nUse lookup.<|im_end|>\n"))
        #expect(rendered.hasSuffix("<|im_start|>assistant\n<think>\n"))
        #expect(!rendered.hasSuffix("<think>\n\n</think>\n\n"))
    }

    @Test("Maple refuses a model with no tokenizer sidecar")
    func missingTokenizerSidecar() async throws {
        let model = try Self.modelDirectory(withTokenizer: false)
        defer { try? FileManager.default.removeItem(at: model) }

        var rejected = false
        do {
            _ = try await MFTokenizer.load(
                forModelDirectory: model,
                environment: [:])
            Issue.record("missing Maple tokenizer sidecar was accepted")
        } catch MFTokenizerError.missingToolTemplate {
            rejected = true
        } catch {
            Issue.record("unexpected error: \(error)")
        }
        #expect(rejected)
    }

    @Test("Prompt-opened think suppresses tokens, flush, drain, and finish")
    func promptOpenedThoughtPrivacy() async throws {
        let (tokenizer, model) = try await Self.loadMapleTokenizer()
        defer { try? FileManager.default.removeItem(at: model) }

        let unfinished = StructuredAssistantDecoder(
            tokenizer: tokenizer,
            allowedTools: [],
            startsInThought: true)
        #expect(try unfinished.consume(tokenID: 872, delta: "private").isEmpty)
        #expect(try unfinished.consumeFlushedText(" tail").isEmpty)
        #expect(unfinished.drain().isEmpty)
        #expect(try unfinished.finish().isEmpty)

        let closed = StructuredAssistantDecoder(
            tokenizer: tokenizer,
            allowedTools: [],
            startsInThought: true)
        #expect(try closed.consume(tokenID: 872, delta: "private").isEmpty)
        #expect(try closed.consume(
            tokenID: try #require(tokenizer.thinkEndID),
            delta: "").isEmpty)
        #expect(try closed.consume(tokenID: 872, delta: "\n\nvisible")
                == [.content("\n\nvisible")])
        #expect(try closed.finish().isEmpty)
    }

    @Test("Maple decoder selects the JSON tool-call parser")
    func jsonToolCallDecoder() async throws {
        let (tokenizer, model) = try await Self.loadMapleTokenizer()
        defer { try? FileManager.default.removeItem(at: model) }

        let decoder = StructuredAssistantDecoder(
            tokenizer: tokenizer,
            allowedTools: ["lookup"],
            idGenerator: { "call_fixed" })
        _ = try decoder.consume(tokenID: tokenizer.toolCallStartID, delta: "")
        for tokenID in tokenizer.encode(
            #"{"name":"lookup","arguments":{"city":"Paris"}}"#,
            addBOS: false) {
            _ = try decoder.consume(tokenID: tokenID, delta: "")
        }
        let events = try decoder.consume(
            tokenID: tokenizer.toolCallEndID,
            delta: "")
        #expect(events == [.toolCall(ParsedToolCall(
            id: "call_fixed",
            name: "lookup",
            arguments: .object(["city": .string("Paris")]),
            argumentsJSON: #"{"city":"Paris"}"#))])
        #expect(decoder.hasToolCalls)
        #expect(try decoder.finish().isEmpty)
    }

    @Test("Family-neutral ChatML remains Qwen")
    func qwenUnchanged() async throws {
        let tokenizer = try await MFTokenizer.load(
            from: ChatMLTemplateTests.fixtureFolder())
        #expect(!tokenizer.generationPromptStartsInThinking)
        #expect(tokenizer.eosID == 248_044)
        #expect(tokenizer.eosID != tokenizer.endOfTurnID)
        #expect(tokenizer.vocabSize == 248_320)
    }
}

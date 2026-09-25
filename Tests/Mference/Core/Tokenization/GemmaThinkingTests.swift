import Foundation
import CryptoKit
import Testing
import MferenceValidationSupport
@testable import Mference
@testable import MferenceCLICore

@Suite("Gemma thinking template and decoder")
struct GemmaThinkingTests {
    @Test(arguments: [false, true], [false, true])
    func recoveryBoundaryExcludesOnlyTheActualGenerationThought(thinking: Bool, preserve: Bool) async throws {
        let tok = try await MFTokenizer.load(from: Self.fixtureFolder(), family: .gemma4)
        let messages: [MFTokenizer.Message] = [.init(role: .user, content: "Earlier"),
            .init(role: .assistant, content: "Done", reasoningContent: "Old thought"),
            .init(role: .user, content: "Next")]
        let effort: QwenReasoningEffort = thinking ? .medium : .off
        let ids = try tok.encodeChat(messages: messages, reasoningEffort: effort, preserveThinking: preserve)
        let boundary = try #require(try tok.gemmaRecoveryBoundary(messages: messages, tools: [],
            reasoningEffort: effort, preserveThinking: preserve, promptIDs: ids))
        let head = tok.decode(Array(ids[..<boundary]), skipSpecialTokens: false)
        #expect(head.hasSuffix("<|turn>model\n"))
        #expect(tok.decode(Array(ids[boundary...]), skipSpecialTokens: false)
            == (thinking ? "" : "<|channel>thought\n<channel|>"))
        #expect(!head.contains("Old thought"))
    }

    static func fixtureFolder() -> URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .appendingPathComponent("Fixtures/GemmaThinkingTokenizer")
    }

    struct Oracle: Decodable {
        let template_sha256: String
        let cases: [Case]
        struct Case: Decodable {
            let name: String
            let thinking: Bool
            let preserve: Bool
            let generate: Bool
            let render: String
            let ids: [Int32]
            let messages: [Message]
            let tools: [Tool]
        }
        struct Function: Decodable {
            let name: String
            let description: String?
            let parameters: JSONValue?
            let arguments: JSONValue?
        }
        struct Tool: Decodable { let function: Function }
        struct Call: Decodable { let id: String; let function: Function }
        struct Message: Decodable {
            let role: String
            let content: String?
            let reasoning_content: String?
            let tool_calls: [Call]?
            let tool_call_id: String?
        }
    }

    @Test func canonicalRendersMatchIndependentPythonOracle() async throws {
        let fixture = Self.fixtureFolder()
        // The same proof can use an existing install's vocabulary and an
        // independently rendered oracle for it, without loading model weights.
        let environment = ProcessInfo.processInfo.environment
        let installedFolder = environment["MFERENCE_GEMMA_TOKENIZER_DIR"]
        let installedOracle = environment["MFERENCE_GEMMA_TEMPLATE_ORACLE"]
        #expect((installedFolder == nil) == (installedOracle == nil))
        let tokenizerFolder = installedFolder.map { URL(fileURLWithPath: $0) } ?? fixture
        let oracleURL = installedOracle.map { URL(fileURLWithPath: $0) }
            ?? fixture.appendingPathComponent("oracle.json")
        let oracle = try JSONDecoder().decode(Oracle.self, from: Data(contentsOf: oracleURL))
        let source = try MFTokenizer.gemmaChatTemplateData()
        let referenceSource = try Data(contentsOf: fixture.appendingPathComponent("chat_template.jinja"))
        #expect(source == referenceSource)
        #expect(SHA256.hash(data: source).map { String(format: "%02x", $0) }.joined() == oracle.template_sha256)
        let tok = try await MFTokenizer.load(from: tokenizerFolder, family: .gemma4)
        for item in oracle.cases {
            let messages: [MFTokenizer.Message] = try item.messages.map { message in
                .init(role: try #require(MFTokenizer.Role(rawValue: message.role)), content: message.content,
                      toolCalls: (message.tool_calls ?? []).map { call in
                          .init(id: call.id, name: call.function.name,
                                arguments: call.function.arguments ?? .object([:]))
                      }, toolCallID: message.tool_call_id, reasoningContent: message.reasoning_content)
            }
            let tools: [MFTokenizer.FunctionDefinition] = item.tools.map {
                .init(name: $0.function.name, description: $0.function.description ?? "",
                      parameters: $0.function.parameters ?? .object([:]))
            }
            let ids = try tok.encodeToolChat(messages: messages, tools: tools,
                                             reasoningEffort: item.thinking ? .medium : .off,
                                             preserveThinking: item.preserve,
                                             addGenerationPrompt: item.generate)
            #expect(ids == item.ids, "\(item.name), thinking=\(item.thinking), preserve=\(item.preserve), generate=\(item.generate)")
            #expect(tok.decode(ids, skipSpecialTokens: false) == item.render)
            if item.generate {
                #expect(tok.startsInThinking(reasoningEffort: item.thinking ? .medium : .off, promptIDs: ids)
                        == item.render.hasSuffix("<|channel>thought\n"))
            }
        }
    }

    @Test func ordinaryStringAPIUsesTheSameCanonicalHistoryPolicy() async throws {
        let tok = try await MFTokenizer.load(from: Self.fixtureFolder(), family: .gemma4)
        let messages: [MFTokenizer.Message] = [
            .init(role: .user, content: "Hi"),
            .init(role: .assistant, content: "Done", reasoningContent: "Check"),
        ]
        let canonical = try tok.encodeChat(messages: messages)
        #expect(try tok.applyChatTemplate(messages) == tok.decode(canonical, skipSpecialTokens: false))
        #expect(try tok.applyChatTemplate(messages).contains("<|channel>thought\nCheck\n<channel|>"))
    }

    @Test func aliasesAreBinaryAndOmissionIsOff() async throws {
        let tok = try await MFTokenizer.load(from: Self.fixtureFolder(), family: .gemma4)
        let messages: [MFTokenizer.Message] = [.init(role: .user, content: "Hi")]
        let enabled = try tok.encodeChat(messages: messages, reasoningEffort: .medium)
        for effort in [QwenReasoningEffort.low, .xhigh] {
            #expect(try tok.encodeChat(messages: messages, reasoningEffort: effort) == enabled)
        }
        #expect(try tok.encodeChat(messages: messages) == tok.encodeChat(messages: messages, reasoningEffort: .off))
        #expect(try tok.encodeChat(messages: messages, preserveThinking: true) == tok.encodeChat(messages: messages))
        for effort in QwenReasoningEffort.allCases {
            let args = try Args.parse(["--model", "gemma4.gturbo", "--chat", "--reasoning-effort", effort.rawValue])
            #expect(args.reasoningEffort == effort)
            #expect(args.maxNew == 1024)
        }
    }

    @Test func loadedGemmaAcceptsThinkingAndOpensTheCanonicalPrompt() async throws {
        let tok = try await MFTokenizer.load(from: Self.fixtureFolder(), family: .gemma4)
        #expect(tok.acceptsReasoningEffort)
        let ids = try tok.encodeChat(messages: [.init(role: .user, content: "Hi")],
                                     reasoningEffort: .medium)
        #expect(tok.decode(ids, skipSpecialTokens: false)
                == "<bos><|turn>system\n<|think|>\n<turn|>\n<|turn>user\nHi<turn|>\n<|turn>model\n")
        #expect(!tok.startsInThinking(reasoningEffort: .medium, promptIDs: ids))
    }

    @Test func reasoningAndLiteralToolSyntaxStayInTheReasoningChannel() async throws {
        let tok = try await MFTokenizer.load(from: Self.fixtureFolder(), family: .gemma4)
        let decoder = StructuredAssistantDecoder(tokenizer: tok, allowedTools: [])
        var reasoning = ""
        decoder.onReasoning = { reasoning += $0 }
        #expect(try decoder.consume(tokenID: tok.channelStartID, delta: "<|channel>") == [])
        #expect(try decoder.consume(tokenID: -1, delta: "thou") == [])
        #expect(try decoder.consume(tokenID: -1, delta: "ght\nCheck ") == [])
        #expect(try decoder.consume(tokenID: tok.toolCallStartID, delta: "<|tool_call>") == [])
        #expect(try decoder.consume(tokenID: -1, delta: "call:example{}") == [])
        #expect(try decoder.consume(tokenID: tok.toolCallEndID, delta: "<tool_call|>") == [])
        #expect(try decoder.consumeFlushedText(" carefully.") == [])
        #expect(try decoder.consume(tokenID: tok.channelEndID, delta: "<channel|>") == [])
        #expect(try decoder.consume(tokenID: -1, delta: "Done.") == [.content("Done.")])
        #expect(reasoning == "Check <|tool_call>call:example{}<tool_call|> carefully.")
    }

    @Test(arguments: [false, true])
    func thoughtCutoffKeepsReasoningWithoutInventingAnAnswer(preopened: Bool) async throws {
        let tok = try await MFTokenizer.load(from: Self.fixtureFolder(), family: .gemma4)
        let decoder = StructuredAssistantDecoder(tokenizer: tok, allowedTools: [],
                                                  startsInThought: preopened)
        var reasoning = ""
        var events: [StructuredAssistantEvent] = []
        decoder.onReasoning = { reasoning += $0 }
        let text = (preopened ? "" : "<|channel>thought\n") + "Still checking"
        for id in tok.encode(text, addBOS: false) {
            events += try decoder.consume(tokenID: id, delta: tok.decode([id], skipSpecialTokens: false))
        }
        events += try decoder.consumeFlushedText(".")
        events += try decoder.finish()
        #expect(reasoning == "Still checking.")
        #expect(events.isEmpty)
        #expect(!decoder.hasToolCalls)
        #expect(decoder.payloadTokenCounts == nil)
    }

    @Test func finalChannelAndIncompleteToolsKeepTheirDistinctBoundaries() async throws {
        let tok = try await MFTokenizer.load(from: Self.fixtureFolder(), family: .gemma4)
        let decoder = StructuredAssistantDecoder(tokenizer: tok, allowedTools: ["lookup"])
        var reasoning = ""
        decoder.onReasoning = { reasoning += $0 }
        #expect(try decoder.consume(tokenID: tok.channelStartID, delta: "<|channel>") == [])
        #expect(try decoder.consume(tokenID: -1, delta: "fi") == [])
        #expect(try decoder.consumeFlushedText("nal\nReady.") == [.content("Ready.")])
        #expect(reasoning.isEmpty)
        // A truncated visible tool payload remains invalid, even though a
        // truncated thought is a valid length-limited reasoning response.
        #expect(try decoder.consume(tokenID: tok.toolCallStartID, delta: "<|tool_call>") == [])
        for id in tok.encode("call:lookup{", addBOS: false) {
            #expect(try decoder.consume(tokenID: id, delta: tok.decode([id], skipSpecialTokens: false)) == [])
        }
        #expect(throws: ToolCallParserError.self) { try decoder.finish() }
        #expect(!decoder.hasToolCalls)
    }
    @Test func rawLoopPreservesBytesAcrossThoughtAndToolBoundaries() async throws {
        // Exercise the production loop and SentencePiece byte fallback, not
        // pre-decoded synthetic deltas. No model weights are needed.
        let tok = try await MFTokenizer.load()
        let context = try MetalContext()
        let text = "<|channel>thought\nCheck 🦙<|tool_call>call:example{}<tool_call|>🦙<channel|>Done 🦙<|channel>final\nOK"
        let sequence = tok.encode(text, addBOS: false) + [tok.endOfTurnID]
        let producer = ScriptedLogitProducer(vocabSize: tok.vocabSize) { _, index in
            .argmax(sequence[min(index, sequence.count - 1)])
        }
        let scratch = try RawCompletionScratch(context: context, vocab: tok.vocabSize)
        let decoder = StructuredAssistantDecoder(tokenizer: tok, allowedTools: [])
        var reasoning = ""
        var events: [StructuredAssistantEvent] = []
        var decodingError: Error?
        decoder.onReasoning = { reasoning += $0 }
        let result = try await runRawCompletion(
            producer: producer, tokenizer: tok, promptIds: [tok.bosID],
            config: GenerationConfig(maxNewTokens: sequence.count + 1, temperature: 0),
            context: context, scratch: scratch, prefillConfig: .off
        ) { progress in
            do {
                switch progress {
                case .prefill: break
                case .token(_, let id, let delta):
                    events += try decoder.consume(tokenID: id, delta: delta)
                case .tail(let text): events += try decoder.consumeFlushedText(text)
                }
            } catch { decodingError = error }
        }
        if let decodingError { throw decodingError }
        events += try decoder.finish()
        #expect(result.reason == .endOfTurn)
        #expect(reasoning == "Check 🦙<|tool_call>call:example{}<tool_call|>🦙")
        #expect(events.compactMap { if case .content(let text) = $0 { text } else { nil } }.joined() == "Done 🦙OK")
        #expect(!decoder.hasToolCalls)
    }

}

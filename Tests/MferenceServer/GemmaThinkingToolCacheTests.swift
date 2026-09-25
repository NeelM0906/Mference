import Foundation
import Testing
@testable import Mference
@testable import MferenceServerCore

extension GemmaThinkingServerTests {
    @Test(arguments: [false, true], [false, true])
    func repeatedToolRoundsReuseOnlyTheirOwnContinuation(thinking: Bool, preserve: Bool) async throws {
        let folder = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .deletingLastPathComponent().appendingPathComponent("Mference/Core/Tokenization/Fixtures/GemmaThinkingTokenizer")
        let tok = try await MFTokenizer.load(from: folder, family: .gemma4)
        let domain = ServerPromptCacheDomain(modelID: "gemma-alias", sourceSnapshotHash: "source",
            runtimeProfileHash: "profile", maximumContext: 4096, kvStorage: "fp16",
            fp16RingEnabled: true, templateSHA256: "pinned-template")
        let tools: [MFTokenizer.FunctionDefinition] = [.init(name: "lookup", description: "Lookup",
            parameters: .object(["type": .string("object"), "properties": .object([
                "query": .object(["type": .string("string")])]), "required": .array([.string("query")])]))]
        func request(_ messages: [MFTokenizer.Message], effort: QwenReasoningEffort? = nil,
                     definitions: [MFTokenizer.FunctionDefinition]? = nil) -> ValidatedChatRequest {
            .init(reasoningEffort: effort ?? (thinking ? .medium : .off), preserveThinking: preserve,
                  messages: messages, tools: definitions ?? tools, stream: false, includeUsage: false,
                  generationConfig: .init(maxNewTokens: 128, temperature: 0), maximumCompletionTokens: 128)
        }
        func render(_ request: ValidatedChatRequest) throws -> [Int32] {
            try tok.encodeChat(messages: request.messages, tools: request.tools,
                reasoningEffort: request.reasoningEffort, preserveThinking: request.preserveThinking)
        }
        var cache = ServerPromptCache()
        var current = request([.init(role: .user, content: "Lookup A")])
        var effective = try render(current)
        let oracle = try #require(JSONSerialization.jsonObject(with: Data(contentsOf:
            folder.appendingPathComponent("oracle.json"))) as? [String: Any])
        let cases = try #require(oracle["cases"] as? [[String: Any]])
        for round in 1...2 {
            let ids = round == 1 ? ["call_1"] : ["call_2", "call_3"]
            let reasoning = round == 1 ? "Need lookup." : "Again."
            let calls = ids.map { ParsedToolCall(id: $0, name: "lookup",
                arguments: .object(["query": .string("A")]), argumentsJSON: #"{"query":"A"}"#) }
            let assistant = MFTokenizer.Message(role: .assistant, content: nil,
                toolCalls: calls.map { .init(id: $0.id, name: $0.name, arguments: $0.arguments) },
                reasoningContent: reasoning)
            // Generated whitespace need not equal a canonical re-render. Keep
            // the actual committed prefix and bridge only at the tool boundary.
            let opener = tok.startsInThinking(reasoningEffort: current.reasoningEffort,
                                              promptIDs: effective) ? "" : "<|channel>thought\n"
            let generated = opener + reasoning + "<channel|>" + ids.map { _ in
                #"<|tool_call>call:lookup{query:<|"|>A<|"|>}<tool_call|>"#
            }.joined()
            let kv = effective + tok.encode(generated, addBOS: false)
            cache.publish(domain: domain, request: current, content: "", calls: calls,
                result: RawDecodeResult(prefillTokens: effective.count, cachedPromptTokens: 0,
                    computedPrefillTokens: effective.count, prefillSeconds: 0, newTokens: 1,
                    decodeSeconds: 0, reason: .toolCalls, kvPosition: kv.count,
                    kvBackedTokenIDs: kv, uncommittedBoundaryTokenIDs: [tok.toolResponseID],
                    prefillExecution: nil), reasoningContent: reasoning)
            let resultTexts = round == 1 ? ["Found A"] : ["Second A", "Third A"]
            let results = zip(ids, resultTexts).map { MFTokenizer.Message(role: .tool, content: $0.1, toolCallID: $0.0) }
            let next = request(current.messages + [assistant] + results,
                               effort: thinking ? .low : .off)
            let rendered = try render(next)
            let reference = try #require(cases.first { item in
                item["name"] as? String == (round == 1 ? "tool_result" : "repeated_tools")
                    && item["thinking"] as? Bool == thinking && item["preserve"] as? Bool == preserve
                    && item["generate"] as? Bool == true
            })
            #expect(rendered == (reference["ids"] as? [Int])?.map(Int32.init))
            guard case .hit(let resumed, let cached) = cache.match(domain: domain, request: next,
                renderedPromptIDs: rendered, tokenizer: tok) else {
                Issue.record("round \(round) must reuse the actual Gemma tool-call prefix")
                return
            }
            let expectedTail = resultTexts.map {
                "<|tool_response>response:lookup{value:<|\"|>\($0)<|\"|>}<tool_response|>"
            }.joined() + (thinking ? "<|channel>thought\n" : "")
            #expect(cached == kv.count)
            #expect(resumed == kv + tok.encode(expectedTail, addBOS: false))
            #expect(tok.startsInThinking(reasoningEffort: next.reasoningEffort, promptIDs: resumed) == thinking)
            #expect(!rendered.prefix(kv.count).elementsEqual(kv))
            for changedReasoning in [nil, "Edited"] as [String?] {
                var altered = next.messages
                altered[current.messages.count] = .init(role: .assistant, content: nil,
                    toolCalls: assistant.toolCalls, reasoningContent: changedReasoning)
                let edited = request(altered)
                #expect(cache.match(domain: domain, request: edited,
                    renderedPromptIDs: try render(edited), tokenizer: tok) == .miss)
            }
            let changedTools = request(next.messages, definitions: [.init(name: "lookup",
                description: "Different schema", parameters: .object(["type": .string("object")]))])
            #expect(cache.match(domain: domain, request: changedTools,
                renderedPromptIDs: try render(changedTools), tokenizer: tok) == .miss)
            current = next
            effective = resumed
        }
        let newUser = request(current.messages + [.init(role: .user, content: "Again")])
        #expect(cache.match(domain: domain, request: newUser,
            renderedPromptIDs: try render(newUser), tokenizer: tok) == .miss)
    }
}

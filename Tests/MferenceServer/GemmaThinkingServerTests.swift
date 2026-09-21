import Foundation
import Testing
@testable import Mference
@testable import MferenceServerCore

@Suite("Gemma server thinking", .serialized)
struct GemmaThinkingServerTests {
    @Test func thinkingKeepsGemmasDefaultCompletionBudget() throws {
        for model in ["custom-alias", "gemma4@gemma4.gturbo"] {
            let data = Data("""
            {"model":"\(model)","chat_template_kwargs":{"enable_thinking":true},
             "messages":[{"role":"user","content":"Hi"}]}
            """.utf8)
            let request = try JSONDecoder().decode(OpenAIChatRequest.self, from: data)
            let validated = try OpenAIRequestValidator.validate(
                request, modelID: model, dialect: .gemma,
                acceptsReasoningEffort: true, qwenReasoning: false)
            #expect(validated.reasoningEffort == .xhigh)
            #expect(validated.maximumCompletionTokens == 4096)
        }
    }

    @Test(arguments: [false, true], [false, true])
    func controlsRenderAndReasoningStreamsThroughBothHTTPModes(stream: Bool, libraryMode: Bool) async throws {
        let fixture = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .deletingLastPathComponent().appendingPathComponent("Mference/Core/Tokenization/Fixtures/GemmaThinkingTokenizer")
        let tok = try await MFTokenizer.load(from: fixture, family: .gemma4)
        let backend = GemmaTemplateBackend(tokenizer: tok)
        let modelID = libraryMode ? "gemma4@gemma4.gturbo" : "custom-gemma-alias"
        let server: MferenceHTTPServer
        if libraryMode {
            let entry = ServerLibraryEntry(modelID: modelID, familyModelID: "gemma4",
                basename: "gemma4.gturbo", directory: URL(fileURLWithPath: "/unused/gemma4.gturbo"), family: .gemma4)
            server = MferenceHTTPServer(library: ServerModelLibrary(index: .init(entries: [entry])) { _ in backend }, queueLimit: 1)
        } else {
            server = MferenceHTTPServer(modelID: modelID, queueLimit: 1, backend: backend, chatDialect: .gemma)
        }
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)
        do {
            func send(_ fields: String) async throws -> (Data, URLResponse) {
                var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!)
                request.httpMethod = "POST"
                request.setValue("application/json", forHTTPHeaderField: "content-type")
                request.httpBody = Data("""
                {"model":"\(modelID)","stream":\(stream),\(fields)
                 "messages":[{"role":"user","content":"Hi"}]}
                """.utf8)
                return try await URLSession.shared.data(for: request)
            }
            let (data, response) = try await send("\"reasoning_effort\":\"low\",\"chat_template_kwargs\":{\"enable_thinking\":false,\"preserve_thinking\":true},")
            #expect((response as? HTTPURLResponse)?.statusCode == 200)
            let received = await backend.received
            #expect(received?.request.reasoningEffort == .low)
            #expect(received?.request.preserveThinking == false)
            #expect(received?.request.maximumCompletionTokens == 4096)
            let rendered = tok.decode(received?.promptIDs ?? [], skipSpecialTokens: false)
            #expect(rendered.contains("<|think|>"))
            let text = String(decoding: data, as: UTF8.self)
            #expect(text.contains("reasoning_content"))
            #expect(!text.contains("<|channel>"))
            var returnedReasoning = ""
            var returnedContent = ""
            let payloads: [Data]
            if stream {
                #expect(text.hasSuffix("data: [DONE]\n\n"))
                payloads = text.components(separatedBy: "\n").filter {
                    $0.hasPrefix("data: {")
                }.map { Data($0.dropFirst(6).utf8) }
            } else { payloads = [data] }
            for payload in payloads {
                let object = try #require(JSONSerialization.jsonObject(with: payload) as? [String: Any])
                let choices = try #require(object["choices"] as? [[String: Any]])
                let message = try #require(choices.first?[stream ? "delta" : "message"] as? [String: Any])
                returnedReasoning += message["reasoning_content"] as? String ?? ""
                returnedContent += message["content"] as? String ?? ""
            }
            #expect(returnedReasoning == "Checked.")
            #expect(returnedContent == "Done.")
            for fields in ["", "\"chat_template_kwargs\":{\"preserve_thinking\":true},",
                           "\"reasoning_effort\":\"none\",\"chat_template_kwargs\":{\"enable_thinking\":true},"] {
                let (_, reply) = try await send(fields)
                #expect((reply as? HTTPURLResponse)?.statusCode == 200)
                let prepared = await backend.received
                let renderedOff = tok.decode(prepared?.promptIDs ?? [], skipSpecialTokens: false)
                #expect(!renderedOff.contains("<|think|>"))
            }
            for fields in ["\"reasoning_effort\":\"high\",", "\"chat_template_kwargs\":{\"enable_thinking\":\"true\"},",
                           "\"chat_template_kwargs\":{\"preserve_thinking\":1},"] {
                let (error, reply) = try await send(fields)
                #expect((reply as? HTTPURLResponse)?.statusCode == 400)
                #expect(!String(decoding: error, as: UTF8.self).contains("data: [DONE]"))
            }
            try await server.shutdown()
        } catch {
            try await server.shutdown()
            throw error
        }
    }
}

/// Uses the real renderer and decoder at the HTTP seam; real model proof is
/// separate. No synthetic backend can establish GPU or KV correctness.
private actor GemmaTemplateBackend: ServerLoadedModel {
    nonisolated let chatDialect: ChatDialect = .gemma
    nonisolated var acceptsReasoningEffort: Bool { tokenizer.acceptsReasoningEffort }
    nonisolated let tokenizer: MFTokenizer
    private(set) var received: PreparedGeneration?
    init(tokenizer: MFTokenizer) { self.tokenizer = tokenizer }

    func prepare(_ request: ValidatedChatRequest) throws -> PreparedGeneration {
        let ids = try tokenizer.encodeChat(messages: request.messages, tools: request.tools,
            reasoningEffort: request.reasoningEffort, preserveThinking: request.preserveThinking)
        return PreparedGeneration(request: request, promptIDs: ids)
    }

    func generate(_ prepared: PreparedGeneration,
                  onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void) async throws -> ServerCompletion {
        received = prepared
        let decoder = StructuredAssistantDecoder(tokenizer: tokenizer, allowedTools: [],
            startsInThought: tokenizer.startsInThinking(reasoningEffort: prepared.request.reasoningEffort,
                                                        promptIDs: prepared.promptIDs))
        var reasoning = ""
        var content = ""
        decoder.onReasoning = { reasoning += $0; onEvent(.reasoning($0)) }
        let thinking = prepared.request.reasoningEffort != nil && prepared.request.reasoningEffort != .off
        let output = (thinking ? "<|channel>thought\nChecked.<channel|>" : "") + "Done."
        for id in tokenizer.encode(output, addBOS: false) {
            for event in try decoder.consume(tokenID: id, delta: tokenizer.decode([id], skipSpecialTokens: false)) {
                if case .content(let text) = event { content += text; onEvent(.content(text)) }
            }
        }
        _ = try decoder.finish()
        var result = ServerCompletion(content: content, toolCalls: [], finishReason: "stop",
            usage: OpenAIUsage(promptTokens: prepared.promptIDs.count, completionTokens: 10,
                               totalTokens: prepared.promptIDs.count + 10))
        result.reasoningContent = reasoning.isEmpty ? nil : reasoning
        return result
    }
}

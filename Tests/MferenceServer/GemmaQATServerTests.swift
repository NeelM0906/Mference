import Foundation
import Testing
@testable import Mference
@testable import MferenceServerCore

@Suite("Gemma QAT server profile", .serialized)
struct GemmaQATServerTests {
    static func tokenizer(qat: Bool = true) async throws -> MFTokenizer {
        let directory = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .deletingLastPathComponent().appendingPathComponent("Mference/Core/Tokenization/Fixtures/" +
                (qat ? "GemmaQATTokenizer" : "GemmaThinkingTokenizer"))
        return try await MFTokenizer.load(from: directory, family: .gemma4)
            .forCheckpoint(qat ? CheckpointIdentity.gemma4QAT : "gemma-4-26b-a4b-it")
    }

    static func entry(_ id: String, qat: Bool) -> ServerLibraryEntry {
        .init(modelID: id, familyModelID: qat ? CheckpointIdentity.gemma4QAT : "gemma-4-26b-a4b-it",
            basename: qat ? "qat.gturbo" : "original.gturbo",
            directory: URL(fileURLWithPath: qat ? "/unused/qat.gturbo" : "/unused/original.gturbo"), family: .gemma4)
    }

    static func request(port: Int, model: String, stream: Bool, fields: String = "") -> URLRequest {
        var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "content-type")
        request.httpBody = Data("""
        {"model":"\(model)","stream":\(stream),\(fields)
         "messages":[{"role":"user","content":"Hi"}]}
        """.utf8)
        request.timeoutInterval = 10
        return request
    }

    @Test(arguments: [false, true], [false, true])
    func defaultsAndControlsUseTheSelectedProfile(stream: Bool, libraryMode: Bool) async throws {
        let backend = QATProfileBackend(tokenizer: try await Self.tokenizer())
        let id = libraryMode ? "\(CheckpointIdentity.gemma4QAT)@renamed.gturbo" : "custom-alias"
        let server: MferenceHTTPServer
        if libraryMode {
            let library = ServerModelLibrary(index: .init(entries: [Self.entry(id, qat: true)])) { _ in backend }
            server = MferenceHTTPServer(library: library, queueLimit: 1)
        } else {
            server = MferenceHTTPServer(modelID: id, queueLimit: 1, backend: backend, chatDialect: .gemma)
        }
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)
        do {
            let (data, response) = try await URLSession.shared.data(for: Self.request(port: port, model: id, stream: stream,
                fields: "\"chat_template_kwargs\":{\"enable_thinking\":true},"))
            #expect((response as? HTTPURLResponse)?.statusCode == 200)
            let received = try #require(await backend.received)
            #expect(received.request.generationConfig.temperature == 1)
            #expect(received.request.generationConfig.topK == 64)
            #expect(received.request.generationConfig.topP == 0.95)
            #expect(received.request.generationConfig.minP == 0)
            #expect(received.request.generationConfig.repetitionPenalty == 1)
            #expect(received.request.reasoningEffort == .xhigh)
            #expect(received.request.maximumCompletionTokens == 4096)
            #expect(backend.tokenizer.decode(received.promptIDs, skipSpecialTokens: false).hasSuffix("<|turn>model\n"))
            let text = String(decoding: data, as: UTF8.self)
            #expect(text.contains("reasoning_content"))
            #expect(!text.contains("<|channel>"))
            if stream { #expect(text.hasSuffix("data: [DONE]\n\n")) }

            for fields in [
                "\"temperature\":0,\"top_k\":0,\"top_p\":1,\"min_p\":0,\"repeat_penalty\":1,\"presence_penalty\":0,\"frequency_penalty\":0,",
                "\"min_p\":0.2,", ""
            ] {
                let (_, reply) = try await URLSession.shared.data(for: Self.request(port: port, model: id, stream: stream, fields: fields))
                #expect((reply as? HTTPURLResponse)?.statusCode == 200)
                let config = try #require(await backend.received?.request.generationConfig)
                if fields.hasPrefix("\"temperature\"") {
                    #expect(config.temperature == 0 && config.topK == nil && config.topP == 1 && config.minP == 0)
                    #expect(config.repetitionPenalty == 1 && config.presencePenalty == 0 && config.frequencyPenalty == 0)
                } else {
                    #expect(config.temperature == 1 && config.topK == 64 && config.topP == 0.95)
                    #expect(config.minP == (fields.isEmpty ? 0 : 0.2))
                }
            }
            let (error, reply) = try await URLSession.shared.data(for: Self.request(port: port, model: id, stream: stream,
                fields: "\"chat_template_kwargs\":{\"preserve_thinking\":true},"))
            #expect((reply as? HTTPURLResponse)?.statusCode == 400)
            #expect((reply as? HTTPURLResponse)?.value(forHTTPHeaderField: "content-type")?.contains("application/json") == true)
            #expect(String(decoding: error, as: UTF8.self).contains("unsupported_value"))
            #expect(!String(decoding: error, as: UTF8.self).contains("data: "))
            try await server.shutdown()
        } catch {
            try await server.shutdown()
            throw error
        }
    }

    @Test func swappingRestoresEachTemplateAndSamplingProfile() async throws {
        let original = QATProfileBackend(tokenizer: try await Self.tokenizer(qat: false))
        let qat = QATProfileBackend(tokenizer: try await Self.tokenizer())
        let library = ServerModelLibrary(index: .init(entries: [Self.entry("original", qat: false), Self.entry("qat", qat: true)])) {
            $0.lastPathComponent == "qat.gturbo" ? qat : original
        }
        let server = MferenceHTTPServer(library: library, queueLimit: 1)
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)
        do {
            var originalIDs: [Int32] = []
            for id in ["original", "qat", "original"] {
                let (_, response) = try await URLSession.shared.data(for: Self.request(port: port, model: id, stream: false))
                #expect((response as? HTTPURLResponse)?.statusCode == 200)
                let prepared = try #require(await (id == "qat" ? qat : original).received)
                let config = prepared.request.generationConfig
                #expect(config.temperature == (id == "qat" ? 1 : 0.8))
                #expect(config.topK == (id == "qat" ? 64 : 40))
                #expect(config.minP == (id == "qat" ? 0 : 0.05))
                if id == "original" {
                    if !originalIDs.isEmpty { #expect(prepared.promptIDs == originalIDs) }
                    originalIDs = prepared.promptIDs
                }
            }
            try await server.shutdown()
        } catch {
            try await server.shutdown()
            throw error
        }
    }

    @Test func queuedQATPreserveThinkingFailsBeforeStreaming() async throws {
        let blocker = QATProfileBackend(tokenizer: try await Self.tokenizer(qat: false), blocks: true)
        let qat = QATProfileBackend(tokenizer: try await Self.tokenizer())
        let library = ServerModelLibrary(index: .init(entries: [Self.entry("original", qat: false), Self.entry("qat", qat: true)])) {
            $0.lastPathComponent == "qat.gturbo" ? qat : blocker
        }
        let server = MferenceHTTPServer(library: library, queueLimit: 1)
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)
        let first = Task { try await URLSession.shared.data(for: Self.request(port: port, model: "original", stream: false)) }
        do {
            for _ in 0..<200 {
                if await blocker.started { break }
                try await Task.sleep(for: .milliseconds(5))
            }
            #expect(await blocker.started)
            let (_, response) = try await URLSession.shared.bytes(for: Self.request(port: port, model: "qat", stream: true,
                fields: "\"chat_template_kwargs\":{\"preserve_thinking\":true},"))
            #expect((response as? HTTPURLResponse)?.statusCode == 400)
            #expect((response as? HTTPURLResponse)?.value(forHTTPHeaderField: "content-type")?.contains("application/json") == true)
            #expect(await qat.received == nil)
            await blocker.release()
            _ = try await first.value
            try await server.shutdown()
        } catch {
            await blocker.release()
            _ = try? await first.value
            try await server.shutdown()
            throw error
        }
    }
}

private actor QATProfileBackend: ServerLoadedModel {
    nonisolated let tokenizer: MFTokenizer
    nonisolated let chatDialect: ChatDialect = .gemma
    nonisolated var acceptsReasoningEffort: Bool { tokenizer.acceptsReasoningEffort }
    nonisolated var isGemmaQAT: Bool { tokenizer.isGemmaQAT }
    nonisolated var generationDefaults: GenerationConfig { tokenizer.generationDefaults }
    private(set) var received: PreparedGeneration?
    private(set) var started = false
    private var blocks: Bool
    private var continuation: CheckedContinuation<Void, Never>?

    init(tokenizer: MFTokenizer, blocks: Bool = false) { self.tokenizer = tokenizer; self.blocks = blocks }

    func release() { blocks = false; continuation?.resume(); continuation = nil }

    func prepare(_ request: ValidatedChatRequest) throws -> PreparedGeneration {
        let ids = try tokenizer.encodeChat(messages: request.messages, tools: request.tools,
            reasoningEffort: request.reasoningEffort, preserveThinking: request.preserveThinking)
        return PreparedGeneration(request: request, promptIDs: ids)
    }

    func generate(_ prepared: PreparedGeneration,
                  onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void) async throws -> ServerCompletion {
        received = prepared
        started = true
        if blocks { await withCheckedContinuation { continuation = $0 } }
        let decoder = StructuredAssistantDecoder(tokenizer: tokenizer, allowedTools: [],
            startsInThought: tokenizer.startsInThinking(reasoningEffort: prepared.request.reasoningEffort, promptIDs: prepared.promptIDs))
        var reasoning = ""
        var content = ""
        decoder.onReasoning = { reasoning += $0; onEvent(.reasoning($0)) }
        let output = (prepared.request.reasoningEffort != nil && prepared.request.reasoningEffort != .off
            ? "<|channel>thought\nChecked.<channel|>" : "") + "Done."
        for id in tokenizer.encode(output, addBOS: false) {
            for event in try decoder.consume(tokenID: id, delta: tokenizer.decode([id], skipSpecialTokens: false)) {
                if case .content(let text) = event { content += text; onEvent(.content(text)) }
            }
        }
        _ = try decoder.finish()
        var completion = ServerCompletion(content: content, toolCalls: [], finishReason: "stop",
            usage: .init(promptTokens: prepared.promptIDs.count, completionTokens: 10, totalTokens: prepared.promptIDs.count + 10))
        completion.reasoningContent = reasoning.isEmpty ? nil : reasoning
        return completion
    }
}

import Darwin
import Foundation
import NIOCore
import Testing
@testable import Mference
@testable import MferenceServerCore

private actor ScriptedServerBackend: ServerInferenceBackend {
    nonisolated let usesSwiftQwenTemplate: Bool
    nonisolated let supportsQwenReasoningEffort: Bool
    let delayNanoseconds: UInt64
    let diagnostics: RuntimeDiagnostics?

    init(delayNanoseconds: UInt64 = 0, includeDiagnostics: Bool = false, includeReasoning: Bool = false,
         baseQwen: Bool = false) {
        self.usesSwiftQwenTemplate = includeReasoning && !baseQwen
        self.supportsQwenReasoningEffort = includeReasoning
        self.delayNanoseconds = delayNanoseconds
        if includeDiagnostics {
            let result = RawDecodeResult(
                prefillTokens: 3, cachedPromptTokens: 0, computedPrefillTokens: 3,
                prefillSeconds: 0, newTokens: 1, decodeSeconds: 0, reason: .maxTokens,
                kvPosition: 3, kvBackedTokenIDs: [], uncommittedBoundaryTokenIDs: [],
                prefillExecution: nil)
            diagnostics = RuntimeDiagnostics(result: result,
                memory: RuntimeMemorySnapshot(bytes: ["processRSS": 123]))
        } else {
            diagnostics = nil
        }
    }

    func generate(
        _ prepared: PreparedGeneration,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        if delayNanoseconds > 0 {
            try await Task.sleep(nanoseconds: delayNanoseconds)
        }
        if supportsQwenReasoningEffort { onEvent(.reasoning("Check first.")) }
        onEvent(.content("hello"))
        var completion = ServerCompletion(
            content: "hello",
            toolCalls: [],
            finishReason: "stop",
            usage: OpenAIUsage(promptTokens: 3, completionTokens: 1, totalTokens: 4,
                               completionTokensDetails: .init(reasoningTokens: 0, visibleTokens: 1)),
            diagnostics: diagnostics)
        completion.reasoningContent = supportsQwenReasoningEffort ? "Check first." : nil
        return completion
    }
}

/// Qwen 3.6 opts into thinking without the Qwen 3.8 source-template capability.
private actor OptInThinkingBackend: ServerLoadedModel {
    nonisolated let chatDialect: ChatDialect = .chatml
    nonisolated let acceptsReasoningEffort = true
    private(set) var received: ValidatedChatRequest?

    func generate(
        _ prepared: PreparedGeneration,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        received = prepared.request
        onEvent(.reasoning("Check first."))
        onEvent(.content("hello"))
        var completion = ServerCompletion(
            content: "hello", toolCalls: [], finishReason: "stop",
            usage: OpenAIUsage(promptTokens: 3, completionTokens: 1, totalTokens: 4))
        completion.reasoningContent = "Check first."
        return completion
    }
}

private actor MultipleToolBackend: ServerInferenceBackend {
    func generate(
        _ prepared: PreparedGeneration,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        let first = ParsedToolCall(
            id: "call_000000000000000000000001",
            name: "read",
            arguments: .object(["path": .string("/tmp/a")]),
            argumentsJSON: #"{"path":"/tmp/a"}"#)
        let second = ParsedToolCall(
            id: "call_000000000000000000000002",
            name: "read",
            arguments: .object(["path": .string("/tmp/b")]),
            argumentsJSON: #"{"path":"/tmp/b"}"#)
        onEvent(.toolCall(first))
        onEvent(.toolCall(second))
        return ServerCompletion(
            content: "",
            toolCalls: [first, second],
            finishReason: "tool_calls",
            usage: OpenAIUsage(promptTokens: 3, completionTokens: 8, totalTokens: 11))
    }
}

private actor ContentAndToolBackend: ServerInferenceBackend {
    func generate(
        _ prepared: PreparedGeneration,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        let content = "I will read it."
        let call = ParsedToolCall(
            id: "call_000000000000000000000003",
            name: "read",
            arguments: .object(["path": .string("/tmp/mixed")]),
            argumentsJSON: #"{"path":"/tmp/mixed"}"#)
        onEvent(.content(content))
        onEvent(.toolCall(call))
        return ServerCompletion(
            content: content,
            toolCalls: [call],
            finishReason: "tool_calls",
            usage: OpenAIUsage(promptTokens: 3, completionTokens: 8, totalTokens: 11))
    }
}

private actor PipelinedRequestBackend: ServerInferenceBackend {
    private var continuation: CheckedContinuation<Void, Never>?
    private(set) var generationCount = 0

    var isWaiting: Bool { continuation != nil }

    func release() {
        continuation?.resume()
        continuation = nil
    }

    func generate(
        _ prepared: PreparedGeneration,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        generationCount += 1
        await withCheckedContinuation { continuation = $0 }
        onEvent(.content("first"))
        return ServerCompletion(
            content: "first",
            toolCalls: [],
            finishReason: "stop",
            usage: OpenAIUsage(promptTokens: 3, completionTokens: 1, totalTokens: 4))
    }
}

private struct DecodeFailure: Error {}

/// Fails after `onEvent` has already put content on the wire, the way a decode
/// failure does mid-generation.
private actor FailingMidStreamBackend: ServerInferenceBackend {
    func generate(
        _ prepared: PreparedGeneration,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        onEvent(.content("partial"))
        throw DecodeFailure()
    }
}

/// Rejects from inside `generate`, once the streaming head is committed. The
/// live equivalent is the effective-prompt check, which cannot run earlier
/// because it depends on the KV prefix the request is matched against.
private actor RejectingBackend: ServerInferenceBackend {
    func generate(
        _ prepared: PreparedGeneration,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        throw ServerRequestError.invalid(
            message: "effective prompt exceeds the configured context",
            param: "messages",
            code: "context_length_exceeded")
    }
}

/// Rejects during `prepare`, the way an overlong prompt does. Mirrors the
/// actor-isolated, non-`async` shape `ServerModelSession` uses, so a witness
/// that silently fell back to the pass-through default would fail this test.
private actor RejectingPrepareBackend: ServerInferenceBackend {
    func prepare(_ request: ValidatedChatRequest) throws -> PreparedGeneration {
        throw ServerRequestError.invalid(
            message: "prompt exceeds the configured context",
            param: "messages",
            code: "context_length_exceeded")
    }

    func generate(
        _ prepared: PreparedGeneration,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        Issue.record("generate must not run after prepare fails")
        return ServerCompletion(
            content: "unexpected",
            toolCalls: [],
            finishReason: "stop",
            usage: OpenAIUsage(promptTokens: 1, completionTokens: 1, totalTokens: 2))
    }
}

/// Counts renders and parks in `generate`, so a request that the queue turns
/// away can be shown never to have paid for tokenization.
private actor RenderCountingBackend: ServerInferenceBackend {
    private(set) var prepareCount = 0
    private var parked: [CheckedContinuation<Void, Never>] = []
    private var released = false

    func prepare(_ request: ValidatedChatRequest) throws -> PreparedGeneration {
        prepareCount += 1
        return PreparedGeneration(request: request)
    }

    func generate(
        _ prepared: PreparedGeneration,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        if !released {
            await withCheckedContinuation { parked.append($0) }
        }
        return ServerCompletion(
            content: "done",
            toolCalls: [],
            finishReason: "stop",
            usage: OpenAIUsage(promptTokens: 1, completionTokens: 1, totalTokens: 2))
    }

    /// Latches open: a queued request that reaches `generate` only after the
    /// release must not park again, or shutdown would wait on it forever.
    func releaseAll() {
        released = true
        for continuation in parked { continuation.resume() }
        parked.removeAll()
    }
}

private actor CancellableServerBackend: ServerInferenceBackend {
    private(set) var startedCount = 0
    private(set) var cancellationCount = 0

    func generate(
        _ prepared: PreparedGeneration,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        startedCount += 1
        do {
            try await Task.sleep(for: .seconds(30))
        } catch is CancellationError {
            cancellationCount += 1
            throw CancellationError()
        }
        return ServerCompletion(
            content: "unexpected",
            toolCalls: [],
            finishReason: "stop",
            usage: OpenAIUsage(promptTokens: 1, completionTokens: 1, totalTokens: 2))
    }
}

@Suite("OpenAI HTTP server", .serialized)
struct HTTPServerTests {
    @Test(arguments: [false, true], [false, true])
    func recommendedSamplingProfileIsAccepted(stream: Bool, libraryMode: Bool) async throws {
        let backend = OptInThinkingBackend()
        let server = samplingServer(backend: backend, libraryMode: libraryMode)
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)
        var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "content-type")
        request.httpBody = Data("""
        {"model":"qwen36-alias","messages":[{"role":"user","content":"Reply with READY."}],
         "chat_template_kwargs":{"enable_thinking":true},"stream":\(stream),
         "temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,
         "presence_penalty":1.5,"repetition_penalty":1.0,"seed":777,"max_completion_tokens":32}
        """.utf8)
        let (data, response) = try await URLSession.shared.data(for: request)
        let received = await backend.received
        try await server.shutdown()
        #expect((response as? HTTPURLResponse)?.statusCode == 200)
        #expect(received?.reasoningEffort == .xhigh)
        #expect(received?.generationConfig.temperature == 1)
        #expect(received?.generationConfig.topP == 0.95)
        #expect(received?.generationConfig.topK == 20)
        #expect(received?.generationConfig.repetitionPenalty == 1)
        #expect(received?.generationConfig.seed == 777)
        #expect(received?.maximumCompletionTokens == 32)
        let text = String(decoding: data, as: UTF8.self)
        #expect(text.contains(#""content":"hello""#))
        #expect(!text.contains(#""error""#))
        if stream { #expect(text.hasSuffix("data: [DONE]\n\n")) }
    }

    @Test(arguments: [false, true], [false, true])
    func outOfRangeMinPIsRejectedOverHTTP(stream: Bool, libraryMode: Bool) async throws {
        let backend = OptInThinkingBackend()
        let server = samplingServer(backend: backend, libraryMode: libraryMode)
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)
        var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "content-type")
        request.httpBody = Data("""
        {"model":"qwen36-alias","messages":[{"role":"user","content":"hi"}],
         "stream":\(stream),"min_p":1.1}
        """.utf8)
        let (data, response) = try await URLSession.shared.data(for: request)
        let received = await backend.received
        try await server.shutdown()
        #expect((response as? HTTPURLResponse)?.statusCode == 400)
        #expect(received == nil)
        let text = String(decoding: data, as: UTF8.self)
        #expect(text.contains(#""param":"min_p""#))
        #expect(text.contains(#""code":"invalid_value""#))
        #expect(!text.contains("data: [DONE]"))
    }

    private func samplingServer(backend: OptInThinkingBackend,
                                libraryMode: Bool) -> MferenceHTTPServer {
        if libraryMode {
            let entry = ServerLibraryEntry(
                modelID: "qwen36-alias", familyModelID: "qwen3.6-35b-a3b",
                basename: "qwen36.gturbo",
                directory: URL(fileURLWithPath: "/unused/qwen36.gturbo"), family: .qwen36)
            let library = ServerModelLibrary(index: ServerLibraryIndex(entries: [entry])) { _ in backend }
            return MferenceHTTPServer(library: library, queueLimit: 1)
        }
        return MferenceHTTPServer(modelID: "qwen36-alias", queueLimit: 1,
                                  backend: backend, chatDialect: .chatml)
    }

    /// A library load refused because --max-context is above the model's
    /// native limit reaches the client as a 400 naming the limit, not a 500.
    @Test(arguments: [false, true])
    func contextAboveTheModelLimitReachesTheClient(stream: Bool) async throws {
        let refusal = ContextLimitError(family: .maple, requested: 262_144, maximum: 128_000)
        let entry = ServerLibraryEntry(
            modelID: "maple", familyModelID: "maple-preview-2bit-mlx",
            basename: "maple.gturbo",
            directory: URL(fileURLWithPath: "/unused/maple.gturbo"), family: .maple)
        let library = ServerModelLibrary(index: ServerLibraryIndex(entries: [entry])) { _ in
            throw refusal
        }
        let server = MferenceHTTPServer(library: library, queueLimit: 1)
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)
        var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "content-type")
        request.httpBody = Data("""
        {"model":"maple","messages":[{"role":"user","content":"hi"}],"stream":\(stream)}
        """.utf8)
        let (data, response) = try await URLSession.shared.data(for: request)
        try await server.shutdown()
        #expect((response as? HTTPURLResponse)?.statusCode == 400)
        let envelope = try JSONDecoder().decode(OpenAIErrorEnvelope.self, from: data)
        #expect(envelope.error.code == "context_exceeds_model")
        #expect(envelope.error.param == "model")
        #expect(envelope.error.message == refusal.description)
    }

    @Test(arguments: [false, true], [false, true])
    func penaltiesAndMinPReachTheBackend(stream: Bool, libraryMode: Bool) async throws {
        let backend = OptInThinkingBackend()
        let server = samplingServer(backend: backend, libraryMode: libraryMode)
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)
        var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "content-type")
        request.httpBody = Data("""
        {"model":"qwen36-alias","messages":[{"role":"user","content":"hi"}],
         "chat_template_kwargs":{"enable_thinking":true},"stream":\(stream),
         "temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.25,
         "presence_penalty":1.5,"frequency_penalty":0.75,"repeat_penalty":1.1,"repeat_last_n":32}
        """.utf8)
        let (_, response) = try await URLSession.shared.data(for: request)
        let received = await backend.received
        try await server.shutdown()
        #expect((response as? HTTPURLResponse)?.statusCode == 200)
        #expect(received?.generationConfig.presencePenalty == 1.5)
        #expect(received?.generationConfig.minP == 0.25)
        #expect(received?.generationConfig.frequencyPenalty == 0.75)
        #expect(received?.generationConfig.repetitionPenalty == 1.1)
        #expect(received?.generationConfig.repeatLastN == 32)
    }

    @Test(arguments: [#""presence_penalty":2.01"#, #""presence_penalty":-2.01"#,
                      #""presence_penalty":"NaN""#, #""min_p":"Infinity""#])
    func invalidSamplingFieldsFailBeforeStreaming(fields: String) async throws {
        let backend = OptInThinkingBackend()
        let server = samplingServer(backend: backend, libraryMode: false)
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)
        var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "content-type")
        request.httpBody = Data("""
        {"model":"qwen36-alias","messages":[{"role":"user","content":"hi"}],
         "stream":true,\(fields)}
        """.utf8)
        let (data, response) = try await URLSession.shared.data(for: request)
        let received = await backend.received
        try await server.shutdown()
        #expect((response as? HTTPURLResponse)?.statusCode == 400)
        #expect(received == nil)
        _ = try JSONDecoder().decode(OpenAIErrorEnvelope.self, from: data)
        #expect(!String(decoding: data, as: UTF8.self).contains("data: [DONE]"))
    }

    @Test(arguments: [false, true], [false, true])
    func qwen36ThinkingUsesBackendCapabilityWithCustomAlias(stream: Bool, kwargs: Bool) async throws {
        let backend = OptInThinkingBackend()
        let server = MferenceHTTPServer(modelID: "qwen36-alias", queueLimit: 1,
                                        backend: backend, chatDialect: .chatml)
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)
        var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "content-type")
        let control = kwargs
            ? #""chat_template_kwargs":{"enable_thinking":true,"preserve_thinking":true}"#
            : #""reasoning_effort":"medium""#
        request.httpBody = Data("""
        {"model":"qwen36-alias",\(control),"stream":\(stream),"messages":[
          {"role":"user","content":"A"},
          {"role":"assistant","content":"B","reasoning_content":"Check A"},
          {"role":"user","content":"C"}]}
        """.utf8)
        let (data, response) = try await URLSession.shared.data(for: request)
        #expect((response as? HTTPURLResponse)?.statusCode == 200)
        let text = String(decoding: data, as: UTF8.self)
        #expect(text.contains(#""reasoning_content":"Check first.""#))
        #expect(text.contains(#""content":"hello""#))
        if stream { #expect(text.hasSuffix("data: [DONE]\n\n")) }
        let received = await backend.received
        #expect(received?.reasoningEffort == (kwargs ? .xhigh : .medium))
        #expect(received?.messages[1].reasoningContent == "Check A")
        #expect(received?.maximumCompletionTokens == 32_768)
        try await server.shutdown()
    }

    /// A burst of connects waits in the listen queue until the accept loop
    /// drains it. With 16 slots a burst overflowed, which macOS 27 answers
    /// with RST (drumih/turbo-fieldfare#151, #153); the queue now matches
    /// NIO's own default of 128.
    @Test func listenBacklogAbsorbsAConnectBurst() async throws {
        let server = MferenceHTTPServer(modelID: "m", queueLimit: 1, backend: ScriptedServerBackend())
        let channel = try await server.start(port: 0)
        let backlog = try await channel.getOption(ChannelOptions.backlog).get()
        #expect(backlog >= 128)
        try await server.shutdown()
    }

    @Test(arguments: [false, true], [false, true])
    func swiftQwenReasoningUsesSeparateResponseFieldWithCustomModelAlias(stream: Bool, base: Bool) async throws {
        let server = MferenceHTTPServer(modelID: "custom-alias", queueLimit: 1,
                                        backend: ScriptedServerBackend(includeReasoning: true, baseQwen: base))
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)
        var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "content-type")
        request.httpBody = Data("""
        {"model":"custom-alias","reasoning_effort":"low","messages":[{"role":"user","content":"Hi"}],"stream":\(stream),"stream_options":{"include_usage":true}}
        """.utf8)
        let (data, response) = try await URLSession.shared.data(for: request)
        #expect((response as? HTTPURLResponse)?.statusCode == 200)
        let text = String(decoding: data, as: UTF8.self)
        #expect(text.contains(#""reasoning_content":"Check first.""#))
        #expect(text.contains(#""content":"hello""#))
        #expect(text.contains(#""completion_tokens_details""#))
        #expect(text.contains(#""visible_tokens":1"#))
        #expect(text.contains(#""reasoning_tokens":0"#))
        if stream { #expect(text.hasSuffix("data: [DONE]\n\n")) }
        try await server.shutdown()
    }
    @Test(arguments: [false, true])
    func healthModelsAndNonStreamingCompletion(includeDiagnostics: Bool) async throws {
        let server = MferenceHTTPServer(
            modelID: "test-model",
            queueLimit: 1,
            backend: ScriptedServerBackend(includeDiagnostics: includeDiagnostics))
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)

        let health = try await URLSession.shared.data(
            from: URL(string: "http://127.0.0.1:\(port)/health")!).0
        #expect(String(decoding: health, as: UTF8.self).contains(#""status":"ok""#))

        let models = try await URLSession.shared.data(
            from: URL(string: "http://127.0.0.1:\(port)/v1/models")!).0
        #expect(String(decoding: models, as: UTF8.self).contains("test-model"))

        var request = URLRequest(
            url: URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "content-type")
        request.httpBody = Data(#"""
        {"model":"test-model","messages":[{"role":"user","content":"hi"}]}
        """#.utf8)
        let (data, response) = try await URLSession.shared.data(for: request)
        #expect((response as? HTTPURLResponse)?.statusCode == 200)
        let object = try #require(
            JSONSerialization.jsonObject(with: data) as? [String: Any])
        let choices = try #require(object["choices"] as? [[String: Any]])
        let message = try #require(choices[0]["message"] as? [String: Any])
        #expect(message["content"] as? String == "hello")
        #expect(object["diagnostics"] == nil)
        #expect(!String(decoding: data, as: UTF8.self).contains("processRSS"))
        let usage = try #require(object["usage"] as? [String: Any])
        let details = try #require(usage["prompt_tokens_details"] as? [String: Any])
        #expect(details["cached_tokens"] as? Int == 0)

        try await server.shutdown()
    }

    /// Single-model mode reports the context its model was loaded with.
    @Test func singleModelListingReportsItsContext() async throws {
        let server = MferenceHTTPServer(modelID: "test-model", queueLimit: 1,
                                        backend: ScriptedServerBackend(), maxModelLen: 131_072)
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)
        let data = try await URLSession.shared.data(
            from: URL(string: "http://127.0.0.1:\(port)/v1/models")!).0
        try await server.shutdown()
        let object = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let models = try #require(object["data"] as? [[String: Any]])
        #expect(models.count == 1)
        #expect(models.first?["id"] as? String == "test-model")
        #expect(models.first?["max_model_len"] as? Int == 131_072)
    }

    @Test func routesIgnoreQueryComponentOfRequestURI() async throws {
        let server = MferenceHTTPServer(
            modelID: "test-model",
            queueLimit: 1,
            backend: ScriptedServerBackend())
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)

        for uri in ["/v1/models", "/v1/models?", "/v1/models?foo=bar"] {
            let response = try rawGET(uri: uri, port: port)
            #expect(response.contains("HTTP/1.1 200"), "\(uri) did not return 200")
            #expect(response.contains("test-model"), "\(uri) did not list the model")
        }
        let health = try rawGET(uri: "/health?probe=1", port: port)
        #expect(health.contains(#""status":"ok""#))

        let missing = try rawGET(uri: "/v1/nope?foo=bar", port: port)
        #expect(missing.contains("HTTP/1.1 404"))
        #expect(missing.contains("not_found"))

        try await server.shutdown()
    }

    @Test(arguments: [false, true])
    func streamingUsesStableShapeAndDoneMarker(includeDiagnostics: Bool) async throws {
        let server = MferenceHTTPServer(
            modelID: "test-model",
            queueLimit: 1,
            backend: ScriptedServerBackend(includeDiagnostics: includeDiagnostics))
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)
        var request = URLRequest(
            url: URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "content-type")
        request.httpBody = Data(#"""
        {"model":"test-model","messages":[{"role":"user","content":"hi"}],
         "stream":true,"stream_options":{"include_usage":true}}
        """#.utf8)
        let (data, response) = try await URLSession.shared.data(for: request)
        #expect((response as? HTTPURLResponse)?.statusCode == 200)
        let text = String(decoding: data, as: UTF8.self)
        #expect(text.contains(#""role":"assistant""#))
        #expect(text.contains(#""content":"hello""#))
        #expect(text.contains(#""finish_reason":"stop""#))
        #expect(text.contains(#""prompt_tokens":3"#))
        #expect(text.contains(#""cached_tokens":0"#))
        #expect(!text.contains("diagnostics"))
        #expect(!text.contains("processRSS"))
        #expect(text.hasSuffix("data: [DONE]\n\n"))

        try await server.shutdown()
    }

    @Test func streamingFailureAfterHeadReportsErrorInBand() async throws {
        let server = MferenceHTTPServer(
            modelID: "test-model",
            queueLimit: 1,
            backend: FailingMidStreamBackend())
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)
        var request = URLRequest(
            url: URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "content-type")
        request.httpBody = Data(#"""
        {"model":"test-model","messages":[{"role":"user","content":"hi"}],"stream":true}
        """#.utf8)
        let (data, response) = try await URLSession.shared.data(for: request)
        #expect((response as? HTTPURLResponse)?.statusCode == 200)
        let text = String(decoding: data, as: UTF8.self)
        #expect(text.contains(#""content":"partial""#))
        #expect(text.contains(#""code":"internal_error""#))
        #expect(text.contains(#""type":"server_error""#))
        #expect(!text.contains(#""finish_reason":"stop""#))
        #expect(text.hasSuffix("data: [DONE]\n\n"))

        try await server.shutdown()
    }

    @Test func streamingPrepareRejectionKeepsItsStatusCode() async throws {
        let server = MferenceHTTPServer(
            modelID: "test-model",
            queueLimit: 1,
            backend: RejectingPrepareBackend())
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)
        var request = URLRequest(
            url: URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "content-type")
        request.httpBody = Data(#"""
        {"model":"test-model","messages":[{"role":"user","content":"hi"}],"stream":true}
        """#.utf8)
        let (data, response) = try await URLSession.shared.data(for: request)
        let httpResponse = try #require(response as? HTTPURLResponse)
        // Rejected before the head, so the stream was never opened.
        #expect(httpResponse.statusCode == 400)
        #expect(httpResponse.value(forHTTPHeaderField: "content-type") == "application/json")
        let text = String(decoding: data, as: UTF8.self)
        #expect(text.contains(#""code":"context_length_exceeded""#))
        #expect(!text.contains("data:"))

        try await server.shutdown()
    }

    @Test func streamingRejectionAfterHeadNamesTheCause() async throws {
        let server = MferenceHTTPServer(
            modelID: "test-model",
            queueLimit: 1,
            backend: RejectingBackend())
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)
        var request = URLRequest(
            url: URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "content-type")
        request.httpBody = Data(#"""
        {"model":"test-model","messages":[{"role":"user","content":"hi"}],"stream":true}
        """#.utf8)
        // A dropped connection fails this call: the body must end normally.
        let (data, _) = try await URLSession.shared.data(for: request)
        let text = String(decoding: data, as: UTF8.self)
        #expect(text.contains(#""code":"context_length_exceeded""#))
        #expect(text.contains(#""param":"messages""#))
        #expect(!text.contains(#""finish_reason":"stop""#))
        #expect(text.hasSuffix("data: [DONE]\n\n"))

        // The same request without a stream still gets a real status code.
        request.httpBody = Data(#"""
        {"model":"test-model","messages":[{"role":"user","content":"hi"}]}
        """#.utf8)
        let (blocking, blockingResponse) = try await URLSession.shared.data(for: request)
        #expect((blockingResponse as? HTTPURLResponse)?.statusCode == 400)
        #expect(String(decoding: blocking, as: UTF8.self)
            .contains(#""code":"context_length_exceeded""#))

        try await server.shutdown()
    }

    @Test func wrongModelUsesOpenAIErrorEnvelope() async throws {
        let server = MferenceHTTPServer(
            modelID: "test-model",
            queueLimit: 1,
            backend: ScriptedServerBackend())
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)
        var request = URLRequest(
            url: URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "content-type")
        request.httpBody = Data(#"""
        {"model":"wrong","messages":[{"role":"user","content":"hi"}]}
        """#.utf8)
        let (data, response) = try await URLSession.shared.data(for: request)
        #expect((response as? HTTPURLResponse)?.statusCode == 404)
        #expect(String(decoding: data, as: UTF8.self).contains("model_not_found"))

        try await server.shutdown()
    }

    @Test func streamingHeartbeatKeepsSlowFirstEventAlive() async throws {
        let server = MferenceHTTPServer(
            modelID: "test-model",
            queueLimit: 1,
            backend: ScriptedServerBackend(delayNanoseconds: 50_000_000),
            heartbeatInterval: .milliseconds(10))
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)
        var request = URLRequest(
            url: URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "content-type")
        request.httpBody = Data(#"""
        {"model":"test-model","messages":[{"role":"user","content":"hi"}],"stream":true}
        """#.utf8)
        let data = try await URLSession.shared.data(for: request).0
        #expect(String(decoding: data, as: UTF8.self).contains(": ping\n\n"))

        try await server.shutdown()
    }

    @Test func streamingMultipleToolsUseDistinctIndexes() async throws {
        let server = MferenceHTTPServer(
            modelID: "test-model",
            queueLimit: 1,
            backend: MultipleToolBackend())
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)
        var request = URLRequest(
            url: URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "content-type")
        request.httpBody = Data(#"""
        {"model":"test-model","messages":[{"role":"user","content":"read both"}],
         "stream":true}
        """#.utf8)
        let text = String(decoding: try await URLSession.shared.data(for: request).0,
                          as: UTF8.self)
        #expect(text.contains(#""index":0"#))
        #expect(text.contains(#""index":1"#))
        #expect(text.contains(#""finish_reason":"tool_calls""#))

        request.httpBody = Data(#"""
        {"model":"test-model","messages":[{"role":"user","content":"read both"}]}
        """#.utf8)
        let data = try await URLSession.shared.data(for: request).0
        let object = try #require(
            JSONSerialization.jsonObject(with: data) as? [String: Any])
        let choices = try #require(object["choices"] as? [[String: Any]])
        let message = try #require(choices[0]["message"] as? [String: Any])
        #expect(message["content"] is NSNull)
        #expect((message["tool_calls"] as? [[String: Any]])?.count == 2)

        try await server.shutdown()
    }

    @Test func nonStreamingToolCallRetainsVisibleContent() async throws {
        let server = MferenceHTTPServer(
            modelID: "test-model",
            queueLimit: 1,
            backend: ContentAndToolBackend())
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)
        var request = URLRequest(
            url: URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "content-type")
        request.httpBody = Data(#"""
        {"model":"test-model","messages":[{"role":"user","content":"read"}]}
        """#.utf8)

        let data = try await URLSession.shared.data(for: request).0
        let object = try #require(
            JSONSerialization.jsonObject(with: data) as? [String: Any])
        let choices = try #require(object["choices"] as? [[String: Any]])
        let message = try #require(choices[0]["message"] as? [String: Any])
        #expect(message["content"] as? String == "I will read it.")
        #expect((message["tool_calls"] as? [[String: Any]])?.count == 1)
        #expect(choices[0]["finish_reason"] as? String == "tool_calls")

        request.httpBody = Data(#"""
        {"model":"test-model","messages":[{"role":"user","content":"read"}],"stream":true}
        """#.utf8)
        let stream = String(
            decoding: try await URLSession.shared.data(for: request).0,
            as: UTF8.self)
        #expect(stream.contains(#""content":"I will read it.""#))
        #expect(stream.contains(#""tool_calls""#))
        #expect(stream.contains(#""finish_reason":"tool_calls""#))

        try await server.shutdown()
    }

    @Test func pipelinedStreamingThenHealthResponsesRemainOrdered() async throws {
        let backend = PipelinedRequestBackend()
        let server = MferenceHTTPServer(
            modelID: "test-model",
            queueLimit: 1,
            backend: backend,
            heartbeatInterval: .seconds(10))
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)
        let socket = try connectedSocket(port: port)

        let body = #"{"model":"test-model","messages":[{"role":"user","content":"hi"}],"stream":true}"#
        let firstRequest =
            "POST /v1/chat/completions HTTP/1.1\r\n"
            + "Host: 127.0.0.1:\(port)\r\n"
            + "Content-Type: application/json\r\n"
            + "Content-Length: \(body.utf8.count)\r\n"
            + "Connection: keep-alive\r\n"
            + "\r\n"
            + body
        let secondRequest =
            "GET /health HTTP/1.1\r\n"
            + "Host: 127.0.0.1:\(port)\r\n"
            + "Connection: close\r\n"
            + "\r\n"
        try writeAll(socket: socket, text: firstRequest + secondRequest)
        let waitDeadline = ContinuousClock.now + .seconds(2)
        while await !backend.isWaiting, ContinuousClock.now < waitDeadline {
            try await Task.sleep(for: .milliseconds(1))
        }
        #expect(await backend.isWaiting)

        var response = try readAvailable(socket: socket, timeoutMilliseconds: 200)
        #expect(response.contains("text/event-stream"))
        #expect(response.components(separatedBy: "HTTP/1.1 200").count - 1 == 1)
        #expect(!response.contains(#""status":"ok""#))

        await backend.release()
        response += try readUntil(
            socket: socket,
            timeoutMilliseconds: 2_000,
            condition: { $0.contains(#""status":"ok""#) })
        #expect(response.components(separatedBy: "HTTP/1.1 200").count - 1 == 2)
        let done = try #require(response.range(of: "data: [DONE]"))
        let health = try #require(response.range(of: #""status":"ok""#))
        #expect(done.lowerBound < health.lowerBound)
        #expect(await backend.generationCount == 1)

        Darwin.close(socket)
        try await server.shutdown()
    }

    @Test func queueOverflowIsRejectedBeforeTheRequestIsRendered() async throws {
        let backend = RenderCountingBackend()
        let server = MferenceHTTPServer(
            modelID: "test-model",
            queueLimit: 1,
            backend: backend)
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)
        let sockets = try (0..<3).map { _ in try connectedSocket(port: port) }
        defer { sockets.forEach { Darwin.close($0) } }
        let body =
            #"{"model":"test-model","messages":[{"role":"user","content":"wait"}]}"#
        let request =
            "POST /v1/chat/completions HTTP/1.1\r\n"
            + "Host: 127.0.0.1:\(port)\r\n"
            + "Content-Type: application/json\r\n"
            + "Content-Length: \(body.utf8.count)\r\n"
            + "Connection: keep-alive\r\n"
            + "\r\n"
            + body

        // One generating, one queued: the gate is now full.
        try writeAll(socket: sockets[0], text: request)
        try writeAll(socket: sockets[1], text: request)
        let deadline = ContinuousClock.now + .seconds(2)
        while await server.queuedRequestCount != 1, ContinuousClock.now < deadline {
            await Task.yield()
        }
        #expect(await server.queuedRequestCount == 1)
        #expect(await backend.prepareCount == 2)

        try writeAll(socket: sockets[2], text: request)
        let response = try readUntil(
            socket: sockets[2],
            timeoutMilliseconds: 2_000,
            condition: { $0.contains("queue_full") })
        #expect(response.contains("HTTP/1.1 429"))
        // The rejected request must never have reached the tokenizer.
        #expect(await backend.prepareCount == 2)

        await backend.releaseAll()
        try await server.shutdown()
    }

    @Test func shutdownAfterListenerClosesIsIdempotent() async throws {
        let server = MferenceHTTPServer(
            modelID: "test-model",
            queueLimit: 1,
            backend: ScriptedServerBackend())
        let channel = try await server.start(port: 0)

        try await channel.close().get()
        try await server.shutdown()
        try await server.shutdown()
    }

    @Test func shutdownCancelsActiveAndQueuedRequestsBeforeReturning() async throws {
        let backend = CancellableServerBackend()
        let server = MferenceHTTPServer(
            modelID: "test-model",
            queueLimit: 1,
            backend: backend)
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)
        let firstSocket = try connectedSocket(port: port)
        let secondSocket = try connectedSocket(port: port)
        defer {
            Darwin.close(firstSocket)
            Darwin.close(secondSocket)
        }
        let body =
            #"{"model":"test-model","messages":[{"role":"user","content":"wait"}]}"#
        let request =
            "POST /v1/chat/completions HTTP/1.1\r\n"
            + "Host: 127.0.0.1:\(port)\r\n"
            + "Content-Type: application/json\r\n"
            + "Content-Length: \(body.utf8.count)\r\n"
            + "Connection: keep-alive\r\n"
            + "\r\n"
            + body

        try writeAll(socket: firstSocket, text: request)
        let activeDeadline = ContinuousClock.now + .seconds(2)
        while await backend.startedCount != 1, ContinuousClock.now < activeDeadline {
            await Task.yield()
        }
        #expect(await backend.startedCount == 1)

        try writeAll(socket: secondSocket, text: request)
        let queuedDeadline = ContinuousClock.now + .seconds(2)
        while await server.queuedRequestCount != 1, ContinuousClock.now < queuedDeadline {
            await Task.yield()
        }
        #expect(await server.queuedRequestCount == 1)
        #expect(await server.acceptedConnectionCount == 2)

        try await server.shutdown()

        #expect(await backend.cancellationCount == 1)
        #expect(await backend.startedCount == 1)
        #expect(await server.queuedRequestCount == 0)
        #expect(await !server.hasActiveRequest)
        #expect(await server.acceptedConnectionCount == 0)
        try await server.shutdown()
    }
}

private enum RawSocketError: Error {
    case systemCall(String, Int32)
    case timeout
}

private func connectedSocket(port: Int) throws -> Int32 {
    let descriptor = Darwin.socket(AF_INET, SOCK_STREAM, 0)
    guard descriptor >= 0 else {
        throw RawSocketError.systemCall("socket", errno)
    }
    var address = sockaddr_in()
    address.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
    address.sin_family = sa_family_t(AF_INET)
    address.sin_port = in_port_t(port).bigEndian
    address.sin_addr = in_addr(s_addr: inet_addr("127.0.0.1"))
    let result = withUnsafePointer(to: &address) {
        $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
            Darwin.connect(descriptor, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
        }
    }
    guard result == 0 else {
        let code = errno
        Darwin.close(descriptor)
        throw RawSocketError.systemCall("connect", code)
    }
    return descriptor
}

private func writeAll(socket: Int32, text: String) throws {
    let bytes = Array(text.utf8)
    var written = 0
    while written < bytes.count {
        let count = bytes.withUnsafeBytes {
            Darwin.send(socket, $0.baseAddress!.advanced(by: written),
                        bytes.count - written, 0)
        }
        guard count > 0 else {
            throw RawSocketError.systemCall("send", errno)
        }
        written += count
    }
}

/// Sends a request line verbatim so the exact URI reaches the router, which
/// `URLSession` would otherwise normalise.
private func rawGET(uri: String, port: Int) throws -> String {
    let socket = try connectedSocket(port: port)
    defer { Darwin.close(socket) }
    try writeAll(socket: socket,
                 text: "GET \(uri) HTTP/1.1\r\n"
                     + "Host: 127.0.0.1:\(port)\r\n"
                     + "Connection: close\r\n"
                     + "\r\n")
    return try readAvailable(socket: socket, timeoutMilliseconds: 500)
}

private func readAvailable(socket: Int32, timeoutMilliseconds: Int32) throws -> String {
    var result: [UInt8] = []
    var descriptor = pollfd(fd: socket, events: Int16(POLLIN), revents: 0)
    while Darwin.poll(&descriptor, 1, timeoutMilliseconds) > 0 {
        var buffer = [UInt8](repeating: 0, count: 4_096)
        let count = Darwin.recv(socket, &buffer, buffer.count, 0)
        guard count >= 0 else {
            throw RawSocketError.systemCall("recv", errno)
        }
        if count == 0 { break }
        result.append(contentsOf: buffer.prefix(count))
        descriptor.revents = 0
    }
    return String(decoding: result, as: UTF8.self)
}

private func readUntil(
    socket: Int32,
    timeoutMilliseconds: Int32,
    condition: (String) -> Bool
) throws -> String {
    let deadline = Date().addingTimeInterval(Double(timeoutMilliseconds) / 1_000)
    var result = ""
    while Date() < deadline {
        result += try readAvailable(socket: socket, timeoutMilliseconds: 50)
        if condition(result) { return result }
    }
    throw RawSocketError.timeout
}

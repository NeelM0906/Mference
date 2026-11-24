import Foundation
import Metal
import Synchronization
import Testing
@testable import Mference
@testable import MferenceEngine
@testable import MferenceServerCore

// K0's acceptance fixtures F01, F02, F03 and F06
// (`Scripts/parity/inkling2/server-fixtures/`) through the HTTP server in
// library mode, model-free. Everything but the logits is production code:
// the request validator, the engine bridge, `NativeEngineModel` (render with
// the vendor's chat template, `runRawCompletion`, the structured decoder,
// usage), the host and the SSE writer, over the Inkling 2 tokenizer fixture
// (the 60 special tokens at their real ids). A scripted runner makes the
// model "generate" the fixture's `model_generates` text, token by token.

/// Emits `script` one token per decode step once the prompt is prefilled.
/// The prompt length comes from the engine's own `prepare` (`RecordingModel`)
/// just before generation, so the runner needs no knowledge of the render.
/// Prefill steps whose logits are never sampled are not written.
final class ScriptedInkling2Runner: ContinuableLogitProducer, @unchecked Sendable {
    let vocabSize: Int
    let eos: Int32
    private let state = Mutex<(promptLength: Int, script: [Int32], inputs: [Int32], position: Int)>(
        (0, [], [], 0))

    init(vocabSize: Int, eos: Int32) {
        self.vocabSize = vocabSize
        self.eos = eos
    }

    /// The next request generates `script` (then EOS).
    func plan(script: [Int32]) {
        state.withLock { $0.script = script; $0.inputs = [] }
    }

    /// Set by `RecordingModel.prepare`: the prompt generation prefills.
    func prepared(promptLength: Int) {
        state.withLock { $0.promptLength = promptLength; $0.inputs = [] }
    }

    /// Every token the runner was fed: the prompt, then the generated ones.
    var inputs: [Int32] { state.withLock { $0.inputs } }

    var continuationPosition: Int { state.withLock { $0.position } }
    func prepareForContinuation(expectedPosition: Int) throws {}
    func reset() { state.withLock { $0.position = 0 } }

    func produce(token: Int32, position: Int, into logits: MTLBuffer) async throws {
        let target: Int32? = state.withLock { s in
            s.inputs.append(token)
            s.position = position + 1
            let index = position + 1 - s.promptLength
            if index < 0 { return nil }   // a prefill step: these logits are not sampled
            return index < s.script.count ? s.script[index] : eos
        }
        guard let target else { return }
        let values = logits.contents().bindMemory(to: Float16.self, capacity: vocabSize)
        for i in 0..<vocabSize { values[i] = Float16(-30.0) }
        values[Int(target)] = Float16(30.0)
    }
}

/// Serves one prepared model, and records each request the server hands it.
struct OneModelBackend: EngineBackend {
    let model: RecordingModel
    var engine: EngineKind { .mference }
    func load(_ descriptor: EngineModelDescriptor, options: EngineLoadOptions) async throws -> any EngineModel {
        model
    }
}

/// The real model, plus a record of each request and its rendered prompt,
/// which it also hands the scripted runner before generation starts.
final class RecordingModel: EngineModel, @unchecked Sendable {
    let inner: NativeEngineModel
    let runner: ScriptedInkling2Runner
    private let recorded = Mutex<[EngineRequest]>([])
    private let prompts = Mutex<[[Int32]]>([])
    init(_ inner: NativeEngineModel, runner: ScriptedInkling2Runner) {
        self.inner = inner
        self.runner = runner
    }
    var descriptor: EngineModelDescriptor { inner.descriptor }
    var traits: EngineModelTraits { inner.traits }
    var requests: [EngineRequest] { recorded.withLock { $0 } }
    var lastPrompt: [Int32] { prompts.withLock { $0.last ?? [] } }
    func prepare(_ request: EngineRequest) async throws -> EnginePreparedPrompt {
        recorded.withLock { $0.append(request) }
        let prepared = try await inner.prepare(request)
        prompts.withLock { $0.append(prepared.promptTokens) }
        runner.prepared(promptLength: prepared.promptTokens.count)
        return prepared
    }
    func generate(_ prepared: EnginePreparedPrompt,
                  onEvent: @escaping @Sendable (EngineEvent) -> Void) async throws -> EngineCompletion {
        try await inner.generate(prepared, onEvent: onEvent)
    }
    func unload() async { await inner.unload() }
}

@Suite("Inkling 2 through the server (K0 fixtures F01-F04, F06)", .serialized)
struct Inkling2ServerTests {
    static let modelID = "inkling-2-mini-ud-q4_k_xl"
    static let mdl = "<|message_model|>", txt = "<|content_text|>", thk = "<|content_thinking|>"
    static let end = "<|end_message|>", eos = "<|content_model_end_sampling|>"
    static let invoke = "<|content_invoke_tool_json|>"

    static func fixtureFolder(_ name: String = "Inkling2Tokenizer") -> URL {
        var root = URL(fileURLWithPath: #filePath)
        for _ in 0..<3 { root.deleteLastPathComponent() }   // Tests/MferenceServer/<file> -> repo root
        return root.appendingPathComponent("Tests/Mference/Core/Tokenization/Fixtures/\(name)")
    }

    /// A running server over one scripted Inkling 2 model.
    final class Harness: @unchecked Sendable {
        let tokenizer: MFTokenizer
        let runner: ScriptedInkling2Runner
        let model: RecordingModel
        let server: MferenceHTTPServer
        let port: Int

        /// Inkling 2 by default; another tokenizer fixture serves as another
        /// dialect, to pin behaviour that must not change for it.
        init(fixture: String = "Inkling2Tokenizer", family: ModelFamily = .inkling2Loop,
             modelID: String = Inkling2ServerTests.modelID, product: EngineProductSlot? = .inkling2Mini,
             eosText: String = Inkling2ServerTests.eos, maxContext: Int = 4_096,
             promptCacheMode: EnginePromptCacheMode = .off) async throws {
            let folder = Inkling2ServerTests.fixtureFolder(fixture)
            // As `NativeBackend` loads it: `load(forModelDirectory:)` is this
            // load plus the checkpoint profile (the vendor's sampling defaults).
            tokenizer = try await MFTokenizer.load(from: folder, family: family).forCheckpoint(modelID)
            let eos = try #require(tokenizer.encode(eosText, addBOS: false).first)
            runner = ScriptedInkling2Runner(vocabSize: tokenizer.vocabSize, eos: eos)
            let descriptor = EngineModelDescriptor(
                id: modelID, displayName: modelID, engine: .mference,
                format: .gturbo(family: family, checkpointID: nil),
                location: folder, sizeBytes: 1, contextLength: family.maximumContext, product: product)
            let native = try NativeEngineModel.forTesting(descriptor: descriptor, context: try MetalContext(),
                                                          tokenizer: tokenizer, runner: runner,
                                                          maxContext: maxContext,
                                                          promptCacheMode: promptCacheMode)
            model = RecordingModel(native, runner: runner)
            let library = ServerModelLibrary(catalog: EngineCatalog(models: [descriptor]),
                                             backends: [OneModelBackend(model: model)],
                                             loadOptions: EngineLoadOptions(nativeMaxContext: maxContext))
            server = MferenceHTTPServer(library: library, queueLimit: 1)
            let channel = try await server.start(port: 0)
            guard let port = channel.localAddress?.port else {
                try? await server.shutdown()
                throw URLError(.cannotConnectToHost)
            }
            self.port = port
        }

        func post(_ body: [String: Any]) async throws -> (status: Int, text: String) {
            var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!)
            request.httpMethod = "POST"
            request.setValue("application/json", forHTTPHeaderField: "content-type")
            request.httpBody = try JSONSerialization.data(withJSONObject: body)
            let (data, response) = try await URLSession.shared.data(for: request)
            return ((response as? HTTPURLResponse)?.statusCode ?? 0, String(decoding: data, as: UTF8.self))
        }

        /// Sends `body` with the model generating `generates`, and checks that
        /// the runner was fed exactly the rendered prompt, then the script's
        /// tokens in order (a misaligned script would show here).
        func run(_ body: [String: Any], generates: String) async throws
            -> (status: Int, text: String, prompt: [Int32]) {
            let script = tokenizer.encode(generates, addBOS: false)
            runner.plan(script: script)
            let answer = try await post(body)
            let prompt = model.lastPrompt
            let fed = runner.inputs
            #expect(Array(fed.prefix(prompt.count)) == prompt, "the runner was not fed the rendered prompt")
            let decoded = Array(fed.dropFirst(prompt.count))
            #expect(decoded == Array(script.prefix(decoded.count)), "decode steps out of step with the script")
            return (answer.status, answer.text, prompt)
        }

        func close() async throws { try await server.shutdown() }
    }

    /// Runs `body` with a fresh harness and always shuts its server down,
    /// awaited, before returning or rethrowing.
    static func withHarness(fixture: String = "Inkling2Tokenizer", family: ModelFamily = .inkling2Loop,
                            modelID: String = Inkling2ServerTests.modelID,
                            product: EngineProductSlot? = .inkling2Mini,
                            eosText: String = Inkling2ServerTests.eos,
                            promptCacheMode: EnginePromptCacheMode = .off,
                            _ body: (Harness) async throws -> Void) async throws {
        let h = try await Harness(fixture: fixture, family: family, modelID: modelID, product: product,
                                  eosText: eosText, promptCacheMode: promptCacheMode)
        do {
            try await body(h)
        } catch {
            try? await h.close()
            throw error
        }
        try await h.close()
    }

    /// The `data:` payloads of an SSE body, the final `[DONE]` as a string.
    static func events(_ text: String) -> [Any] {
        text.components(separatedBy: "\n").compactMap { line -> Any? in
            guard line.hasPrefix("data: ") else { return nil }
            let payload = String(line.dropFirst(6))
            if payload == "[DONE]" { return payload }
            return try? JSONSerialization.jsonObject(with: Data(payload.utf8))
        }
    }

    /// What one streamed response says, chunk by chunk.
    struct Stream {
        var roles: [String] = []
        var reasoning = ""
        var content = ""
        var toolCalls: [[String: Any]] = []
        var finishReasons: [String] = []
        var usage: [String: Any]?
        var models = Set<String>()
        var objects = Set<String>()
        var errors = 0
        var done = false
        var deltaKeysInOrder: [String] = []
        /// Every chunk carried `id`, `created` and `object`, a choice had
        /// index 0, and the usage chunk had no choices.
        var shapeOK = true

        init(_ text: String) {
            for event in Inkling2ServerTests.events(text) {
                if let marker = event as? String { done = marker == "[DONE]"; continue }
                guard let chunk = event as? [String: Any] else { continue }
                if chunk["error"] != nil { errors += 1; continue }
                if chunk["id"] as? String == nil || chunk["created"] as? Int == nil
                    || chunk["object"] as? String == nil || chunk["model"] as? String == nil {
                    shapeOK = false
                }
                if chunk["usage"] is [String: Any], (chunk["choices"] as? [Any])?.isEmpty != true { shapeOK = false }
                if let model = chunk["model"] as? String { models.insert(model) }
                if let object = chunk["object"] as? String { objects.insert(object) }
                if let usage = chunk["usage"] as? [String: Any] { self.usage = usage }
                for choice in chunk["choices"] as? [[String: Any]] ?? [] {
                    if choice["index"] as? Int != 0 { shapeOK = false }
                    if let reason = choice["finish_reason"] as? String { finishReasons.append(reason) }
                    guard let delta = choice["delta"] as? [String: Any] else { continue }
                    if let role = delta["role"] as? String { roles.append(role); deltaKeysInOrder.append("role") }
                    if let r = delta["reasoning_content"] as? String, !r.isEmpty {
                        reasoning += r
                        if deltaKeysInOrder.last != "reasoning" { deltaKeysInOrder.append("reasoning") }
                    }
                    if let c = delta["content"] as? String, !c.isEmpty {
                        content += c
                        if deltaKeysInOrder.last != "content" { deltaKeysInOrder.append("content") }
                    }
                    for call in delta["tool_calls"] as? [[String: Any]] ?? [] {
                        toolCalls.append(call)
                        deltaKeysInOrder.append("tool_call")
                    }
                }
            }
        }
    }

    static let framing = ["<|message_model|>", "<|content_", "<|end_message|>", "<|message_"]

    static func base(_ messages: [[String: Any]], maxTokens: Int = 512, seed: Int = 1,
                     tools: [[String: Any]]? = nil) -> [String: Any] {
        var body: [String: Any] = ["model": modelID, "stream": true, "stream_options": ["include_usage": true],
                                   "max_completion_tokens": maxTokens, "seed": seed, "messages": messages]
        if let tools { body["tools"] = tools }
        return body
    }

    static func tool(_ name: String, _ description: String) -> [String: Any] {
        ["type": "function", "function": [
            "name": name, "description": description,
            "parameters": ["type": "object", "properties": ["city": ["type": "string"]], "required": ["city"]],
        ] as [String: Any]]
    }

    /// F01: thinking then text; reasoning and content on their own channels,
    /// usage with the thinking payload's token count.
    @Test func f01ThinkingThenText() async throws {
        try await Self.withHarness { h in
            #expect(h.tokenizer.dialect == .inkling2)
            #expect(h.tokenizer.encode(Self.eos, addBOS: false) == [200_006])
            let thinking = "The user asks for a capital."
            let generates = Self.mdl + Self.thk + thinking + Self.end + Self.mdl + Self.txt + "Paris." + Self.end
                + Self.eos
            let question = "What is the capital of France?"
            let (status, text, prompt) = try await h.run(
                Self.base([["role": "user", "content": question]]), generates: generates)
            #expect(status == 200, "\(text)")
            let s = Stream(text)
            #expect(s.roles == ["assistant"] && s.done && s.errors == 0 && s.shapeOK)
            #expect(s.deltaKeysInOrder == ["role", "reasoning", "content"])
            #expect(s.reasoning == thinking && s.content == "Paris.")
            #expect(s.finishReasons == ["stop"])
            #expect(s.models == [Self.modelID] && s.objects == ["chat.completion.chunk"])
            for framing in Self.framing { #expect(!s.content.contains(framing) && !s.reasoning.contains(framing)) }
            let usage = try #require(s.usage)
            #expect(usage["prompt_tokens"] as? Int == prompt.count)
            let generated = h.tokenizer.encode(generates, addBOS: false)
            #expect(usage["completion_tokens"] as? Int == generated.count)
            let details = try #require(usage["completion_tokens_details"] as? [String: Any])
            #expect(details["reasoning_tokens"] as? Int == h.tokenizer.encode(thinking, addBOS: false).count)
            // The request reached the engine as the fixture says: one user
            // message, no tools, no effort; the template's default level 0.9
            // is rendered, and the prompt is the one the template gives.
            let request = try #require(h.model.requests.last)
            #expect(request.messages.count == 1 && request.messages.first?.content == question)
            #expect(request.tools.isEmpty && request.reasoningEffort == nil)
            #expect(prompt == (try h.tokenizer.encodeChat(messages: request.messages, tools: [],
                                                         reasoningEffort: nil,
                                                         preserveThinking: request.preserveThinking)))
            let rendered = h.tokenizer.decode(prompt, skipSpecialTokens: false)
            #expect(rendered.contains("<|message_system|><|content_text|>Thinking effort level: 0.9"), "\(rendered)")
            #expect(rendered.contains("<|message_user|><|content_text|>" + question))
        }
    }

    /// F02: two tool calls in one turn: indexes 0 and 1, distinct ids,
    /// arguments as the args object's JSON, `tool_calls` finish, no content.
    @Test func f02TwoToolCalls() async throws {
        try await Self.withHarness { h in
            let thinking = "Two tools needed."
            let generates = Self.mdl + Self.thk + thinking + Self.end
                + Self.mdl + "get_weather" + Self.invoke + #"{"name":"get_weather","args":{"city":"Paris"}}"#
                + Self.end
                + Self.mdl + "get_time" + Self.invoke + #"{"name":"get_time","args":{"city":"Paris"}}"# + Self.end
                + Self.eos
            let body = Self.base([["role": "user", "content": "Weather and time in Paris?"]],
                                 tools: [Self.tool("get_weather", "Current weather for a city."),
                                         Self.tool("get_time", "Local time in a city.")])
            let (status, text, _) = try await h.run(body, generates: generates)
            #expect(status == 200, "\(text)")
            let s = Stream(text)
            #expect(s.errors == 0 && s.done && s.shapeOK)
            #expect(s.deltaKeysInOrder == ["role", "reasoning", "tool_call", "tool_call"])
            #expect(s.reasoning == thinking)
            #expect(s.content.isEmpty)
            #expect(s.finishReasons == ["tool_calls"])
            #expect(s.toolCalls.count == 2)
            #expect(s.toolCalls.map { $0["index"] as? Int } == [0, 1])
            let ids = s.toolCalls.compactMap { $0["id"] as? String }
            #expect(ids.count == 2 && Set(ids).count == 2 && ids.allSatisfy { !$0.isEmpty })
            let functions = s.toolCalls.compactMap { $0["function"] as? [String: Any] }
            #expect(functions.map { $0["name"] as? String } == ["get_weather", "get_time"])
            for function in functions {
                let arguments = try #require(function["arguments"] as? String)
                let parsed = try #require(
                    try JSONSerialization.jsonObject(with: Data(arguments.utf8)) as? [String: String])
                #expect(parsed == ["city": "Paris"])
            }
            #expect(s.toolCalls.allSatisfy { $0["type"] as? String == "function" })
            #expect(!s.content.contains("get_weather") && !s.content.contains("get_time"))
            let usage = try #require(s.usage)
            #expect(usage["completion_tokens"] as? Int == h.tokenizer.encode(generates, addBOS: false).count)
            let details = try #require(usage["completion_tokens_details"] as? [String: Any])
            #expect(details["reasoning_tokens"] as? Int == h.tokenizer.encode(thinking, addBOS: false).count)
        }
    }

    /// F03: the client's history shape (empty content beside tool calls,
    /// arguments as a JSON string, a tool message with only its call id,
    /// reasoning in history) renders the vendor's fragments in order and no
    /// empty text block.
    @Test func f03ToolResultsBackWithThinkingInHistory() async throws {
        try await Self.withHarness { h in
            let messages: [[String: Any]] = [
                ["role": "user", "content": "Weather in Paris?"],
                ["role": "assistant", "content": "", "reasoning_content": "Need the weather.",
                 "tool_calls": [["id": "call_1", "type": "function",
                                 "function": ["name": "get_weather", "arguments": #"{"city":"Paris"}"#]]]],
                ["role": "tool", "tool_call_id": "call_1", "content": #"{"temp_c": 18, "sky": "cloudy"}"#],
            ]
            let answer = "It is 18 C and cloudy in Paris."
            let (status, text, prompt) = try await h.run(
                Self.base(messages, tools: [Self.tool("get_weather", "Current weather for a city.")]),
                generates: Self.mdl + Self.txt + answer + Self.end + Self.eos)
            #expect(status == 200, "\(text)")
            let s = Stream(text)
            #expect(s.roles == ["assistant"] && s.errors == 0 && s.done && s.shapeOK)
            #expect(s.content == answer && s.finishReasons == ["stop"])
            #expect(s.reasoning.isEmpty && s.toolCalls.isEmpty)
            let details = s.usage?["completion_tokens_details"] as? [String: Any]
            #expect((details?["reasoning_tokens"] as? Int ?? 0) == 0)
            let rendered = h.tokenizer.decode(prompt, skipSpecialTokens: false)
            let fragments = [
                Self.mdl + Self.thk + "Need the weather." + Self.end,
                Self.mdl + "get_weather" + Self.invoke + #"{"name":"get_weather","args":{"city":"Paris"}}"#
                    + Self.end + Self.eos,
                "<|message_tool|>get_weather" + Self.txt + #"{"temp_c": 18, "sky": "cloudy"}"# + Self.end,
            ]
            var cursor = rendered.startIndex
            for fragment in fragments {
                let found = try #require(rendered.range(of: fragment, range: cursor..<rendered.endIndex),
                                         "missing in order: \(fragment)\n\(rendered)")
                cursor = found.upperBound
            }
            #expect(!rendered.contains(Self.mdl + Self.txt + Self.end), "empty text block rendered")
            // The history reached the engine as the template needs it.
            let request = try #require(h.model.requests.last)
            #expect(request.messages.map(\.role) == [.user, .assistant, .tool])
            let assistant = request.messages[1], tool = request.messages[2]
            #expect(assistant.content == nil || assistant.content == "")
            #expect(assistant.reasoningContent == "Need the weather.")
            #expect(assistant.toolCalls.map(\.name) == ["get_weather"] && assistant.toolCalls.first?.id == "call_1")
            #expect(tool.toolCallID == "call_1" && tool.name == nil)
        }
    }

    /// F06: the product sampling defaults and the seed reach the generation
    /// loop; the completion cap ends with `length`; EOS 200006 ends with
    /// `stop` and nothing after it is emitted.
    @Test func f06SamplingDefaultsSeedCapAndEOS() async throws {
        try await Self.withHarness { h in
            let hi: [[String: Any]] = [["role": "user", "content": "Hi"]]

            let (status1, text1, _) = try await h.run(
                Self.base(hi, seed: 7), generates: Self.mdl + Self.txt + "Hello." + Self.end + Self.eos)
            #expect(status1 == 200, "\(text1)")
            let config = try #require(h.model.requests.last?.generationConfig)
            // No truncation: the server writes an absent top_p as 1.0 for every
            // dialect (`OpenAIModels`), which the sampler treats as no cut.
            #expect(config.temperature == 1.0 && config.topK == nil && config.topP == 1.0)
            #expect(config.minP == 0 && config.repetitionPenalty == 1)
            #expect(config.seed == 7)
            // max_new_tokens = min(512, context - prompt) in the loop.
            #expect(h.model.requests.last?.maximumCompletionTokens == 512)

            let (status2, text2, _) = try await h.run(
                Self.base(hi, maxTokens: 3),
                generates: Self.mdl + Self.txt + "one two three four five" + Self.end + Self.eos)
            #expect(status2 == 200, "\(text2)")
            let capped = Stream(text2)
            #expect(capped.finishReasons == ["length"] && capped.errors == 0 && capped.shapeOK)
            #expect(capped.usage?["completion_tokens"] as? Int == 3)

            let (status3, text3, _) = try await h.run(
                Self.base(hi),
                generates: Self.mdl + Self.txt + "Hello." + Self.end + Self.eos
                    + Self.mdl + Self.txt + "must not appear" + Self.end)
            #expect(status3 == 200, "\(text3)")
            let stopped = Stream(text3)
            #expect(stopped.finishReasons == ["stop"] && stopped.content == "Hello.")
            #expect(!text3.contains("must not appear"))
        }
    }

    /// F04: for Inkling 2, a tool call the parser rejects (invalid JSON, a
    /// tool the request did not offer, header and body naming different
    /// tools) or one still open at EOS reaches the client as the block's body
    /// text with `finish_reason: stop` and HTTP 200, streamed or not: no
    /// error frame, no 500, no tool call. Ruling B, extended: when the
    /// request offered no tools (`tools` omitted or empty) or set
    /// `tool_choice: "none"` (the server then offers none), any call, even a
    /// well-formed one, names a tool not offered and is text too.
    @Test func f04MalformedToolCallsAreTextAndStop() async throws {
        try await Self.withHarness { h in
            let question: [[String: Any]] = [["role": "user", "content": "Weather in Paris?"]]
            let offered = Self.base(question, tools: [Self.tool("get_weather", "Current weather for a city.")])
            var toolChoiceNone = offered
            toolChoiceNone["tool_choice"] = "none"
            let wellFormed = #"{"name":"get_weather","args":{"city":"Paris"}}"#
            let call = Self.mdl + "get_weather" + Self.invoke + wellFormed + Self.end + Self.eos
            // `engine`: the tools the server hands the engine for that request.
            let weather = ["get_weather"], none: [String] = []
            let variants: [(name: String, request: [String: Any], engine: [String], generates: String,
                            body: String)] = [
                ("invalid-json-body", offered, weather,
                 Self.mdl + "get_weather" + Self.invoke + #"{"name":"get_weather","args":{"city":}"# + Self.end
                    + Self.eos,
                 #"{"name":"get_weather","args":{"city":}"#),
                ("unknown-tool", offered, weather,
                 Self.mdl + "send_email" + Self.invoke + #"{"name":"send_email","args":{"to":"a@b.c"}}"# + Self.end
                    + Self.eos,
                 #"{"name":"send_email","args":{"to":"a@b.c"}}"#),
                ("header-body-name-mismatch", offered, weather,
                 Self.mdl + "get_weather" + Self.invoke + #"{"name":"get_time","args":{"city":"Paris"}}"# + Self.end
                    + Self.eos,
                 #"{"name":"get_time","args":{"city":"Paris"}}"#),
                ("unclosed-at-eos", offered, weather,
                 Self.mdl + "get_weather" + Self.invoke + wellFormed + Self.eos,
                 wellFormed),
                ("no-tools-offered", Self.base(question), none, call, wellFormed),
                ("empty-tools", Self.base(question, tools: []), none, call, wellFormed),
                ("tool-choice-none", toolChoiceNone, none, call, wellFormed),
            ]
            for variant in variants {
                var body = variant.request
                let (status, text, _) = try await h.run(body, generates: variant.generates)
                #expect(status == 200, "\(variant.name): \(text)")
                #expect(h.model.requests.last?.tools.map(\.name) == variant.engine,
                        "\(variant.name): tools the engine was handed")
                let s = Stream(text)
                #expect(s.errors == 0 && s.done && s.shapeOK, "\(variant.name): \(text)")
                #expect(s.roles == ["assistant"], "\(variant.name)")
                #expect(s.content == variant.body, "\(variant.name): \(s.content)")
                #expect(s.toolCalls.isEmpty && s.finishReasons == ["stop"], "\(variant.name)")
                #expect(!s.content.contains("<|") && s.reasoning.isEmpty, "\(variant.name)")
                #expect(s.usage?["completion_tokens"] as? Int
                        == h.tokenizer.encode(variant.generates, addBOS: false).count, "\(variant.name)")

                body["stream"] = false
                body.removeValue(forKey: "stream_options")
                let (plainStatus, plain, _) = try await h.run(body, generates: variant.generates)
                #expect(plainStatus == 200, "\(variant.name): \(plain)")
                #expect(h.model.requests.last?.tools.map(\.name) == variant.engine,
                        "\(variant.name): tools the engine was handed (not streamed)")
                let object = try #require(try JSONSerialization.jsonObject(with: Data(plain.utf8)) as? [String: Any])
                let choice = try #require((object["choices"] as? [[String: Any]])?.first)
                let message = try #require(choice["message"] as? [String: Any])
                #expect(message["content"] as? String == variant.body, "\(variant.name)")
                #expect((message["tool_calls"] as? [Any])?.isEmpty ?? true, "\(variant.name)")
                #expect(choice["finish_reason"] as? String == "stop", "\(variant.name)")
            }
        }
    }

    /// The same malformed call from another dialect still fails as before:
    /// an in-band error frame after the head when streaming, HTTP 500
    /// otherwise (`HTTPServerTests.streamingFailureAfterHeadReportsErrorInBand`
    /// pins the error frame's shape and is unchanged).
    @Test func otherDialectsStillFailOnAMalformedToolCall() async throws {
        try await Self.withHarness(fixture: "ChatMLTokenizer", family: .qwen36, modelID: "qwen3.6-35b-a3b",
                                   product: .qwen36, eosText: "<|im_end|>") { h in
            #expect(h.tokenizer.dialect == .chatml)
            let generates = "<tool_call>" + #"{"name":"get_weather","arguments":{"city":}"# + "</tool_call>"
                + "<|im_end|>"
            var body: [String: Any] = ["model": "qwen3.6-35b-a3b", "stream": true, "max_completion_tokens": 64,
                                       "messages": [["role": "user", "content": "Weather in Paris?"]],
                                       "tools": [Self.tool("get_weather", "Current weather for a city.")]]
            let (status, text, _) = try await h.run(body, generates: generates)
            #expect(status == 200)
            let s = Stream(text)
            #expect(s.errors == 1 && s.content.isEmpty && s.toolCalls.isEmpty, "\(text)")
            body["stream"] = false
            let (plainStatus, plain, _) = try await h.run(body, generates: generates)
            #expect(plainStatus == 500, "\(plain)")

            // With no tools offered, a ChatML model's call still fails the
            // request as it does today (in-band error after the head).
            body["stream"] = true
            body.removeValue(forKey: "tools")
            let valid = "<tool_call>" + #"{"name":"get_weather","arguments":{"city":"Paris"}}"# + "</tool_call>"
                + "<|im_end|>"
            let (noToolsStatus, noToolsText, _) = try await h.run(body, generates: valid)
            #expect(noToolsStatus == 200)
            #expect(Stream(noToolsText).errors == 1, "\(noToolsText)")
        }
    }

    /// A tool-call block cut off by `max_completion_tokens` ends with
    /// `length` and the body generated so far as text, once (no tail text
    /// added twice).
    @Test func aToolCallCutByTheCapIsItsBodySoFar() async throws {
        try await Self.withHarness { h in
            let generates = Self.mdl + "get_weather" + Self.invoke + #"{"name":"get_weather","args":{"city":"Paris"}}"#
                + Self.end + Self.eos
            let script = h.tokenizer.encode(generates, addBOS: false)
            let invoke = try #require(h.tokenizer.encode(Self.invoke, addBOS: false).first)
            let start = try #require(script.firstIndex(of: invoke)) + 1
            let cap = start + 10
            let expected = h.tokenizer.decode(Array(script[start..<cap]), skipSpecialTokens: true)
            #expect(!expected.isEmpty)
            let body = Self.base([["role": "user", "content": "Weather in Paris?"]], maxTokens: cap,
                                 tools: [Self.tool("get_weather", "Current weather for a city.")])
            let (status, text, _) = try await h.run(body, generates: generates)
            #expect(status == 200, "\(text)")
            let s = Stream(text)
            #expect(s.errors == 0 && s.finishReasons == ["length"] && s.toolCalls.isEmpty)
            #expect(s.content == expected, "\(s.content)")
        }
    }

    /// With single-prefix reuse on, a turn whose malformed call became text
    /// is not kept: the next turn prefills its whole prompt (the KV held the
    /// call's framing, a re-render of the reply does not). A normal text turn
    /// is reused, which shows the test can see reuse.
    @Test func aConvertedTurnIsNotReusedAsAPrefix() async throws {
        try await Self.withHarness(promptCacheMode: .singlePrefix) { h in
            let tools = [Self.tool("get_weather", "Current weather for a city.")]
            /// Sends one turn; returns its content and how many prompt tokens
            /// the runner was fed against the rendered prompt.
            func turn(_ messages: [[String: Any]], _ generates: String) async throws
                -> (content: String, prefilled: Int, prompt: Int) {
                h.runner.plan(script: h.tokenizer.encode(generates, addBOS: false))
                let (status, text) = try await h.post(Self.base(messages, tools: tools))
                #expect(status == 200, "\(text)")
                let s = Stream(text)
                let completion = try #require(s.usage?["completion_tokens"] as? Int)
                // Every generated token but the last (EOS) is fed back.
                return (s.content, h.runner.inputs.count - (completion - 1), h.model.lastPrompt.count)
            }
            let question: [String: Any] = ["role": "user", "content": "Weather in Paris?"]
            let thanks: [String: Any] = ["role": "user", "content": "Thanks."]

            let plain = try await turn([question], Self.mdl + Self.txt + "Let me check." + Self.end + Self.eos)
            #expect(plain.content == "Let me check." && plain.prefilled == plain.prompt)
            let reused = try await turn([question, ["role": "assistant", "content": plain.content], thanks],
                                        Self.mdl + Self.txt + "You are welcome." + Self.end + Self.eos)
            #expect(reused.prefilled < reused.prompt, "a normal turn should be reused")

            let malformed = Self.mdl + "get_weather" + Self.invoke + #"{"name":"get_weather","args":{"city":}"#
                + Self.end + Self.eos
            let converted = try await turn([question], malformed)
            #expect(converted.content == #"{"name":"get_weather","args":{"city":}"#)
            let after = try await turn([question, ["role": "assistant", "content": converted.content], thanks],
                                       Self.mdl + Self.txt + "You are welcome." + Self.end + Self.eos)
            #expect(after.prefilled == after.prompt, "a converted turn must not be reused")
        }
    }

    /// Ruling A, detail: a `role: tool` message without `tool_call_id` is
    /// rejected before anything renders (400), as it is today; the client
    /// always sends the id.
    @Test func aToolResultWithoutItsCallIdIsRejected() async throws {
        try await Self.withHarness { h in
            let messages: [[String: Any]] = [
                ["role": "user", "content": "Weather in Paris?"],
                ["role": "assistant", "content": "",
                 "tool_calls": [["id": "call_1", "type": "function",
                                 "function": ["name": "get_weather", "arguments": #"{"city":"Paris"}"#]]]],
                ["role": "tool", "name": "get_weather", "content": "18 C"],
            ]
            h.runner.plan(script: [])
            let (status, text) = try await h.post(
                Self.base(messages, tools: [Self.tool("get_weather", "Current weather for a city.")]))
            #expect(status == 400, "\(text)")
            #expect(text.contains("invalid_tool_result"))
            #expect(h.model.requests.isEmpty)
        }
    }
}

import Foundation
import Testing
import Mference
import MferenceEngine

/// Opt-in real-install check of `NativeBackend` (spec gate G5): runs a fixed
/// prompt set through `EngineHost` + `NativeBackend` at temperature 0 and
/// writes one canonical text per prompt, for comparison with
/// `MferenceServer`'s output for the same requests.
///
/// Gates (all required, else the suite is skipped):
/// - `MFERENCE_ENGINE_NATIVE_GTURBO`: the `.gturbo` install.
/// - `MFERENCE_ENGINE_PARITY_PROMPTS`: the prompt-set JSON, a list of
///   `{"name", "max_tokens", "messages": [{"role", "content"}], "tools"?}`
///   with OpenAI-shaped function tools.
/// - `MFERENCE_ENGINE_PARITY_OUT`: directory that receives `<name>.txt`.
///
/// The request is built the way the server's validator builds one that sets
/// only `temperature: 0` and `max_tokens`: every other sampling field comes
/// from the checkpoint's generation defaults. Messages use the system, user,
/// assistant, and tool roles only: the validator's developer-role rewrite is
/// not mirrored, and a developer message is refused. Run only after the AGENTS.md
/// preflight, under the machine lock, with no other model process.
@Suite(.serialized) struct NativeBackendInstalledTests {
    private static let environment = ProcessInfo.processInfo.environment
    private static let install = environment["MFERENCE_ENGINE_NATIVE_GTURBO"]
    private static let prompts = environment["MFERENCE_ENGINE_PARITY_PROMPTS"]
    private static let output = environment["MFERENCE_ENGINE_PARITY_OUT"]
    private static var enabled: Bool { install != nil && prompts != nil && output != nil }

    struct PromptCase: Decodable {
        struct Message: Decodable {
            let role: String
            let content: String
        }
        struct Tool: Decodable {
            struct Function: Decodable {
                let name: String
                let description: String?
                let parameters: JSONValue
            }
            let function: Function
        }
        let name: String
        let max_tokens: Int
        let messages: [Message]
        let tools: [Tool]?
    }

    /// The server's validator for a request that sets only `temperature: 0`
    /// and `max_tokens`.
    static func request(_ prompt: PromptCase, defaults: GenerationConfig) throws -> EngineRequest {
        let messages = try prompt.messages.map { message -> MFTokenizer.Message in
            // The server's validator rewrites some roles (developer becomes
            // system outside Gemma); this harness does not, so the prompt set
            // must not use them.
            guard message.role != "developer",
                  let role = MFTokenizer.Role(rawValue: message.role) else {
                throw EngineError.invalidRequest(message: "role \(message.role)", param: nil, code: "test")
            }
            return MFTokenizer.Message(role: role, content: message.content)
        }
        let tools = (prompt.tools ?? []).map {
            MFTokenizer.FunctionDefinition(name: $0.function.name,
                                           description: $0.function.description ?? "",
                                           parameters: $0.function.parameters)
        }
        let topK = defaults.topK ?? 0
        let config = GenerationConfig(maxNewTokens: prompt.max_tokens,
                                      temperature: 0,
                                      topK: topK == 0 ? nil : topK,
                                      topP: defaults.topP ?? 1,
                                      repetitionPenalty: defaults.repetitionPenalty,
                                      presencePenalty: defaults.presencePenalty,
                                      frequencyPenalty: defaults.frequencyPenalty,
                                      repeatLastN: defaults.repeatLastN,
                                      minP: defaults.minP,
                                      seed: nil,
                                      stopStrings: [])
        return EngineRequest(messages: messages, tools: tools, generationConfig: config,
                             maximumCompletionTokens: prompt.max_tokens)
    }

    /// Content, reasoning, and tool calls (name and arguments; call ids are
    /// random) in one text, the same canonical form the server-side
    /// comparison script writes.
    static func canonical(_ completion: EngineCompletion) -> String {
        var text = "content:\n" + completion.content + "\nreasoning:\n" + (completion.reasoningContent ?? "")
        text += "\ntool_calls:\n"
        for call in completion.toolCalls {
            text += call.name + " " + call.argumentsJSON + "\n"
        }
        text += "finish_reason: " + completion.finishReason.rawValue + "\n"
        return text
    }

    @Test(.enabled(if: enabled)) func fixedPromptSetAtTemperatureZero() async throws {
        let directory = URL(fileURLWithPath: try #require(Self.install), isDirectory: true)
        let promptCases = try JSONDecoder().decode(
            [PromptCase].self, from: Data(contentsOf: URL(fileURLWithPath: try #require(Self.prompts))))
        let out = URL(fileURLWithPath: try #require(Self.output), isDirectory: true)
        try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)

        let catalog = EngineCatalogDiscovery.discover(
            roots: [], ggufDirectories: [], scope: .all, explicitModel: directory)
        let descriptor = try #require(catalog.models.first)
        let events = Collector<String>()
        let host = EngineHost(catalog: catalog,
                              backends: [NativeBackend()],
                              loadOptions: EngineLoadOptions(),
                              observer: { events.append("\($0)") })
        for prompt in promptCases {
            let streamed = Collector<EngineEvent>()
            let completion = try await host.withModel(descriptor.id) { model in
                let request = try Self.request(prompt, defaults: model.traits.generationDefaults)
                let prepared = try await model.prepare(request)
                return try await model.generate(prepared) { streamed.append($0) }
            }
            // Streaming and the completion agree.
            var streamedContent = ""
            for case .content(let text) in streamed.all { streamedContent += text }
            #expect(streamedContent == completion.content)
            try Self.canonical(completion).write(to: out.appendingPathComponent("\(prompt.name).txt"),
                                                 atomically: true, encoding: .utf8)
            FileHandle.standardError.write(Data(
                "[engine parity] \(descriptor.id) \(prompt.name) prompt=\(completion.usage.promptTokens) new=\(completion.usage.completionTokens) finish=\(completion.finishReason.rawValue)\n".utf8))
        }
        await host.shutdown()
        for line in events.all { FileHandle.standardError.write(Data("[engine parity] \(line)\n".utf8)) }
    }
}

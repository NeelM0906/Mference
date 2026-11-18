import Foundation
import Jinja
import Mference

/// A GGUF model's own chat template (`tokenizer.chat_template`), rendered with
/// the repository's patched swift-jinja.
///
/// The context is the one swift-transformers builds for a native model
/// (`lstripBlocks` and `trimBlocks`, `messages`, `tools`,
/// `add_generation_prompt`, `bos_token` / `eos_token`, plus keyword
/// arguments), and messages are converted exactly as `MFTokenizer` converts
/// them, so the native and llama.cpp paths agree. The rendered bytes are
/// checked against llama.cpp's own renderer at the pin by committed fixtures.
struct LlamaChatTemplate: Sendable {
    let source: String
    private let template: Template
    private let bosToken: String?
    private let eosToken: String?

    init(source: String, bosToken: String?, eosToken: String?) throws {
        self.source = source
        do {
            template = try Template(source, with: .init(lstripBlocks: true, trimBlocks: true))
        } catch {
            throw LlamaBackendError.invalidChatTemplate("\(error)")
        }
        self.bosToken = bosToken
        self.eosToken = eosToken
    }

    /// Removes a leading BOS / trailing EOS the template printed when
    /// tokenization adds that token itself, as llama.cpp's
    /// `common_chat_templates_apply` does, so the prompt never carries it
    /// twice.
    func strippingImplicitSpecialTokens(_ rendered: String, addsBOS: Bool, addsEOS: Bool) -> String {
        var text = Substring(rendered)
        if addsBOS, let bosToken, !bosToken.isEmpty, text.hasPrefix(bosToken) {
            text = text.dropFirst(bosToken.count)
        }
        if addsEOS, let eosToken, !eosToken.isEmpty, text.hasSuffix(eosToken) {
            text = text.dropLast(eosToken.count)
        }
        return String(text)
    }

    /// Whether the template reads `enable_thinking`, the Qwen thinking switch.
    var readsEnableThinking: Bool { source.contains("enable_thinking") }

    /// Whether the template reads `reasoning_effort` (Qwen 3.8's source
    /// template: `xhigh`, `medium`, `low`).
    var readsReasoningEffort: Bool { source.contains("reasoning_effort") }

    /// The template kwargs for a request, as llama-server passes them: an
    /// omitted effort sets nothing, so the template's own default holds;
    /// `none` turns thinking off; any other effort turns it on and, for a
    /// template that reads it, names the level.
    static func keywordArguments(reasoningEffort: QwenReasoningEffort?,
                                 preserveThinking: Bool,
                                 readsEnableThinking: Bool,
                                 readsReasoningEffort: Bool = false) -> [String: any Sendable] {
        var kwargs: [String: any Sendable] = [:]
        if readsEnableThinking, let reasoningEffort {
            kwargs["enable_thinking"] = reasoningEffort != .off
        }
        if readsReasoningEffort, let reasoningEffort, reasoningEffort != .off {
            kwargs["reasoning_effort"] = reasoningEffort.rawValue
        }
        if preserveThinking { kwargs["preserve_thinking"] = true }
        return kwargs
    }

    func render(messages: [MFTokenizer.Message],
                tools: [MFTokenizer.FunctionDefinition],
                addGenerationPrompt: Bool = true,
                keywordArguments: [String: any Sendable] = [:]) throws -> String {
        var context: [String: Value] = [
            "messages": try .array(messages.map { try Value(any: Self.templateMessage($0)) }),
            "add_generation_prompt": .boolean(addGenerationPrompt),
        ]
        if !tools.isEmpty {
            context["tools"] = try .array(tools.map(Self.templateTool))
        }
        if let bosToken { context["bos_token"] = .string(bosToken) }
        if let eosToken { context["eos_token"] = .string(eosToken) }
        for (key, value) in keywordArguments {
            context[key] = try Value(any: value)
        }
        do {
            return try template.render(context)
        } catch {
            throw EngineError.invalidRequest(message: "chat template failed: \(error)",
                                             param: "messages", code: "invalid_chat_template")
        }
    }

    /// The dictionary `MFTokenizer.encodeToolChat` passes for a message.
    static func templateMessage(_ message: MFTokenizer.Message) throws -> [String: any Sendable] {
        var value: [String: any Sendable] = [
            "role": message.role.rawValue,
            "content": message.content as (any Sendable)?,
        ]
        if !message.toolCalls.isEmpty {
            value["tool_calls"] = try message.toolCalls.map { call -> [String: any Sendable] in
                [
                    "id": call.id,
                    "type": "function",
                    "function": [
                        "name": call.name,
                        "arguments": try call.arguments.jinjaSendableValue(),
                    ] as [String: any Sendable],
                ]
            }
        }
        if let toolCallID = message.toolCallID { value["tool_call_id"] = toolCallID }
        if let name = message.name { value["name"] = name }
        if let reasoning = message.reasoningContent { value["reasoning_content"] = reasoning }
        return value
    }

    /// A tool in the OpenAI shape, keyed in the order llama.cpp builds it
    /// (`type`, then `function` with `name`, `description`, `parameters`), so
    /// a template's `tool | tojson` prints the same bytes as llama.cpp's
    /// renderer. The schema itself is a `JSONValue`, whose object keys carry
    /// no order; they render sorted, as on the native path.
    static func templateTool(_ tool: MFTokenizer.FunctionDefinition) throws -> Value {
        let function: OrderedDictionary<String, Value> = [
            "name": .string(tool.name),
            "description": .string(tool.description),
            "parameters": try Value(any: tool.parameters.jinjaSendableValue()),
        ]
        let wrapper: OrderedDictionary<String, Value> = [
            "type": .string("function"),
            "function": .object(function),
        ]
        return .object(wrapper)
    }
}

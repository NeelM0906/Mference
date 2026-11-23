import Foundation
import Jinja
import Tokenizers

/// The special tokens of the Inkling 2 chat format (`thinkingmachines/Inkling-2-Mini`
/// `b85c3cf`, ids 199,998 to 200,057). A rendered conversation is a list of
/// messages, each `<|message_ROLE|>[name]<|content_KIND|>…<|end_message|>`;
/// an assistant turn ends with `<|content_model_end_sampling|>`, the EOS.
/// The template has no generation prompt: the model writes its own
/// `<|message_model|>` header after the prompt's last `<|end_message|>`.
public struct Inkling2SpecialTokens: Sendable, Equatable {
    public let messageUser: Int32
    public let messageModel: Int32
    public let messageSystem: Int32
    public let messageTool: Int32
    public let contentText: Int32
    public let contentThinking: Int32
    public let contentToolError: Int32
    public let contentXML: Int32
    public let contentInvokeToolJSON: Int32
    public let contentInvokeToolText: Int32
    public let endMessage: Int32
    public let endSampling: Int32
}

extension MFTokenizer {
    static let inkling2MessageUser = "<|message_user|>"
    static let inkling2MessageModel = "<|message_model|>"
    static let inkling2MessageSystem = "<|message_system|>"
    static let inkling2MessageTool = "<|message_tool|>"
    static let inkling2ContentText = "<|content_text|>"
    static let inkling2ContentThinking = "<|content_thinking|>"
    static let inkling2ContentToolError = "<|content_tool_error|>"
    static let inkling2ContentXML = "<|content_xml|>"
    static let inkling2InvokeToolJSON = "<|content_invoke_tool_json|>"
    static let inkling2InvokeToolText = "<|content_invoke_tool_text|>"
    static let inkling2EndMessage = "<|end_message|>"
    static let inkling2EndSampling = "<|content_model_end_sampling|>"

    /// Inkling 2's thinking effort, the template's `reasoning_effort`: one of
    /// its named levels or a number in `0 ... 0.99`. Omitted, the template
    /// uses 0.9 (`high`). Even the minimum does not fully disable thinking.
    public enum Inkling2Effort: Sendable, Equatable {
        case named(String)
        case level(Double)

        /// The template's default when no effort is given.
        public static let defaultLevel = 0.9
        /// The template's `effort_map`, name to level.
        public static let namedLevels: [String: Double] = [
            "none": 0.0, "minimal": 0.1, "low": 0.2, "medium": 0.7,
            "high": 0.9, "xhigh": 0.99, "max": 0.99,
        ]

        /// The level the template will print.
        public var level: Double? {
            switch self {
            case .named(let name):
                return Self.namedLevels[name.trimmingCharacters(in: .whitespaces)]
            case .level(let value):
                return value
            }
        }

        /// Rejects what the template would reject, with the same rule: an
        /// unknown name, or a level outside `0 ... 0.99`.
        public func validate() throws {
            guard let level, level.isFinite, level >= 0, level <= 0.99 else {
                throw MFTokenizerError.invalidChatTemplate(
                    "Inkling 2 reasoning effort must be one of \(Self.namedLevels.keys.sorted()) or a number in [0.0, 0.99]")
            }
        }

        var templateValue: any Sendable {
            switch self {
            case .named(let name): return name
            case .level(let value): return value
            }
        }
    }

    /// The Inkling 2 tokens; nil for every other dialect.
    public var inkling2Tokens: Inkling2SpecialTokens? { inkling2SpecialTokens }

    /// The template's `effort_map` names for the shared effort setting.
    public var inkling2EffortNames: [String] { Inkling2Effort.namedLevels.keys.sorted() }

    /// Renders `messages` and `tools` through the installed
    /// `chat_template.jinja` (the vendor's, byte for byte) with the vendored
    /// Jinja, in the context transformers builds: `messages`, `tools` when
    /// given, `add_generation_prompt` (unused by this template) and
    /// `reasoning_effort` when given. The render has no generation prompt;
    /// the model writes its own `<|message_model|>` header next.
    public func renderInkling2Chat(messages: [Message],
                                   tools: [FunctionDefinition] = [],
                                   effort: Inkling2Effort? = nil) throws -> String {
        guard dialect == .inkling2 else {
            throw MFTokenizerError.unsupportedForDialect("Inkling 2 chat rendering")
        }
        guard let source = inkling2ChatTemplateSource else {
            throw MFTokenizerError.missingToolTemplate
        }
        try effort?.validate()
        let inputs = try inkling2TemplateInputs(messages: messages, tools: tools)
        do {
            // transformers compiles chat templates with trim_blocks and lstrip_blocks.
            let template = try Jinja.Template(source, with: .init(lstripBlocks: true, trimBlocks: true))
            var context: [String: Jinja.Value] = [
                "messages": .array(try inputs.messages.map { try Jinja.Value(any: $0) }),
                "add_generation_prompt": .boolean(false),
            ]
            if !inputs.tools.isEmpty {
                context["tools"] = .array(try inputs.tools.map { try Jinja.Value(any: $0) })
            }
            if let effort {
                context["reasoning_effort"] = try Jinja.Value(any: effort.templateValue)
            }
            return try template.render(context)
        } catch let error as MFTokenizerError {
            throw error
        } catch {
            // The template raises for unknown roles and malformed tool calls.
            throw MFTokenizerError.invalidChatTemplate("Inkling 2 template: \(error)")
        }
    }

    /// `renderInkling2Chat`, tokenized with `ignore_merges` like the reference.
    public func encodeInkling2Chat(messages: [Message],
                                   tools: [FunctionDefinition] = [],
                                   effort: Inkling2Effort? = nil) throws -> [Int32] {
        encode(try renderInkling2Chat(messages: messages, tools: tools, effort: effort), addBOS: false)
    }

    /// The messages and tools in the shape the vendor template reads (the
    /// OpenAI / transformers shape): `tool_calls[].function.arguments` as a
    /// parsed object, `reasoning_content` for an assistant's thinking,
    /// `name` or `tool_call_id` naming a tool result. Swift dictionaries reach
    /// the template with sorted keys, so nested argument objects render in
    /// sorted key order (the template itself sorts the top level).
    func inkling2TemplateInputs(messages: [Message],
                                tools: [FunctionDefinition]) throws
        -> (messages: [[String: any Sendable]], tools: [[String: any Sendable]]) {
        let renderedMessages: [[String: any Sendable]] = try messages.map { message in
            if message.role == .developer {
                throw MFTokenizerError.invalidChatTemplate(
                    "Inkling 2's template has no developer role; send it as a leading system message")
            }
            var value: [String: any Sendable] = [
                "role": message.role.rawValue,
                "content": message.content,
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
        let renderedTools: [[String: any Sendable]] = try tools.map { tool in
            [
                "type": "function",
                "function": [
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": try tool.parameters.jinjaSendableValue(),
                ] as [String: any Sendable],
            ]
        }
        return (renderedMessages, renderedTools)
    }

    /// Tokens that follow a cached assistant turn when the user replies: the
    /// sampled EOS (`<|content_model_end_sampling|>`, which ended generation
    /// and never entered the cache) and the user's message. The effort line
    /// is rendered once, before the first non-system message, so a later
    /// turn does not repeat it.
    func inkling2TextContinuation(userContent: String) -> [Int32] {
        [endOfTurnID] + encode(Self.inkling2MessageUser + Self.inkling2ContentText
                               + userContent + Self.inkling2EndMessage,
                               addBOS: false)
    }
}

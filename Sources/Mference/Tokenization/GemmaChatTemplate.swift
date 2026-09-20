import CryptoKit
import Foundation
import Jinja
import Tokenizers

extension MFTokenizer {
    /// The source macro emits a null argument as Python's `None`. Configure
    /// that output at the interpreter boundary without rewriting the template,
    /// converting nulls to strings, or changing other checkpoints' rendering.
    func encodeGemmaQATChat(messages: [Tokenizers.Message], tools: [ToolSpec],
                            enableThinking: Bool, addGenerationPrompt: Bool) throws -> [Int32] {
        let template = try Template(String(decoding: effectiveGemmaChatTemplateData(), as: UTF8.self),
                                    with: .init(lstripBlocks: true, trimBlocks: true))
        let environment = Environment()
        environment.policies.nullOutput = "None"
        let context: [String: Value] = try [
            "messages": .array(messages.map { try Value(any: $0) }),
            "tools": .array(tools.map { try Value(any: $0) }),
            "bos_token": .string(decode([bosID], skipSpecialTokens: false)),
            "enable_thinking": .boolean(enableThinking),
            "add_generation_prompt": .boolean(addGenerationPrompt),
        ]
        return encode(try template.render(context, environment: environment), addBOS: false)
    }

    /// Bound to this checkpoint copy, never shared through the tokenizer cache.
    public func effectiveGemmaChatTemplateData() throws -> Data {
        if isGemmaQAT {
            guard let installedGemmaTemplate else { throw MFTokenizerError.missingToolTemplate }
            return installedGemmaTemplate
        }
        return try Self.gemmaChatTemplateData()
    }

    /// The effective Gemma contract belongs to the app, leaving existing
    /// installs and their integrity receipts untouched. No inference-time fetch.
    public static func gemmaChatTemplateData() throws -> Data {
        try bundledGemmaTemplate.get()
    }

    private static let bundledGemmaTemplate: Result<Data, Error> = Result {
        guard let url = Bundle.module.url(forResource: "chat_template", withExtension: "jinja",
                                         subdirectory: "Gemma4") else {
            throw MFTokenizerError.invalidChatTemplate(
                "Gemma chat resource is missing; rebuild or reinstall the Mference application (model weights do not need reinstalling)")
        }
        let data = try Data(contentsOf: url)
        let digest = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        guard digest == "ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4" else {
            throw MFTokenizerError.invalidChatTemplate(
                "Gemma chat resource failed its integrity check; rebuild or reinstall the Mference application")
        }
        return data
    }
}

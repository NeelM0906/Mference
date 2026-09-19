import CryptoKit
import Foundation

extension MFTokenizer {
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

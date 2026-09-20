import Foundation
import Testing
@testable import Mference

/// Tokenizer-only installed gate: no weight loading, inference, or downloads.
@Suite struct QwenMatchedPromptTests {
    @Test func installedReleaseCorpusHasIdenticalPromptIDs() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let basePath = env["MFERENCE_MATCHED_QWEN_BASE"],
              let swiftPath = env["MFERENCE_MATCHED_QWEN_SWIFT"] else { return }
        let baseURL = URL(fileURLWithPath: basePath)
        let swiftURL = URL(fileURLWithPath: swiftPath)
        #expect(try ManifestReader.peekModelID(directoryURL: baseURL) == CheckpointIdentity.baseQwen38)
        #expect(try ManifestReader.peekModelID(directoryURL: swiftURL) == CheckpointIdentity.swiftQwen38)
        let base = try await MFTokenizer.load(forModelDirectory: baseURL)
        let swift = try await MFTokenizer.load(forModelDirectory: swiftURL)
        struct Tool: Decodable { let name: String; let description: String; let parameters: JSONValue }
        struct Case: Decodable { let id: String; let prompt: String; let tool: Tool? }
        var root = URL(fileURLWithPath: #filePath)
        for _ in 0..<5 { root.deleteLastPathComponent() }
        let cases = try JSONDecoder().decode([Case].self, from: Data(contentsOf:
            root.appendingPathComponent("docs/benchmark-prompts/release-screen-v1/cases.json")))
        #expect(cases.count == 60)
        for item in cases {
            let tools = item.tool.map { [MFTokenizer.FunctionDefinition(
                name: $0.name, description: $0.description, parameters: $0.parameters)] } ?? []
            for effort in QwenReasoningEffort.allCases {
                let messages: [MFTokenizer.Message] = [.init(role: .user, content: item.prompt)]
                let baseIDs = try base.encodeChat(messages: messages, tools: tools, reasoningEffort: effort)
                let swiftIDs = try swift.encodeChat(messages: messages, tools: tools, reasoningEffort: effort)
                #expect(baseIDs == swiftIDs, "\(item.id)/\(effort): mismatched model inputs")
                #expect(baseIDs.count < 4096)
            }
        }
        print("[qwen-matched-prompts] 60 cases x 4 efforts: identical installed prompt IDs")
    }
}

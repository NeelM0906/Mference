import Foundation
import Testing
@testable import Mference
@testable import MferenceServerCore

@Suite struct SwiftQwenServerTests {
    @Test func effortAndHistoricalReasoningSurviveValidation() throws {
        for effort in ["xhigh", "medium", "low", "none"] {
            let data = Data("""
            {"model":"swift-qwen3.8-27b-int4g64","reasoning_effort":"\(effort)","messages":[
              {"role":"user","content":"A"},
              {"role":"assistant","content":"B","reasoning_content":"Check A"},
              {"role":"user","content":"C"}]}
            """.utf8)
            let request = try JSONDecoder().decode(OpenAIChatRequest.self, from: data)
            let validated = try OpenAIRequestValidator.validate(request, modelID: request.model, dialect: .chatml)
            #expect(validated.reasoningEffort?.rawValue == effort)
            #expect(validated.messages[1].reasoningContent == "Check A")
        }
    }

    @Test func unsupportedEffortAndDeveloperAreRejected() throws {
        for fragment in ["\"reasoning_effort\":\"high\",", ""] {
            let role = fragment.isEmpty ? "developer" : "system"
            let data = Data("""
            {"model":"swift-qwen3.8-27b-int4g64",\(fragment)"messages":[
              {"role":"\(role)","content":"Guide"},{"role":"user","content":"Hi"}]}
            """.utf8)
            let request = try JSONDecoder().decode(OpenAIChatRequest.self, from: data)
            #expect(throws: ServerRequestError.self) {
                try OpenAIRequestValidator.validate(request, modelID: request.model, dialect: .chatml)
            }
        }
    }

    @Test func checkpointIdentityIsIndependentOfDirectoryName() throws {
        let root = try ServerLibraryFixture.makeRoot("swift-qwen-identities")
        defer { try? FileManager.default.removeItem(at: root) }
        for (directory, id) in [("renamed", CheckpointIdentity.swiftQwen38), ("qwen38", "qwen3.8-27b-4bit")] {
            let folder = root.appendingPathComponent(directory)
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            try JSONSerialization.data(withJSONObject: ["modelID": id, "arch": ["family": "qwen38"]])
                .write(to: folder.appendingPathComponent("manifest.json"))
            #expect(try ManifestReader.peekModelID(directoryURL: folder) == id)
        }
        let index = ServerLibraryDiscovery.discover(roots: [root], probe: { directory in
            directory == root ? .notAModelDirectory : .complete(family: .qwen38)
        })
        #expect(Set(index.entries.map(\.modelID)) == [CheckpointIdentity.swiftQwen38, "qwen3.8-27b-4bit"])
        #expect(ServerFamilyModelID.modelID(for: .qwen38, checkpointID: CheckpointIdentity.swiftQwen38) == CheckpointIdentity.swiftQwen38)
    }
}

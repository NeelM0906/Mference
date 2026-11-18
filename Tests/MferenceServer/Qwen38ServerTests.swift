import Foundation
import Testing
@testable import Mference
@testable import MferenceServerCore

/// Base Qwen 3.8 27B (`qwen3.8-27b-4bit`) request rules: the model ID alone
/// enables `reasoning_effort`, and an explicit effort routes the request
/// through the source template, which rejects developer turns. These are the
/// base-checkpoint runs of the removed Swift-Qwen server suite.
@Suite struct Qwen38ServerTests {
    @Test func effortAndHistoricalReasoningSurviveValidation() throws {
        let model = CheckpointIdentity.baseQwen38
        for effort in ["xhigh", "medium", "low", "none"] {
            let data = Data("""
            {"model":"\(model)","reasoning_effort":"\(effort)","messages":[
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

    @Test func aliasCapabilitiesAndBaseExplicitRoleContract() throws {
        let data = Data(#"{"model":"alias","reasoning_effort":"low","messages":[{"role":"user","content":"Hi"}]}"#.utf8)
        let request = try JSONDecoder().decode(OpenAIChatRequest.self, from: data)
        #expect(try OpenAIRequestValidator.validate(request, modelID: "alias", dialect: .chatml,
            swiftQwen: false, qwenReasoning: true).reasoningEffort == .low)
        #expect(throws: ServerRequestError.self) {
            try OpenAIRequestValidator.validate(request, modelID: "alias", dialect: .chatml,
                swiftQwen: false, qwenReasoning: false)
        }
        for effort in ["", "\"reasoning_effort\":\"low\","] {
            let body = Data("""
            {"model":"qwen3.8-27b-4bit",\(effort)"messages":[{"role":"developer","content":"Guide"},{"role":"user","content":"Hi"}]}
            """.utf8)
            let base = try JSONDecoder().decode(OpenAIChatRequest.self, from: body)
            if effort.isEmpty {
                #expect(try OpenAIRequestValidator.validate(base, modelID: base.model, dialect: .chatml).reasoningEffort == nil)
            } else {
                #expect(throws: ServerRequestError.self) {
                    try OpenAIRequestValidator.validate(base, modelID: base.model, dialect: .chatml)
                }
            }
        }
    }

    @Test func checkpointIdentityIsIndependentOfDirectoryName() throws {
        let root = try ServerLibraryFixture.makeRoot("qwen38-identities")
        defer { try? FileManager.default.removeItem(at: root) }
        let folder = root.appendingPathComponent("renamed")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        try JSONSerialization.data(withJSONObject: ["modelID": CheckpointIdentity.baseQwen38,
                                                    "arch": ["family": "qwen38"]])
            .write(to: folder.appendingPathComponent("manifest.json"))
        #expect(try ManifestReader.peekModelID(directoryURL: folder) == CheckpointIdentity.baseQwen38)
        let index = ServerLibraryDiscovery.discover(roots: [root], probe: { directory in
            directory == root ? .notAModelDirectory : .complete(family: .qwen38)
        })
        #expect(index.entries.map(\.modelID) == [CheckpointIdentity.baseQwen38])
    }
}

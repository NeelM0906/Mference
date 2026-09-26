import Foundation
import Testing
@testable import Mference

/// Tool schemas rendered through both Gemma chat templates: the bundled
/// original template (Hub tokenizer) and the installed QAT template (fixture
/// copy, same SHA-256 as the checkpoint's).
@Suite("Gemma tool-schema rendering")
struct GemmaToolSchemaRenderTests {
    static func tokenizers() async throws -> [(name: String, tokenizer: MFTokenizer)] {
        let original = try await MFTokenizer.load()
        let qat = try await MFTokenizer.load(from: GemmaQATChatTests.fixtureFolder(), family: .gemma4)
            .forCheckpoint(CheckpointIdentity.gemma4QAT)
        return [("original", original), ("qat", qat)]
    }

    static func schema(_ json: String) throws -> JSONValue {
        try JSONDecoder().decode(JSONValue.self, from: Data(json.utf8))
    }

    static func render(_ tokenizer: MFTokenizer, parameters: JSONValue) throws -> String {
        let tool = MFTokenizer.FunctionDefinition(name: "run", description: "Run it.", parameters: parameters)
        let ids = try tokenizer.encodeToolChat(
            messages: [MFTokenizer.Message(role: .user, content: "go")], tools: [tool])
        return tokenizer.decode(ids, skipSpecialTokens: false)
    }

    /// The template renders an object property without `properties` by
    /// iterating the property's own keys as if they were parameter schemas, so
    /// a kept keyword such as `additionalProperties` or `default` reaches
    /// `value['type'] | upper` on a boolean or a mapping. DeepSeek Harness
    /// declares this shape (turbo-fieldfare #138 / #142).
    @Test("An object property without properties renders on both templates")
    func objectPropertyWithoutPropertiesRenders() async throws {
        let parameters = try Self.schema(#"""
        {"type":"object","required":["workflow"],"properties":{
          "workflow":{"type":"object","description":"Workflow spec","additionalProperties":true},
          "options":{"type":"object","default":{"retries":1}},
          "steps":{"type":"array","items":{"type":"object","properties":{
            "env":{"type":"object","additionalProperties":{"type":"string"}}}}}
        }}
        """#)
        for (name, tokenizer) in try await Self.tokenizers() {
            let rendered = try Self.render(tokenizer, parameters: parameters)
            #expect(rendered.contains(
                #"workflow:{description:<|"|>Workflow spec<|"|>,properties:{},type:<|"|>OBJECT<|"|>}"#),
                "\(name): \(rendered)")
            #expect(rendered.contains(#"options:{properties:{},type:<|"|>OBJECT<|"|>}"#), "\(name)")
            #expect(rendered.contains(#"env:{properties:{},type:<|"|>OBJECT<|"|>}"#), "\(name)")
        }
    }

    /// Schemas that already rendered must render exactly as before: an object
    /// property holding only standard keys, an array of property-less objects,
    /// and a property-less top-level object.
    @Test("Schemas that already rendered are unchanged", arguments: [
        (#"{"type":"object","properties":{"meta":{"type":"object","description":"Meta"}}}"#,
         #"parameters:{properties:{meta:{description:<|"|>Meta<|"|>,properties:{},type:<|"|>OBJECT<|"|>}},type:<|"|>OBJECT<|"|>}"#),
        (#"{"type":"object","properties":{"rows":{"type":"array","items":{"type":"object"}}}}"#,
         #"parameters:{properties:{rows:{items:{type:<|"|>OBJECT<|"|>},type:<|"|>ARRAY<|"|>}},type:<|"|>OBJECT<|"|>}"#),
        (#"{"type":"object"}"#,
         #"parameters:{type:<|"|>OBJECT<|"|>}"#),
    ])
    func renderedSchemasUnchanged(_ probe: (schema: String, expected: String)) async throws {
        for (name, tokenizer) in try await Self.tokenizers() {
            let rendered = try Self.render(tokenizer, parameters: try Self.schema(probe.schema))
            #expect(rendered.contains("declaration:run{description:<|\"|>Run it.<|\"|>," + probe.expected),
                    "\(name): \(rendered)")
        }
    }
}

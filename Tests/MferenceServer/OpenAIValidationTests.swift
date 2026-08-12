import Foundation
import Testing
@testable import Mference
@testable import MferenceServerCore

@Suite("OpenAI request validation")
struct OpenAIValidationTests {
    @Test func capturedOpenCodeInitialRequestValidates() throws {
        let request = try fixture("opencode-1.15.11-initial.json")
        let validated = try OpenAIRequestValidator.validate(
            request, modelID: "gemma-4-26b-a4b-it")
        #expect(validated.stream)
        #expect(validated.includeUsage)
        #expect(validated.tools.count == 1)
        #expect(validated.maximumCompletionTokens == 4096)
    }

    @Test func capturedOpenCodeToolResultValidates() throws {
        let request = try fixture("opencode-1.15.11-tool-result.json")
        let validated = try OpenAIRequestValidator.validate(
            request, modelID: "gemma-4-26b-a4b-it")
        #expect(validated.messages.count == 4)
        #expect(validated.messages[2].toolCalls.count == 1)
        #expect(validated.messages[3].toolCallID == "call_0123456789abcdef01234567")
    }

    @Test func capturedOpenCodePromptFits16KWith4096Completion() async throws {
        let request = try fixture("opencode-1.15.11-tool-result.json")
        let validated = try OpenAIRequestValidator.validate(
            request, modelID: "gemma-4-26b-a4b-it")
        let tokenizer = try await MFTokenizer.load()
        let ids = try tokenizer.encodeToolChat(
            messages: validated.messages, tools: validated.tools)
        #expect(ids.count <= 16_384 - 4_096)
    }

    @Test func requiredToolChoiceIsRejected() throws {
        let data = Data(#"""
        {"model":"m","messages":[{"role":"user","content":"x"}],"tool_choice":"required"}
        """#.utf8)
        let request = try JSONDecoder().decode(OpenAIChatRequest.self, from: data)
        #expect(throws: ServerRequestError.self) {
            try OpenAIRequestValidator.validate(request, modelID: "m")
        }
    }

    @Test func acceptsLeadingSystemAndDeveloperGuidance() throws {
        let data = Data(#"""
        {"model":"m","messages":[
          {"role":"system","content":"system"},
          {"role":"developer","content":"developer"},
          {"role":"user","content":"hello"}
        ]}
        """#.utf8)
        let request = try JSONDecoder().decode(OpenAIChatRequest.self, from: data)
        let validated = try OpenAIRequestValidator.validate(request, modelID: "m")
        #expect(validated.messages.map(\.role) == [.system, .developer, .user])
    }

    @Test func rejectsLateDeveloperGuidance() throws {
        let data = Data(#"""
        {"model":"m","messages":[
          {"role":"user","content":"hello"},
          {"role":"developer","content":"late"}
        ]}
        """#.utf8)
        let request = try JSONDecoder().decode(OpenAIChatRequest.self, from: data)
        #expect(throws: ServerRequestError.self) {
            try OpenAIRequestValidator.validate(request, modelID: "m")
        }
    }

    @Test func wideIntegerToolArgumentsRoundTripExactly() async throws {
        let expected = "9007199254740993"
        let parsed = try GemmaToolCallParser().parse(
            "call:lookup{id:\(expected)}",
            allowedTools: ["lookup"],
            id: "call_0123456789abcdef01234567")
        #expect(parsed.argumentsJSON.contains(#""id":\#(expected)"#))
        let signedMinimum = String(Int64.min)
        let signedMaximum = String(Int64.max)
        let unsignedMaximum = String(UInt64.max)
        let edges = try GemmaToolCallParser().parse(
            "call:lookup{minimum:\(signedMinimum),maximum:\(signedMaximum),unsigned:\(unsignedMaximum)}",
            allowedTools: ["lookup"],
            id: "call_0123456789abcdef01234568")
        #expect(edges.arguments.objectValue?["minimum"] == .integer(.min))
        #expect(edges.arguments.objectValue?["maximum"] == .integer(.max))
        #expect(edges.arguments.objectValue?["unsigned"] == .unsignedInteger(.max))
        let encodedEdges = try edges.arguments.encoded()
        #expect(encodedEdges.contains(signedMinimum))
        #expect(encodedEdges.contains(signedMaximum))
        #expect(encodedEdges.contains(unsignedMaximum))
        #expect(try JSONDecoder().decode(
            JSONValue.self,
            from: Data(encodedEdges.utf8)) == edges.arguments)
        for malformed in ["+1", "01", "1.", ".1", "1e", "--1"] {
            #expect(throws: GemmaToolCallParserError.self) {
                try GemmaToolCallParser().parse(
                    "call:lookup{id:\(malformed)}",
                    allowedTools: ["lookup"],
                    id: "call_0123456789abcdef01234570")
            }
        }

        let data = Data(#"""
        {
          "model":"m",
          "messages":[
            {"role":"user","content":"lookup"},
            {"role":"assistant","tool_calls":[{
              "id":"call_0123456789abcdef01234567",
              "type":"function",
              "function":{"name":"lookup","arguments":"{\"id\":9007199254740993}"}
            }]}
          ],
          "tools":[{
            "type":"function",
            "function":{
              "name":"lookup",
              "parameters":{"type":"object","properties":{"id":{"type":"integer"}}}
            }
          }]
        }
        """#.utf8)
        let request = try JSONDecoder().decode(OpenAIChatRequest.self, from: data)
        let validated = try OpenAIRequestValidator.validate(request, modelID: "m")
        let call = try #require(validated.messages[1].toolCalls.first)
        #expect(try call.arguments.encoded().contains(#""id":\#(expected)"#))
        let tokenizer = try await MFTokenizer.load()
        let rendered = tokenizer.decode(
            try tokenizer.encodeToolChat(
                messages: validated.messages,
                tools: validated.tools),
            skipSpecialTokens: false)
        #expect(rendered.contains(expected))

        let unrepresentableHistory = Data(#"""
        {
          "model":"m",
          "messages":[
            {"role":"user","content":"lookup"},
            {"role":"assistant","tool_calls":[{
              "id":"call_0123456789abcdef01234569",
              "type":"function",
              "function":{"name":"lookup","arguments":"{\"id\":18446744073709551615}"}
            }]}
          ],
          "tools":[{
            "type":"function",
            "function":{
              "name":"lookup",
              "parameters":{"type":"object","properties":{"id":{"type":"integer"}}}
            }
          }]
        }
        """#.utf8)
        let rejected = try JSONDecoder().decode(
            OpenAIChatRequest.self,
            from: unrepresentableHistory)
        #expect(throws: ServerRequestError.self) {
            try OpenAIRequestValidator.validate(rejected, modelID: "m")
        }
    }

    @Test func acceptedNonIdentifierParameterKeysParseAndRender() async throws {
        let data = Data(#"""
        {
          "model":"m",
          "messages":[{"role":"user","content":"lookup"}],
          "tools":[{
            "type":"function",
            "function":{
              "name":"lookup",
              "parameters":{
                "type":"object",
                "properties":{
                  "$id":{"type":"string"},
                  "file-path":{"type":"string"},
                  "nested":{"type":"object","properties":{"child-key":{"type":"integer"}}}
                }
              }
            }
          }]
        }
        """#.utf8)
        let request = try JSONDecoder().decode(OpenAIChatRequest.self, from: data)
        let validated = try OpenAIRequestValidator.validate(request, modelID: "m")
        let tokenizer = try await MFTokenizer.load()
        _ = try tokenizer.encodeToolChat(
            messages: validated.messages,
            tools: validated.tools)
        let parsed = try GemmaToolCallParser().parse(
            #"call:lookup{$id:<|"|>item<|"|>,file-path:<|"|>/tmp/x<|"|>,nested:{child-key:7}}"#,
            allowedTools: ["lookup"],
            id: "call_0123456789abcdef01234567")
        #expect(parsed.arguments.objectValue?["$id"] == .string("item"))
        #expect(parsed.arguments.objectValue?["file-path"] == .string("/tmp/x"))
    }

    @Test func unionAndTypelessToolSchemasRenderWithoutThrowing() async throws {
        // pi's `mcp` tool (anyOf string|object), a kagi-style nullable integer
        // (anyOf integer|null), a github-style anyOf string|array, and a bare
        // enum with no `type` — every shape that used to abort Gemma rendering.
        let data = Data(#"""
        {
          "model":"m",
          "messages":[{"role":"user","content":"hi"}],
          "tools":[{
            "type":"function",
            "function":{
              "name":"mcp",
              "description":"gateway",
              "parameters":{
                "type":"object",
                "properties":{
                  "args":{"description":"tool args","anyOf":[
                    {"type":"string"},
                    {"type":"object","properties":{},"additionalProperties":true}
                  ]},
                  "limit":{"anyOf":[{"type":"integer"},{"type":"null"}]},
                  "files":{"anyOf":[
                    {"type":"string"},
                    {"type":"array","items":{"type":"string"}}
                  ]},
                  "mode":{"enum":["a","b"]},
                  "nick":{"type":["string","null"]}
                }
              }
            }
          }]
        }
        """#.utf8)
        let request = try JSONDecoder().decode(OpenAIChatRequest.self, from: data)
        let validated = try OpenAIRequestValidator.validate(request, modelID: "m")
        let tokenizer = try await MFTokenizer.load()
        let rendered = tokenizer.decode(
            try tokenizer.encodeToolChat(
                messages: validated.messages, tools: validated.tools),
            skipSpecialTokens: false)
        #expect(rendered.contains("mcp"))
        #expect(rendered.contains("args"))
    }

    @Test func unionSchemaCollapsesToFirstConcreteBranch() throws {
        let schema = try JSONDecoder().decode(JSONValue.self, from: Data(#"""
        {"description":"d","anyOf":[{"type":"string"},{"type":"object"}]}
        """#.utf8))
        let normalized = schema.gemmaSchemaNormalized().objectValue
        #expect(normalized?["type"] == .string("string"))
        #expect(normalized?["description"] == .string("d"))
        #expect(normalized?["anyOf"] == nil)
    }

    @Test func nullableTypeArrayCollapsesToConcreteType() throws {
        let schema = try JSONDecoder().decode(JSONValue.self, from: Data(#"""
        {"type":["null","integer"]}
        """#.utf8))
        #expect(schema.gemmaSchemaNormalized().objectValue?["type"] == .string("integer"))
    }

    @Test func typelessSchemaDefaultsByShape() throws {
        let object = try JSONDecoder().decode(JSONValue.self, from: Data(#"""
        {"properties":{"a":{"type":"string"}}}
        """#.utf8))
        #expect(object.gemmaSchemaNormalized().objectValue?["type"] == .string("object"))
        let scalar = try JSONDecoder().decode(JSONValue.self, from: Data(#"{"title":"x"}"#.utf8))
        #expect(scalar.gemmaSchemaNormalized().objectValue?["type"] == .string("string"))
    }

    @Test func nullableUnionPrefersTheConcreteBranch() throws {
        // A NULL-typed parameter tells the model the argument must be `null`,
        // so a `{"type":"null"}` branch may never win over a real type.
        let schema = try JSONDecoder().decode(JSONValue.self, from: Data(#"""
        {"anyOf":[{"type":"null"},{"type":"integer"}]}
        """#.utf8))
        #expect(schema.gemmaSchemaNormalized().objectValue?["type"] == .string("integer"))
    }

    @Test func unionOfNullAloneKeepsTheNullBranch() throws {
        let schema = try JSONDecoder().decode(JSONValue.self, from: Data(#"""
        {"anyOf":[{"type":"null"}]}
        """#.utf8))
        #expect(schema.gemmaSchemaNormalized().objectValue?["type"] == .string("null"))
    }

    @Test func unionCollapseNormalizesKeysMergedFromTheParent() throws {
        let schema = try JSONDecoder().decode(JSONValue.self, from: Data(#"""
        {"allOf":[{"type":"object"}],"properties":{"x":{"anyOf":[{"type":"string"}]}}}
        """#.utf8))
        let normalized = schema.gemmaSchemaNormalized().objectValue
        #expect(normalized?["type"] == .string("object"))
        let property = normalized?["properties"]?.objectValue?["x"]?.objectValue
        #expect(property?["type"] == .string("string"))
        #expect(property?["anyOf"] == nil)
    }

    @Test func unresolvableUnionFallsBackToTheShapeDefault() throws {
        let schema = try JSONDecoder().decode(JSONValue.self, from: Data(#"""
        {"oneOf":[],"properties":{"x":{"anyOf":[{"type":"string"}]}}}
        """#.utf8))
        let normalized = schema.gemmaSchemaNormalized().objectValue
        #expect(normalized?["type"] == .string("object"))
        #expect(normalized?["properties"]?.objectValue?["x"]?.objectValue?["type"]
                == .string("string"))
    }

    @Test func intersectionMergesPropertiesFromEveryBranch() throws {
        let schema = try JSONDecoder().decode(JSONValue.self, from: Data(#"""
        {"allOf":[
          {"type":"object","properties":{"a":{"type":"string"}}},
          {"type":"object","properties":{"b":{"type":"integer"}},"description":"d"}
        ]}
        """#.utf8))
        let normalized = schema.gemmaSchemaNormalized().objectValue
        #expect(normalized?["type"] == .string("object"))
        let properties = normalized?["properties"]?.objectValue
        #expect(properties?["a"]?.objectValue?["type"] == .string("string"))
        #expect(properties?["b"]?.objectValue?["type"] == .string("integer"))
        #expect(normalized?["description"] == .string("d"))
        #expect(normalized?["allOf"] == nil)
    }

    @Test func normalizationIsIdempotent() throws {
        let schemas = [
            #"{"anyOf":[{"type":"null"},{"type":"integer"}]}"#,
            #"{"allOf":[{"type":"object"}],"properties":{"x":{"anyOf":[{"type":"string"}]}}}"#,
            #"{"oneOf":[],"properties":{"x":{"anyOf":[{"type":"string"}]}}}"#,
            #"{"allOf":[{"properties":{"a":{"type":"string"}}},{"properties":{"b":{}}}]}"#,
            #"{"type":["string","null"],"items":{"anyOf":[{"type":"integer"}]}}"#,
        ]
        for text in schemas {
            let schema = try JSONDecoder().decode(JSONValue.self, from: Data(text.utf8))
            let once = schema.gemmaSchemaNormalized()
            #expect(once.gemmaSchemaNormalized() == once, "not idempotent: \(text)")
        }
    }

    @Test func deeplyNestedToolSchemaIsRejectedBeforeItRecurses() throws {
        // The unauthenticated shape that used to take the process down: the
        // nesting reaches `parameters`, which decodes through the recursive
        // `JSONValue`, and a stack overflow is not catchable.
        let nesting = JSONValue.maximumDepth + 8
        let data = Data(#"""
        {"model":"m","max_tokens":1,"messages":[{"role":"user","content":"x"}],
         "tools":[{"type":"function","function":{"name":"f","description":"d",
         "parameters":{"type":"object","properties":{"p":{"x":\#(String(repeating: "[", count: nesting))\#(String(repeating: "]", count: nesting))}}}}}]}
        """#.utf8)
        #expect(throws: DecodingError.self) {
            try JSONDecoder().decode(OpenAIChatRequest.self, from: data)
        }
    }

    @Test func consecutiveSystemMessagesCoalesceButKeepDeveloperDistinct() throws {
        let data = Data(#"""
        {"model":"m","messages":[
          {"role":"system","content":"first"},
          {"role":"system","content":"second"},
          {"role":"developer","content":"dev"},
          {"role":"user","content":"hello"}
        ]}
        """#.utf8)
        let request = try JSONDecoder().decode(OpenAIChatRequest.self, from: data)
        let validated = try OpenAIRequestValidator.validate(request, modelID: "m")
        #expect(validated.messages.map(\.role) == [.system, .developer, .user])
        #expect(validated.messages.first?.content == "first\n\nsecond")
    }

    @Test func consecutiveSystemMessagesCoalesceForChatML() throws {
        let data = Data(#"""
        {"model":"m","messages":[
          {"role":"system","content":"first"},
          {"role":"system","content":"second"},
          {"role":"user","content":"hello"}
        ]}
        """#.utf8)
        let request = try JSONDecoder().decode(OpenAIChatRequest.self, from: data)
        let validated = try OpenAIRequestValidator.validate(
            request, modelID: "m", dialect: .chatml)
        #expect(validated.messages.map(\.role) == [.system, .user])
        #expect(validated.messages.first?.content == "first\n\nsecond")
    }

    @Test func developerGuidanceBecomesSystemForChatML() throws {
        let data = Data(#"""
        {"model":"m","messages":[
          {"role":"system","content":"system"},
          {"role":"developer","content":"developer"},
          {"role":"user","content":"hello"}
        ]}
        """#.utf8)
        let request = try JSONDecoder().decode(OpenAIChatRequest.self, from: data)
        let validated = try OpenAIRequestValidator.validate(
            request, modelID: "m", dialect: .chatml)
        #expect(validated.messages.map(\.role) == [.system, .user])
        #expect(validated.messages.first?.content == "system\n\ndeveloper")
    }

    @Test func soleDeveloperGuidanceBecomesSystemForChatML() throws {
        let data = Data(#"""
        {"model":"m","messages":[
          {"role":"developer","content":"guidance"},
          {"role":"user","content":"hello"}
        ]}
        """#.utf8)
        let request = try JSONDecoder().decode(OpenAIChatRequest.self, from: data)
        let validated = try OpenAIRequestValidator.validate(
            request, modelID: "m", dialect: .chatml)
        #expect(validated.messages.map(\.role) == [.system, .user])
        #expect(validated.messages.first?.content == "guidance")
    }

    @Test func developerGuidanceBecomesSystemForDeepseek() throws {
        let data = Data(#"""
        {"model":"m","messages":[
          {"role":"system","content":"system"},
          {"role":"developer","content":"developer"},
          {"role":"user","content":"hello"}
        ]}
        """#.utf8)
        let request = try JSONDecoder().decode(OpenAIChatRequest.self, from: data)
        let validated = try OpenAIRequestValidator.validate(
            request, modelID: "m", dialect: .deepseek)
        #expect(validated.messages.map(\.role) == [.system, .user])
        #expect(validated.messages.first?.content == "system\n\ndeveloper")
    }

    @Test func rejectsLateDeveloperGuidanceForChatML() throws {
        let data = Data(#"""
        {"model":"m","messages":[
          {"role":"user","content":"hello"},
          {"role":"developer","content":"late"}
        ]}
        """#.utf8)
        let request = try JSONDecoder().decode(OpenAIChatRequest.self, from: data)
        #expect(throws: ServerRequestError.self) {
            try OpenAIRequestValidator.validate(request, modelID: "m", dialect: .chatml)
        }
    }

    @Test func chatMLDialectKeepsUnionBranchesInTheToolSchema() throws {
        // ChatML renders `tool | tojson`, so the union must survive validation
        // untouched; only the Gemma render path flattens it.
        let data = Data(#"""
        {
          "model":"m",
          "messages":[{"role":"user","content":"hi"}],
          "tools":[{
            "type":"function",
            "function":{
              "name":"mcp",
              "parameters":{
                "type":"object",
                "properties":{
                  "args":{"anyOf":[{"type":"string"},{"type":"object"}]}
                }
              }
            }
          }]
        }
        """#.utf8)
        let request = try JSONDecoder().decode(OpenAIChatRequest.self, from: data)
        let validated = try OpenAIRequestValidator.validate(
            request, modelID: "m", dialect: .chatml)
        let properties = validated.tools[0].parameters
            .objectValue?["properties"]?.objectValue
        #expect(properties?["args"]?.objectValue?["anyOf"] != nil)
    }

    @Test func deepseekDialectKeepsUnionBranchesInTheToolSchema() throws {
        // DeepSeek serializes the whole schema to JSON in its native tools
        // section, so the union must survive validation untouched too.
        let data = Data(#"""
        {
          "model":"m",
          "messages":[{"role":"user","content":"hi"}],
          "tools":[{
            "type":"function",
            "function":{
              "name":"mcp",
              "parameters":{
                "type":"object",
                "properties":{
                  "args":{"anyOf":[{"type":"string"},{"type":"object"}]}
                }
              }
            }
          }]
        }
        """#.utf8)
        let request = try JSONDecoder().decode(OpenAIChatRequest.self, from: data)
        let validated = try OpenAIRequestValidator.validate(
            request, modelID: "m", dialect: .deepseek)
        let properties = validated.tools[0].parameters
            .objectValue?["properties"]?.objectValue
        #expect(properties?["args"]?.objectValue?["anyOf"] != nil)
    }

    @Test func ambiguousParameterKeysPassValidationForChatML() throws {
        let data = Data(#"""
        {
          "model":"m",
          "messages":[{"role":"user","content":"lookup"}],
          "tools":[{
            "type":"function",
            "function":{
              "name":"lookup",
              "parameters":{
                "type":"object",
                "allOf":[{
                  "type":"object",
                  "properties":{"bad:key":{"type":"string"}}
                }]
              }
            }
          }]
        }
        """#.utf8)
        let request = try JSONDecoder().decode(OpenAIChatRequest.self, from: data)
        let validated = try OpenAIRequestValidator.validate(
            request, modelID: "m", dialect: .chatml)
        #expect(validated.tools.count == 1)
    }

    @Test func ambiguousParameterKeysPassValidationForDeepseek() throws {
        let data = Data(#"""
        {
          "model":"m",
          "messages":[{"role":"user","content":"lookup"}],
          "tools":[{
            "type":"function",
            "function":{
              "name":"lookup",
              "parameters":{
                "type":"object",
                "allOf":[{
                  "type":"object",
                  "properties":{"bad:key":{"type":"string"}}
                }]
              }
            }
          }]
        }
        """#.utf8)
        let request = try JSONDecoder().decode(OpenAIChatRequest.self, from: data)
        let validated = try OpenAIRequestValidator.validate(
            request, modelID: "m", dialect: .deepseek)
        #expect(validated.tools.count == 1)
    }

    @Test func ambiguousParameterKeysFailValidation() throws {
        let data = Data(#"""
        {
          "model":"m",
          "messages":[{"role":"user","content":"lookup"}],
          "tools":[{
            "type":"function",
            "function":{
              "name":"lookup",
              "parameters":{
                "type":"object",
                "allOf":[{
                  "type":"object",
                  "properties":{"bad:key":{"type":"string"}}
                }]
              }
            }
          }]
        }
        """#.utf8)
        let request = try JSONDecoder().decode(OpenAIChatRequest.self, from: data)
        #expect(throws: ServerRequestError.self) {
            try OpenAIRequestValidator.validate(request, modelID: "m")
        }
    }

    private func fixture(_ name: String) throws -> OpenAIChatRequest {
        let url = try #require(Bundle.module.url(
            forResource: name, withExtension: nil, subdirectory: "Fixtures"))
        return try JSONDecoder().decode(OpenAIChatRequest.self, from: Data(contentsOf: url))
    }
}

@Suite("Gemma tool calls")
struct GemmaToolCallTests {
    @Test func parsesNestedArgumentsAndGemmaQuotes() throws {
        let parsed = try GemmaToolCallParser().parse(
            #"call:read{path:<|"|>/tmp/ü"<|"|>,options:{lines:[1,2],exact:true}}"#,
            allowedTools: ["read"],
            id: "call_0123456789abcdef01234567")
        #expect(parsed.name == "read")
        #expect(parsed.argumentsJSON.contains(#""path":"/tmp/ü\"""#))
        #expect(parsed.argumentsJSON.contains(#""exact":true"#))
    }

    @Test func unknownToolFailsClosed() {
        #expect(throws: GemmaToolCallParserError.unknownTool("write")) {
            try GemmaToolCallParser().parse(
                "call:write{path:<|\"|>/tmp/x<|\"|>}",
                allowedTools: ["read"],
                id: "call_0123456789abcdef01234567")
        }
    }

    @Test func parsesJSONUnicodeEscapesAndSurrogatePairs() throws {
        let parsed = try GemmaToolCallParser().parse(
            #"call:read{path:"\u00fc-\ud83c\udf33",note:"a\b\f"}"#,
            allowedTools: ["read"],
            id: "call_0123456789abcdef01234567")
        #expect(parsed.argumentsJSON.contains(#""path":"ü-🌳""#))
        #expect(parsed.argumentsJSON.contains(#""note":"a\b\f""#))
    }

    @Test func suppressesThoughtBlockAndExposesTextAfterChannelClose() async throws {
        let tokenizer = try await MFTokenizer.load()
        let decoder = StructuredAssistantDecoder(tokenizer: tokenizer, allowedTools: [])
        #expect(try decoder.consume(tokenID: tokenizer.channelStartID, delta: "").isEmpty)
        #expect(try decoder.consume(tokenID: tokenizer.bosID, delta: "thought").isEmpty)
        #expect(try decoder.consume(tokenID: tokenizer.bosID, delta: "\n").isEmpty)
        #expect(try decoder.consume(tokenID: tokenizer.bosID, delta: "private").isEmpty)
        #expect(try decoder.consume(tokenID: tokenizer.channelEndID, delta: "").isEmpty)
        #expect(try decoder.consume(tokenID: tokenizer.bosID, delta: "visible") == [
            .content("visible"),
        ])
    }
}

@Suite("Streaming stop matcher")
struct StreamingStopMatcherTests {
    @Test func withholdsCrossChunkStop() {
        var matcher = StreamingStopMatcher(stops: ["END"])
        #expect(matcher.push("hello E") == "hello ")
        #expect(matcher.push("N") == "")
        #expect(matcher.push("D ignored") == "")
        #expect(matcher.isStopped)
    }

    @Test func flushesUnicodeTail() {
        var matcher = StreamingStopMatcher(stops: ["🌳stop"])
        #expect(matcher.push("hello 🌳") == "hello ")
        #expect(matcher.finish() == "🌳")
    }
}

@Suite("Server arguments")
struct ServerArgumentTests {
    @Test func usageListsInstalledModelFamilies() {
        for modelID in [
            "gemma-4-26b-a4b-it",
            "qwen3.6-35b-a3b",
            "deepseek-v4-flash-2bit-dq",
            "inkling-small-4bit",
            "maple-preview-2bit-mlx",
        ] {
            #expect(ServerArguments.usage.contains(modelID))
        }
    }

    @Test func defaults() throws {
        let arguments = try ServerArguments.parse(["--model", "model.gturbo"])
        #expect(arguments.port == 8080)
        #expect(arguments.bindMode == .loopback)
        #expect(arguments.maxContext == 16_384)
        #expect(arguments.queueLimit == 4)
        #expect(arguments.promptCacheMode == .singlePrefix)
    }

    @Test func parsesAndAdvertises128KContext() throws {
        let arguments = try ServerArguments.parse([
            "--model", "model.gturbo",
            "--max-context", "128000",
        ])
        #expect(arguments.maxContext == 128_000)
        #expect(ServerArguments.usage.contains("128000"))
    }

    @Test func parsesSinglePrefixModeAndRejectsUnknownMode() throws {
        let arguments = try ServerArguments.parse([
            "--model", "model.gturbo",
            "--prompt-cache-mode", "single-prefix",
        ])
        #expect(arguments.promptCacheMode == .singlePrefix)
        let rollback = try ServerArguments.parse([
            "--model", "model.gturbo",
            "--prompt-cache-mode", "off",
        ])
        #expect(rollback.promptCacheMode == .off)
        #expect(throws: ServerArgumentError.self) {
            try ServerArguments.parse([
                "--model", "model.gturbo",
                "--prompt-cache-mode", "many",
            ])
        }
    }

    @Test func parsesBindModeAndRejectsUnknownMode() throws {
        let tailnet = try ServerArguments.parse([
            "--model", "model.gturbo",
            "--bind", "tailnet",
        ])
        #expect(tailnet.bindMode == .tailnet)
        let loopback = try ServerArguments.parse([
            "--model", "model.gturbo",
            "--bind", "loopback",
        ])
        #expect(loopback.bindMode == .loopback)
        for rejected in ["public", "0.0.0.0", "lan", ""] {
            #expect(throws: ServerArgumentError.self) {
                try ServerArguments.parse([
                    "--model", "model.gturbo",
                    "--bind", rejected,
                ])
            }
        }
    }
}

@Suite("Server bind mode")
struct ServerBindModeTests {
    @Test func loopbackIgnoresTailscaleAndBindsLoopback() throws {
        let host = try ServerBindMode.loopback.host(tailnetAddresses: { "100.101.102.103\n" })
        #expect(host == "127.0.0.1")
    }

    @Test func tailnetBindsTheDetectedAddress() throws {
        let host = try ServerBindMode.tailnet.host(tailnetAddresses: { "100.101.102.103\n" })
        #expect(host == "100.101.102.103")
    }

    @Test func tailnetFailsWhenTailscaleCannotBeQueried() throws {
        #expect(throws: ServerArgumentError.self) {
            try ServerBindMode.tailnet.host(tailnetAddresses: {
                throw ServerArgumentError.invalid("could not run tailscale")
            })
        }
    }

    @Test func tailnetFailsOnEmptyOutput() throws {
        for output in ["", "\n", "   \n\t"] {
            #expect(throws: ServerArgumentError.self) {
                try ServerBindMode.tailnet.host(tailnetAddresses: { output })
            }
        }
    }

    @Test func tailnetFailsOnMultipleAddresses() throws {
        #expect(throws: ServerArgumentError.self) {
            try ServerBindMode.tailnet.host(tailnetAddresses: {
                "100.101.102.103\n100.64.0.9\n"
            })
        }
    }

    @Test func tailnetFailsOnIPv6OnlyOutput() throws {
        for output in ["fd7a:115c:a1e0::1\n", "::1\n"] {
            #expect(throws: ServerArgumentError.self) {
                try ServerBindMode.tailnet.host(tailnetAddresses: { output })
            }
        }
    }

    @Test func tailnetFailsOnMalformedOutput() throws {
        let malformed = [
            "no addresses available\n",
            "100.101.102\n",
            "100.101.102.103.4\n",
            "100.101.102.999\n",
            "100.101.102.-3\n",
            "100.101.102.0x3\n",
            "100.101.102.103/32\n",
            "100.101.102.103:8080\n",
        ]
        for output in malformed {
            #expect(throws: ServerArgumentError.self) {
                try ServerBindMode.tailnet.host(tailnetAddresses: { output })
            }
        }
    }

    @Test func tailnetRefusesAddressesOutsideTheTailscaleRange() throws {
        let refused = ["0.0.0.0\n", "127.0.0.1\n", "192.168.1.20\n", "10.0.0.4\n", "100.63.255.255\n"]
        for output in refused {
            #expect(throws: ServerArgumentError.self) {
                try ServerBindMode.tailnet.host(tailnetAddresses: { output })
            }
        }
    }
}

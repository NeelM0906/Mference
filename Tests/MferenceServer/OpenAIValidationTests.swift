import Foundation
import Testing
@testable import Mference
@testable import MferenceServerCore

@Suite("OpenAI request validation")
struct OpenAIValidationTests {
    @Test func samplingDefaultsAreSharedAcrossModelFamilies() throws {
        let data = Data(#"{"model":"custom-alias","messages":[{"role":"user","content":"Hi"}]}"#.utf8)
        let decoded = try JSONDecoder().decode(OpenAIChatRequest.self, from: data)
        for dialect: ChatDialect in [.chatml, .minicpm] {
            let request = try OpenAIRequestValidator.validate(decoded, modelID: "custom-alias", dialect: dialect)
            let c = request.generationConfig
            #expect(c.temperature == 0.8 && c.topK == 40 && c.topP == 0.95 && c.minP == 0.05)
            #expect(c.repetitionPenalty == 1 && c.presencePenalty == 0 && c.frequencyPenalty == 0)
            #expect(c.repeatLastN == 64)
            #expect(request.maximumCompletionTokens == 4096)
        }
    }

    @Test func penaltyAliasesAndWindowAreValidated() throws {
        for fields in [#""repeat_penalty":1.1,"repetition_penalty":1.2"#,
                       #""repeat_last_n":-2"#] {
            #expect(throws: ServerRequestError.self) {
                _ = try OpenAIRequestValidator.validate(samplingRequest(fields), modelID: "qwen36-alias",
                    dialect: .chatml, acceptsReasoningEffort: true, qwenReasoning: false)
            }
        }
        let r = try OpenAIRequestValidator.validate(
            samplingRequest(#""repeat_penalty":1.0,"repeat_last_n":-1"#), modelID: "qwen36-alias",
            dialect: .chatml, acceptsReasoningEffort: true, qwenReasoning: false)
        #expect(r.generationConfig.repeatLastN == -1)
    }
    @Test func llamaSamplingFieldsReachGeneration() throws {
        let data = Data(#"{"model":"m","messages":[{"role":"user","content":"Hi"}],"min_p":0.05,"frequency_penalty":0.75}"#.utf8)
        let decoded = try JSONDecoder().decode(OpenAIChatRequest.self, from: data)
        let request = try OpenAIRequestValidator.validate(decoded, modelID: "m")
        #expect(request.generationConfig.temperature == 0.8)
        #expect(request.generationConfig.topK == 40)
        #expect(request.generationConfig.topP == 0.95)
        #expect(request.generationConfig.minP == 0.05)
        #expect(request.generationConfig.frequencyPenalty == 0.75)
        #expect(request.generationConfig.repeatLastN == 64)
    }
    @Test func omittedSamplingFieldsUseSharedDefaultsAndRequestValuesWin() throws {
        func config(_ fields: String) throws -> GenerationConfig {
            let body = #"{"model":"m","messages":[{"role":"user","content":"Hi"}]"# + fields + "}"
            let decoded = try JSONDecoder().decode(OpenAIChatRequest.self, from: Data(body.utf8))
            return try OpenAIRequestValidator.validate(decoded, modelID: "m").generationConfig
        }
        let d = GenerationConfig.defaults
        let omitted = try config("")
        #expect(omitted.temperature == d.temperature && omitted.topK == d.topK && omitted.topP == d.topP)
        #expect(omitted.minP == d.minP && omitted.repetitionPenalty == d.repetitionPenalty)
        #expect(omitted.presencePenalty == d.presencePenalty && omitted.frequencyPenalty == d.frequencyPenalty)
        #expect(omitted.repeatLastN == d.repeatLastN)

        // Every value differs from its default: a default leaking past a
        // client-supplied field fails here.
        let explicit = try config(#","temperature":0.3,"top_k":7,"top_p":0.6,"min_p":0.2,"#
            + #""repetition_penalty":1.2,"presence_penalty":0.4,"frequency_penalty":-0.3,"repeat_last_n":128"#)
        #expect(explicit.temperature == 0.3 && explicit.topK == 7 && explicit.topP == 0.6 && explicit.minP == 0.2)
        #expect(explicit.repetitionPenalty == 1.2 && explicit.presencePenalty == 0.4)
        #expect(explicit.frequencyPenalty == -0.3 && explicit.repeatLastN == 128)
        try explicit.validate()

        // An explicit zero is a value, not an omission.
        let zeros = try config(#","temperature":0,"min_p":0,"repeat_last_n":0,"top_k":0,"top_p":1"#)
        #expect(zeros.temperature == 0 && zeros.minP == 0 && zeros.repeatLastN == 0)
        #expect(zeros.topK == nil && zeros.topP == 1)
    }

    @Test func recommendedSamplingFieldsAreValidatedAndPropagated() throws {
        for value in ["-2.0", "-1.5", "0.0", "1.5", "2.0"] {
            let request = try samplingRequest(#""presence_penalty":\#(value),"min_p":0.0"#)
            let validated = try OpenAIRequestValidator.validate(
                request, modelID: "qwen36-alias", dialect: .chatml,
                acceptsReasoningEffort: true, qwenReasoning: false)
            #expect(validated.generationConfig.presencePenalty == Float(value))
            #expect(validated.generationConfig.minP == 0)
            #expect(validated.generationConfig.repetitionPenalty == 1)
            #expect(validated.generationConfig.temperature == 1)
            #expect(validated.generationConfig.topP == 0.95)
            #expect(validated.generationConfig.topK == 20)
            #expect(validated.reasoningEffort == .xhigh)
            try validated.generationConfig.validate()
        }
        for fields in ["", #""presence_penalty":0,"min_p":0"#] {
            let validated = try OpenAIRequestValidator.validate(
                samplingRequest(fields), modelID: "qwen36-alias", dialect: .chatml,
                acceptsReasoningEffort: true, qwenReasoning: false)
            #expect(validated.generationConfig.presencePenalty == 0)
            #expect(validated.generationConfig.minP == (fields.isEmpty ? 0.05 : 0))
        }
    }

    @Test func samplingValidationRejectsInvalidAndNonfiniteValues() throws {
        for field in ["presence_penalty", "frequency_penalty", "min_p"] {
            let numbers = field == "min_p" ? ["-0.1", "1.01"] : ["-2.01", "2.01"]
            for value in numbers + [#""NaN""#, #""Infinity""#, #""-Infinity""#] {
                let request = try samplingRequest("\"\(field)\":\(value)", allowNonfinite: true)
                do {
                    _ = try OpenAIRequestValidator.validate(
                        request, modelID: "qwen36-alias", dialect: .chatml,
                        acceptsReasoningEffort: true, qwenReasoning: false)
                    Issue.record("accepted invalid \(field)=\(value)")
                } catch let error as ServerRequestError {
                    #expect(error.envelope.error.param == field)
                    #expect(error.envelope.error.code == "invalid_value")
                }
            }
        }
        for field in ["presence_penalty", "frequency_penalty", "min_p"] {
            #expect(throws: DecodingError.self) {
                _ = try samplingRequest("\"\(field)\":\"NaN\"")
            }
        }
    }

    private func samplingRequest(_ fields: String, allowNonfinite: Bool = false) throws -> OpenAIChatRequest {
        let decoder = JSONDecoder()
        if allowNonfinite {
            decoder.nonConformingFloatDecodingStrategy = .convertFromString(
                positiveInfinity: "Infinity", negativeInfinity: "-Infinity", nan: "NaN")
        }
        let extra = fields.isEmpty ? "" : "," + fields
        return try decoder.decode(OpenAIChatRequest.self, from: Data("""
        {"model":"qwen36-alias","messages":[{"role":"user","content":"Hi"}],
         "chat_template_kwargs":{"enable_thinking":true},"temperature":1.0,
         "top_p":0.95,"top_k":20,"repetition_penalty":1.0\(extra)}
        """.utf8))
    }

    @Test func capturedOpenCodeInitialRequestValidates() throws {
        let request = try fixture("opencode-1.15.11-initial.json")
        let validated = try OpenAIRequestValidator.validate(
            request, modelID: request.model)
        #expect(validated.stream)
        #expect(validated.includeUsage)
        #expect(validated.tools.count == 1)
        #expect(validated.maximumCompletionTokens == 4096)
    }

    @Test func capturedOpenCodeToolResultValidates() throws {
        let request = try fixture("opencode-1.15.11-tool-result.json")
        let validated = try OpenAIRequestValidator.validate(
            request, modelID: request.model)
        #expect(validated.messages.count == 4)
        #expect(validated.messages[2].toolCalls.count == 1)
        #expect(validated.messages[3].toolCallID == "call_0123456789abcdef01234567")
    }

    /// The captured tool-result turn renders through the ChatML template:
    /// the historical call and its result reach the prompt as tool-call and
    /// tool-response blocks.
    @Test func capturedOpenCodeToolResultRendersThroughChatML() async throws {
        let request = try fixture("opencode-1.15.11-tool-result.json")
        let validated = try OpenAIRequestValidator.validate(
            request, modelID: request.model)
        let tokenizer = try await ChatMLTokenizerFixture.load()
        let ids = try tokenizer.encodeToolChat(
            messages: validated.messages, tools: validated.tools)
        #expect(ids.contains(tokenizer.toolCallStartID))
        #expect(ids.contains(tokenizer.toolResponseID))
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

    @Test func wideIntegerToolArgumentsRoundTripExactly() async throws {
        let expected = "9007199254740993"
        // Wide integers at the 64-bit edges survive JSONValue encode and decode
        // exactly, and render into the ChatML tool-call block unchanged.
        let signedMinimum = String(Int64.min)
        let signedMaximum = String(Int64.max)
        let unsignedMaximum = String(UInt64.max)
        let edges: JSONValue = .object(["minimum": .integer(.min), "maximum": .integer(.max),
                                        "unsigned": .unsignedInteger(.max)])
        let encodedEdges = try edges.encoded()
        #expect(encodedEdges.contains(signedMinimum))
        #expect(encodedEdges.contains(signedMaximum))
        #expect(encodedEdges.contains(unsignedMaximum))
        #expect(try JSONDecoder().decode(
            JSONValue.self,
            from: Data(encodedEdges.utf8)) == edges)

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
        let tokenizer = try await ChatMLTokenizerFixture.load()
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
        let tokenizer = try await ChatMLTokenizerFixture.load()
        _ = try tokenizer.encodeToolChat(
            messages: validated.messages,
            tools: validated.tools)
    }

    @Test func unionAndTypelessToolSchemasRenderWithoutThrowing() async throws {
        // pi's `mcp` tool (anyOf string|object), a kagi-style nullable integer
        // (anyOf integer|null), a github-style anyOf string|array, and a bare
        // enum with no `type` — union and typeless shapes real clients send.
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
        let tokenizer = try await ChatMLTokenizerFixture.load()
        let rendered = tokenizer.decode(
            try tokenizer.encodeToolChat(
                messages: validated.messages, tools: validated.tools),
            skipSpecialTokens: false)
        #expect(rendered.contains("mcp"))
        #expect(rendered.contains("args"))
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

    /// MiniCPM has no `developer` role either (its template renders only
    /// system / user / assistant / tool), and its XML tool calls carry
    /// free-form parameter names, so the ChatML rules apply verbatim.
    @Test func developerGuidanceBecomesSystemForMiniCPM() throws {
        let data = Data(#"""
        {"model":"m","messages":[
          {"role":"system","content":"system"},
          {"role":"developer","content":"developer"},
          {"role":"user","content":"hello"}
        ]}
        """#.utf8)
        let request = try JSONDecoder().decode(OpenAIChatRequest.self, from: data)
        let validated = try OpenAIRequestValidator.validate(
            request, modelID: "m", dialect: .minicpm)
        #expect(validated.messages.map(\.role) == [.system, .user])
        #expect(validated.messages.first?.content == "system\n\ndeveloper")
    }

    @Test func nonIdentifierParameterKeysAreAcceptedForMiniCPM() throws {
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
                  "file-path":{"type":"string"}
                }
              }
            }
          }]
        }
        """#.utf8)
        let request = try JSONDecoder().decode(OpenAIChatRequest.self, from: data)
        let validated = try OpenAIRequestValidator.validate(
            request, modelID: "m", dialect: .minicpm)
        #expect(validated.tools.map(\.name) == ["lookup"])
        let parsed = try MiniCPMToolCallParser().parse(
            #" name="lookup"><param name="$id">item</param><param name="file-path">/tmp/x</param>"#,
            allowedTools: ["lookup"], id: "call_0")
        #expect(parsed.arguments.objectValue?["$id"] == .string("item"))
        #expect(parsed.arguments.objectValue?["file-path"] == .string("/tmp/x"))
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
        // untouched.
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

    private func fixture(_ name: String) throws -> OpenAIChatRequest {
        let url = try #require(Bundle.module.url(
            forResource: name, withExtension: nil, subdirectory: "Fixtures"))
        return try JSONDecoder().decode(OpenAIChatRequest.self, from: Data(contentsOf: url))
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
            "qwen3.6-35b-a3b",
            "qwen3.8-27b-4bit",
            "maple-preview-2bit-mlx",
            "minicpm5-2b-int4g64",
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
        #expect(arguments.promptCacheMode == .multiPrefix)
    }

    @Test func parsesAndAdvertises128KContext() throws {
        let arguments = try ServerArguments.parse([
            "--model", "model.gturbo",
            "--max-context", "128000",
        ])
        #expect(arguments.maxContext == 128_000)
        #expect(ServerArguments.usage.contains("128000"))
    }

    /// Any length up to the largest native context; each model then enforces
    /// its own limit when it loads.
    @Test(arguments: [1, 100_000, 131_072, 262_144])
    func parsesAnyContextUpToTheLargestNativeLimit(tokens: Int) throws {
        let arguments = try ServerArguments.parse([
            "--model", "model.gturbo",
            "--max-context", String(tokens),
        ])
        #expect(arguments.maxContext == tokens)
    }

    /// `max` gives every model its own native context, so one library can
    /// serve Maple at 128,000 and Qwen 3.6 at 262,144.
    @Test func maxLeavesTheContextToEachModel() throws {
        let arguments = try ServerArguments.parse(["--library", "--max-context", "max"])
        #expect(arguments.maxContext == nil)
        #expect(ServerArguments.usage.contains("max"))
        #expect(ServerModelSession.resolvedMaxContext(nil, family: .qwen36) == 262_144)
        #expect(ServerModelSession.resolvedMaxContext(nil, family: .qwen38) == 262_144)
        #expect(ServerModelSession.resolvedMaxContext(nil, family: .maple) == 128_000)
        #expect(ServerModelSession.resolvedMaxContext(nil, family: .minicpm5) == 131_072)
        #expect(ServerModelSession.resolvedMaxContext(32_768, family: .qwen36) == 32_768)
    }

    /// Inkling 2 Mini on a Mac under 48 GB (the residency plan stages layers)
    /// runs at most 40,960 tokens, whatever `--max-context` asks; from 48 GB it has
    /// its native 262,144. No other family is capped by the host.
    @Test func inkling2ContextIsCappedUnderFortyEightGigabytes() {
        let small: UInt64 = 32 << 30, big: UInt64 = 64 << 30
        #expect(ServerModelSession.resolvedMaxContext(nil, family: .inkling2Loop, physicalMemory: small) == 40_960)
        #expect(ServerModelSession.resolvedMaxContext(65_536, family: .inkling2Loop, physicalMemory: small) == 40_960)
        #expect(ServerModelSession.resolvedMaxContext(16_384, family: .inkling2Loop, physicalMemory: small) == 16_384)
        #expect(ServerModelSession.resolvedMaxContext(nil, family: .inkling2Loop, physicalMemory: big) == 262_144)
        #expect(ServerModelSession.resolvedMaxContext(nil, family: .qwen36, physicalMemory: small) == 262_144)
        #expect(ServerModelSession.resolvedMaxContext(100_000, family: .maple, physicalMemory: small) == 100_000)
        #expect(ServerModelSession.resolvedMaxContext(32_768, family: .inkling2Loop, physicalMemory: small) == 32_768)
    }

    /// The single-model server path resolves every existing family's context
    /// exactly as before the Inkling 2 cap, at the default, explicit and
    /// `max` settings, on every host size.
    @Test func existingFamiliesResolveTheContextTheyHadBefore() {
        let settings: [Int?] = [ServerArguments.defaultMaxContext, 4_096, 40_960, 131_072, 200_000, 262_144, nil]
        for family in [ModelFamily.qwen36, .qwen38, .minicpm5, .maple] {
            for gigabytes: UInt64 in [8, 16, 32, 47, 48, 64] {
                for setting in settings {
                    #expect(ServerModelSession.resolvedMaxContext(setting, family: family,
                                                                  physicalMemory: gigabytes << 30)
                            == setting ?? family.maximumContext)
                }
            }
        }
    }

    @Test(arguments: ["0", "-1", "262145", "1048576", "16k", "", "MAX"])
    func rejectsContextsNoModelCanUse(value: String) {
        #expect(throws: ServerArgumentError.self) {
            try ServerArguments.parse(["--model", "model.gturbo", "--max-context", value])
        }
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

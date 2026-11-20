import Foundation
import Testing
import MferenceEngine

@Suite("EngineCatalog and identifiers")
struct EngineCatalogTests {
    @Test func modelsAreSortedAndLookedUpByIdentifier() throws {
        let catalog = EngineCatalog(models: [FakeEngineBackend.descriptor(id: "b"),
                                             FakeEngineBackend.descriptor(id: "a")])
        #expect(catalog.models.map(\.id) == ["a", "b"])
        #expect(catalog.descriptor(for: "b")?.id == "b")
        #expect(try catalog.resolve("a").id == "a")
        #expect(throws: EngineError.unknownModel("c")) { try catalog.resolve("c") }
    }

    @Test func duplicateIdentifiersKeepTheFirstAndReportTheRest() {
        let first = FakeEngineBackend.descriptor(id: "a")
        let second = EngineModelDescriptor(id: "a", displayName: "a", engine: .mference,
                                           format: first.format,
                                           location: URL(fileURLWithPath: "/zzz/a"),
                                           sizeBytes: 0, contextLength: 1, product: nil)
        let catalog = EngineCatalog(models: [second, first])
        #expect(catalog.models == [first])
        #expect(catalog.skipped.map(\.location) == [second.location])
    }

    @Test func declinedModelsResolveToTheirReason() {
        let catalog = EngineCatalog(models: [], skipped: [
            EngineCatalogSkip(location: URL(fileURLWithPath: "/m"), reason: "gated", modelID: "m"),
        ])
        #expect(catalog.unavailableReason(for: "m") == "gated")
        #expect(throws: EngineError.modelNotRunnable(id: "m", reason: "gated")) { try catalog.resolve("m") }
    }

    @Test func productSlotsHaveStableIdentifiersAndEngines() {
        #expect(EngineProductSlot.allCases.map(\.modelID) == [
            "qwen3.5-4b", "qwen3.8-27b", "qwen3.6-35b-a3b", "minicpm5-2b-int4g64", "maple-preview-2bit-mlx",
        ])
        #expect(Set(EngineProductSlot.allCases.map(\.modelID)).count == EngineProductSlot.allCases.count)
        #expect(EngineProductSlot.allCases.filter { $0.engine == .llamaCpp } == [.quick, .deep])
        for slot in EngineProductSlot.allCases {
            #expect(EngineProductSlot(modelID: slot.modelID) == slot)
        }
        #expect(EngineProductSlot(modelID: "qwen3.8-flash-next-int4g64") == nil)
    }

    @Test func resolvedContextFollowsTheServerForNativeAndCapsGGUF() {
        let native = FakeEngineBackend.descriptor(id: "n", contextLength: 262_144)
        #expect(native.resolvedContext(requested: nil) == 262_144)
        #expect(native.resolvedContext(requested: 16_384) == 16_384)
        let gguf = FakeEngineBackend.descriptor(id: "g", engine: .llamaCpp, contextLength: 262_144)
        #expect(gguf.resolvedContext(requested: nil) == EngineModelDescriptor.defaultGGUFContext)
        #expect(gguf.resolvedContext(requested: 8_192) == 8_192)
        let small = FakeEngineBackend.descriptor(id: "s", engine: .llamaCpp, contextLength: 4_096)
        #expect(small.resolvedContext(requested: nil) == 4_096)
        #expect(small.resolvedContext(requested: 1_000_000) == 4_096)

        let defaults = EngineLoadOptions()
        #expect(native.resolvedContext(options: defaults) == EngineLoadOptions.defaultNativeMaxContext)
        #expect(gguf.resolvedContext(options: defaults) == EngineModelDescriptor.defaultGGUFContext)
        let custom = EngineLoadOptions(nativeMaxContext: nil, ggufMaxContext: 8_192)
        #expect(native.resolvedContext(options: custom) == 262_144)
        #expect(gguf.resolvedContext(options: custom) == 8_192)
    }

    @Test func wireSpellingsAreStable() {
        #expect(EngineKind.mference.rawValue == "mference")
        #expect(EngineKind.llamaCpp.rawValue == "llama.cpp")
        #expect(EngineFinishReason.toolCalls.rawValue == "tool_calls")
        #expect(EnginePromptCacheMode.multiPrefix.rawValue == "multi-prefix")
        #expect(EngineCatalogScope.allCases.map(\.rawValue) == ["product", "all"])
    }

    @Test func toolTemplateIsNeededForToolsAndToolHistory() {
        let plain = EngineRequest(messages: [.init(role: .user, content: "hi")], maximumCompletionTokens: 1)
        #expect(!plain.needsToolTemplate)
        let withTool = EngineRequest(messages: [.init(role: .tool, content: "42")], maximumCompletionTokens: 1)
        #expect(withTool.needsToolTemplate)
    }
}

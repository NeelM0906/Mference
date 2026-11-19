import Foundation
import Metal
import Testing
@testable import Mference

/// Quick's reference pack, env-gated:
///
/// - `MFERENCE_NATIVE_WRITE_PACK=<gguf>|<output.mfpack>`: write the pack with
///   the reference writer (reads the GGUF, writes a new directory; a model run).
/// - `MFERENCE_NATIVE_QUICK_PACK=<quick.mfpack>`: open it with every hash
///   checked. With `MFERENCE_NATIVE_QUICK_GGUF=<gguf>` also re-check every
///   tensor, the tokenizer data and the template against the source GGUF;
///   with `MFERENCE_NATIVE_GOLDENS` (H's `GoldenStore`) tokenize each golden's
///   rendered prompt and compare with llama.cpp's ids.
@Suite("ReferenceQuickPackTests", .serialized)
struct ReferenceQuickPackTests {
    static let env = ProcessInfo.processInfo.environment

    static let quickConfig = Native.Qwen35Config(
        blockCount: 32, hiddenSize: 2560, ffnSize: 9216, headCount: 16, kvHeadCount: 4, keyLength: 256,
        valueLength: 256, rmsNormEpsilon: 1e-6, ropeFreqBase: 1e7, ropeDimensionCount: 64,
        ropeSections: [11, 11, 10, 0], fullAttentionInterval: 4, ssmConvKernel: 4, ssmStateSize: 128,
        ssmGroupCount: 16, ssmTimeStepRank: 32, ssmInnerSize: 4096, contextLength: 262_144, nextnPredictLayers: 0,
        vocabSize: 248_320, tiedEmbeddings: true)

    @Test(.enabled(if: env["MFERENCE_NATIVE_WRITE_PACK"] != nil))
    func writeQuickPack() throws {
        let parts = try #require(Self.env["MFERENCE_NATIVE_WRITE_PACK"]).split(separator: "|").map(String.init)
        try #require(parts.count == 2)
        let start = Date()
        let manifest = try Native.ReferencePackWriter.write(
            gguf: URL(fileURLWithPath: parts[0]), to: URL(fileURLWithPath: parts[1]), label: "quick-ref",
            log: { print("reference pack writer: \($0)") })
        print("reference pack writer: \(manifest.tensors.count) tensors in \(Date().timeIntervalSince(start)) s")
        #expect(manifest.config == Self.quickConfig)
    }

    @Test(.enabled(if: env["MFERENCE_NATIVE_QUICK_PACK"] != nil))
    func quickPackLoadsAndTokenizesLikeLlamaCpp() throws {
        let device = try #require(MetalContext.makeSystemDefaultDevice())
        let pack = URL(fileURLWithPath: try #require(Self.env["MFERENCE_NATIVE_QUICK_PACK"]))
        let store = try Native.ReferenceWeightStore(directory: pack, device: device, verifyHashes: true)
        #expect(store.config == Self.quickConfig)
        #expect(store.manifest.tensors.count == 426)
        #expect(!store.chatTemplate.template.isEmpty)
        let tokenizer = try store.tokenizer.makeTokenizer()
        #expect(tokenizer.vocabSize == 248_320)
        #expect(tokenizer.eosTokenID == 248_046)
        // With the source GGUF, re-check every tensor against its GGUF bytes
        // and the template and tokenizer data against its metadata.
        if let ggufPath = Self.env["MFERENCE_NATIVE_QUICK_GGUF"] {
            let gguf = try Native.ReferenceGGUFFile(url: URL(fileURLWithPath: ggufPath))
            #expect(gguf.tensors.map(\.name) == store.manifest.tensors.map(\.name))
            for (info, t) in zip(gguf.tensors, store.manifest.tensors) {
                #expect(Native.ReferencePackWriter.sha256Hex(gguf.bytes(of: info)) == t.sourceSHA256, "\(t.name)")
                #expect(info.type == t.type && info.shape == t.shape, "\(t.name)")
            }
            #expect(try gguf.tokenizerData() == store.tokenizer.data)
            #expect(gguf.string("tokenizer.chat_template") == store.chatTemplate.template)
            #expect(try gguf.qwen35Config() == store.config)
        }
        guard GoldenStore.root != nil else { return }
        var checked = 0
        for prompt in try GoldenStore.promptSet().prompts {
            let golden = try GoldenStore.open(prompt.id)
            let text = try golden.renderedPrompt()
            let ids = try tokenizer.encode(text, addSpecial: true, parseSpecial: true)
            #expect(ids == golden.promptTokens, "\(golden.id): \(ids.count) vs \(golden.promptTokens.count) tokens")
            let pieces = ids.flatMap { tokenizer.piece($0, special: true) }
            #expect(pieces == Array(text.utf8), "\(golden.id): pieces do not reproduce the prompt")
            checked += 1
        }
        print("reference tokenizer: \(checked) golden prompts checked")
        #expect(checked == 11)
    }
}

import Foundation
import Testing
@testable import MferenceRepackCore

/// Maps the real Unsloth UD-IQ3_XXS header (as committed in
/// `docs/families/qwen38flashnext-unsloth-ud-iq3_xxs.tensors.json`) to
/// `manifest.arch` and checks it against the vendor Flash-Next values.
@Suite
struct GGUFArchInfoTests {
    private static func tableURL() -> URL {
        var root = URL(fileURLWithPath: #filePath)
        for _ in 0..<5 { root.deleteLastPathComponent() }
        return root.appendingPathComponent("docs/families/qwen38flashnext-unsloth-ud-iq3_xxs.tensors.json")
    }

    /// Rebuilds GGUF metadata and tensor infos from the committed table. JSON
    /// loses the GGUF value types, so integers come back as i64 and floats as
    /// f64; the mapper accepts any integer or float width.
    static func load() throws -> ([String: GGUF.Value], [GGUF.TensorInfo]) {
        let data = try Data(contentsOf: tableURL())
        let root = try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
        let meta = try #require(root["metadata"] as? [String: Any])
        func value(_ v: Any) -> GGUF.Value? {
            switch v {
            case let s as String: return .string(s)
            case let n as NSNumber:
                if CFGetTypeID(n) == CFBooleanGetTypeID() { return .bool(n.boolValue) }
                if CFNumberIsFloatType(n) { return .f64(n.doubleValue) }
                return .i64(n.int64Value)
            case let a as [Any]: return .array(a.compactMap(value))
            default: return nil
            }
        }
        var md: [String: GGUF.Value] = [:]
        for (k, v) in meta {
            if let d = v as? [String: Any], let len = d["array_len"] as? Int {
                // Summarized arrays (vocab, merges): only the length matters here.
                md[k] = .array(Array(repeating: .string(""), count: len))
            } else if let parsed = value(v) {
                md[k] = parsed
            }
        }
        let tensors: [GGUF.TensorInfo] = try (root["tensors"] as? [[String: Any]] ?? []).map { t in
            let name = try #require(t["name"] as? String)
            let typeName = try #require(t["type"] as? String)
            let type = try #require(GGMLType.allCases.first { $0.name == typeName })
            let ne = try #require(t["ne"] as? [NSNumber]).map(\.uint64Value)
            return GGUF.TensorInfo(name: name, ne: ne, type: type,
                                   relativeOffset: try #require(t["offset"] as? NSNumber).uint64Value,
                                   absoluteOffset: try #require(t["abs_offset"] as? NSNumber).uint64Value,
                                   sizeBytes: try #require(t["nbytes"] as? NSNumber).uint64Value)
        }
        return (md, tensors)
    }

    @Test func realHeaderMapsToFlashNextArch() throws {
        let (md, tensors) = try Self.load()
        #expect(tensors.count == 1224)
        let a = try ArchInfo.fromGGUFQwen4Exp(metadata: md, tensors: tensors, path: "unsloth")
        #expect(a.family == .qwen38flashnextudiq3xxs)
        #expect(a.hiddenSize == 2560 && a.numLayers == 48 && a.vocabSize == 248_320)
        #expect(a.numHeads == 24 && a.numKVHeads == 2 && a.headDim == 256)
        #expect(a.intermediateSize == 640 && a.moeIntermediateSize == 640)
        #expect(a.numExperts == 512 && a.topKExperts == 10)
        #expect(a.ropeTheta == 10_000_000 && a.partialRotaryFactor == 0.25)
        #expect(a.attentionScale == 0.0625 && a.attnOutputGate && a.sharedExpertGated)
        var mask = [UInt8](repeating: 2, count: 48)
        for i in stride(from: 3, to: 48, by: 4) { mask[i] = 1 }
        #expect(a.fullAttentionLayerMask == mask)
        #expect(a.linearNumKHeads == 16 && a.linearNumVHeads == 48)
        #expect(a.linearKeyHeadDim == 128 && a.linearValueHeadDim == 128 && a.linearConvKernelSize == 4)
        let f = try #require(a.flashNext)
        #expect(f.hcCount == 4 && f.hcLowRank == 320)
        #expect(f.indexerNumHeads == 4 && f.indexerHeadDim == 128 && f.indexerNumKVHeads == 1)
        #expect(f.indexerBudget == 2048 && f.indexerCompressRatio == 4)
        #expect(f.pleLayerIDs == [2] && f.pleConvKernelSize == 4)
        // The two storage-shaped PLE fields that differ from the vendor config.
        #expect(f.pleNgramShardCount == 1 && f.pleNgramVocabSizeBase == 0)
    }

    @Test func missingKeyFailsClosed() throws {
        var (md, tensors) = try Self.load()
        md["qwen4exp.ssm.time_step_rank"] = nil
        #expect(throws: RepackError.self) {
            try ArchInfo.fromGGUFQwen4Exp(metadata: md, tensors: tensors, path: "unsloth")
        }
        (md, tensors) = try Self.load()
        md["qwen4exp.full_attention_interval"] = .i64(3)
        #expect(throws: RepackError.self) {
            try ArchInfo.fromGGUFQwen4Exp(metadata: md, tensors: tensors, path: "unsloth")
        }
    }
}

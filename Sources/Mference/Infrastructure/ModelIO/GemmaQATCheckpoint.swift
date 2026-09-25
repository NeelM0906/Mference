import Foundation

/// Validation and the currently qualified entry points for the pinned QAT checkpoint.
public enum GemmaQATCheckpoint {
    public static let chatTemplateSHA256 = "94899c0f917d93f6fe81c95744d1e8ddab2d21d39228d2e4aec1fb2a25bff413"
    public static let requiredAssets = ["config.json", "tokenizer.json", "tokenizer_config.json",
                                        "chat_template.jinja", "generation_config.json"]

    /// Called only after the manifest has verified the installed source assets.
    /// Source-absent processors stay neutral instead of inheriting the global
    /// Min-P preset. CLI/request overrides are applied afterwards.
    static func generationDefaults(from data: Data) throws -> GenerationConfig {
        struct Source: Decodable {
            let bos_token_id: Int
            let eos_token_id: [Int]
            let pad_token_id: Int
            let do_sample: Bool
            let temperature: Float
            let top_k: Int
            let top_p: Float
        }
        let source = try JSONDecoder().decode(Source.self, from: data)
        guard source.bos_token_id == 2, source.pad_token_id == 0,
              source.eos_token_id == [1, 106, 50], source.do_sample else {
            throw ModelError.indexCorrupt(detail: "Gemma QAT generation token IDs or sampling mode differ from the pinned checkpoint")
        }
        let config = GenerationConfig(temperature: source.temperature,
            topK: source.top_k, topP: source.top_p, repetitionPenalty: 1,
            presencePenalty: 0, frequencyPenalty: 0, minP: 0)
        try config.validate()
        return config
    }

    static func validateTokenizer(_ tokenizer: MFTokenizer) throws {
        guard tokenizer.dialect == .gemma, tokenizer.bosID == 2,
              tokenizer.eosID == 1, tokenizer.padID == 0,
              tokenizer.endOfTurnID == 106, tokenizer.toolResponseID == 50,
              tokenizer.stopTokenIDs == [1, 106, 50], tokenizer.vocabSize == 262_144 else {
            throw ModelError.indexCorrupt(detail: "Gemma QAT tokenizer token IDs differ from the pinned checkpoint")
        }
    }

    static func validate(_ manifest: Manifest, expected: ArchConfig, directory: URL) throws {
        guard expected.family == .gemma4,
              manifest.sourceSnapshotHash == "sha256:7dbbeef0345505798abcf0ac54434116a48c2f1e7aad828071c17a7a871adfe7",
              let quant = manifest.quant else {
            throw ModelError.indexCorrupt(detail: "Gemma QAT identity/source/quantization is inconsistent")
        }
        for (name, slot) in [("embedding", quant.embedding), ("attention", quant.attention),
                             ("sharedExpert", quant.sharedExpert), ("routedExpert", quant.routedExpert)] {
            guard slot.weightBits == 4, slot.groupSize == 32,
                  slot.scheme.lowercased() == "affine", slot.scaleType.lowercased() == "bf16",
                  slot.biasType.lowercased() == "bf16" else {
                throw ModelError.indexCorrupt(detail: "Gemma QAT requires native INT4/group-32 for \(name)")
            }
        }
        let router = quant.router
        guard router.weightBits == 16, router.groupSize == 0,
              router.scheme.lowercased() == "unquantized", router.scaleType.lowercased() == "none",
              router.biasType.lowercased() == "none" else {
            throw ModelError.indexCorrupt(detail: "Gemma QAT requires an unquantized BF16 router without companions")
        }
        for (name, entry) in manifest.files {
            let attributes = try FileManager.default.attributesOfItem(
                atPath: directory.appendingPathComponent(name).path)
            guard attributes[.type] as? FileAttributeType == .typeRegular,
                  (attributes[.size] as? NSNumber)?.uint64Value == entry.size else {
                throw ModelError.indexCorrupt(detail: "Gemma QAT file missing or wrong size: \(name)")
            }
        }
        // Layout is required install metadata. Its integrity must be checked
        // before the temporary inference gate can classify this install.
        let layoutName = "packed_experts/layout.json"
        guard let layoutEntry = manifest.files[layoutName] else {
            throw ModelError.missingFile(name: layoutName)
        }
        guard layoutEntry.size <= PackedExpertsLayoutReader.defaultMaxBytes else {
            throw ModelError.indexCorrupt(detail: "Gemma QAT layout exceeds metadata cap")
        }
        try Sha256Verifier.verifyFile(at: directory.appendingPathComponent(layoutName),
            named: layoutName, expectedHex: layoutEntry.sha256)
        for name in requiredAssets {
            let relative = "tokenizer/" + name
            guard let entry = manifest.files[relative] else { throw ModelError.missingFile(name: relative) }
            let url = directory.appendingPathComponent(relative)
            let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
            guard attributes[.type] as? FileAttributeType == .typeRegular,
                  let size = attributes[.size] as? NSNumber,
                  size.uint64Value == entry.size,
                  entry.size <= (name == "tokenizer.json" ? 64 * 1024 * 1024 : 4 * 1024 * 1024) else {
                throw ModelError.indexCorrupt(detail: "Gemma QAT required asset \(name) has invalid size/type")
            }
            try Sha256Verifier.verifyFile(at: url, named: relative, expectedHex: entry.sha256)
        }
        // Tokenizer/CLI startup and library discovery share this completeness
        // check, so neither can mistake an incomplete receipt for a capability
        // restriction. This does not replace strict verification of weights.
        let manifestData = try Data(contentsOf: directory.appendingPathComponent("manifest.json"))
        let receipt = try VerifiedInstallReceiptReader.load(directoryURL: directory)
        try VerifiedInstallReceiptReader.validate(receipt, directoryURL: directory,
            manifest: manifest, manifestSha256: Sha256Verifier.hashData(manifestData),
            manifestSize: UInt64(manifestData.count))
    }
}

import Foundation

/// Sampler defaults from an installed checkpoint's generation_config.json.
enum CheckpointGeneration {
    /// Top-K for a checkpoint that sets `top_p < 1` without `top_k`: full-vocabulary
    /// nucleus is not implemented, so the nucleus runs inside this window. 64 is
    /// the largest K the fused `topK64Kernel` sampler handles; a wider window takes
    /// the generic kernel, which makes one full-vocabulary pass per K (K = 256 cost
    /// ~1.2 s/token on MiniCPM5's 130,560-row vocab). Maintainer decision
    /// 2026-11-18; a fast wide-K kernel is a follow-up.
    static let nucleusFallbackTopK = 64

    static func defaults(from data: Data) throws -> GenerationConfig {
        struct Source: Decodable {
            let do_sample: Bool?
            let temperature: Float?
            let top_k: Int?
            let top_p: Float?
            let min_p: Float?
            let repetition_penalty: Float?
        }
        let source = try JSONDecoder().decode(Source.self, from: data)
        // HF writes top_k 0 for "disabled".
        let topK = (source.top_k ?? 0) > 0
            ? source.top_k : ((source.top_p ?? 1) < 1 ? Self.nucleusFallbackTopK : nil)
        // Absent processors stay neutral instead of inheriting the global presets.
        let config = GenerationConfig(
            temperature: source.do_sample == false ? 0 : (source.temperature ?? 1),
            topK: topK, topP: source.top_p,
            repetitionPenalty: source.repetition_penalty ?? 1,
            presencePenalty: 0, frequencyPenalty: 0, minP: source.min_p ?? 0)
        try config.validate()
        return config
    }

    static func defaults(inFolder folder: URL?) throws -> GenerationConfig {
        guard let folder else { return .defaults }
        let url = folder.appendingPathComponent("generation_config.json")
        guard FileManager.default.fileExists(atPath: url.path) else { return .defaults }
        return try defaults(from: Data(contentsOf: url))
    }
}

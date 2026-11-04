import Foundation

/// Sampler defaults from an installed checkpoint's generation_config.json.
enum CheckpointGeneration {
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
        let topK = (source.top_k ?? 0) > 0 ? source.top_k : ((source.top_p ?? 1) < 1 ? 256 : nil)
        // Nucleus without top_k takes the widest window: full-vocabulary nucleus is not implemented.
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

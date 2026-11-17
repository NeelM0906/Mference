import Foundation
import Metal
import Testing
@testable import Mference

/// WS4 U5 step 2: per-layer tensors of the GGUF install at chosen positions of
/// a teacher-forced prompt, for comparison with llama.cpp's f32act build
/// (`mf-trace`, outside the repo). Env-gated model run:
/// - `MFERENCE_UNSLOTH_GTURBO`, `MFERENCE_U5_TOKENS` as for the logits test;
/// - `MFERENCE_U5_TRACE_POSITIONS`: comma-separated positions;
/// - `MFERENCE_U5_TRACE_OUT`: output directory.
///
/// For every layer and every name in `names`, writes `<name>-<layer>.f32`,
/// `[n_positions][per_token]` float32 in position order, under the decode
/// capture's names: `attn_hc_mixed`, `block_out`, `mlp_hc_mixed`,
/// `shared_out`, `moe_out`, `router_logits`, `stream_out`. Positions are
/// captured on the production decode path (one token at a time).
@Suite struct FlashNextGGUFLayerTraceTests {
    static let env = ProcessInfo.processInfo.environment
    static let names = ["attn_hc_mixed", "block_out", "mlp_hc_mixed", "shared_out", "moe_out",
                        "router_logits", "stream_out"]
    static var enabled: Bool {
        ["MFERENCE_UNSLOTH_GTURBO", "MFERENCE_U5_TOKENS", "MFERENCE_U5_TRACE_POSITIONS", "MFERENCE_U5_TRACE_OUT"]
            .allSatisfy { env[$0] != nil }
    }

    @Test(.enabled(if: FlashNextGGUFLayerTraceTests.enabled, "set the MFERENCE_U5_TRACE_* variables"))
    func writeLayerTrace() async throws {
        let dirPath = try #require(Self.env["MFERENCE_UNSLOTH_GTURBO"])
        let tokensPath = try #require(Self.env["MFERENCE_U5_TOKENS"])
        let outPath = try #require(Self.env["MFERENCE_U5_TRACE_OUT"])
        let positionsText = try #require(Self.env["MFERENCE_U5_TRACE_POSITIONS"])
        let parsed = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: tokensPath)))
        let tokens = try #require(parsed as? [Int]).map(Int32.init)
        let positions = positionsText.split(separator: ",").compactMap { Int($0) }.sorted()
        #expect(!positions.isEmpty && positions.last! < tokens.count)
        let out = URL(fileURLWithPath: outPath)
        try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)

        let context = try MetalContext()
        let model = try Model.load(directoryURL: URL(fileURLWithPath: dirPath), device: context.device,
                                   streamingMode: .pread(slotCount: 16))
        let runner = try FlashNextForwardRunner(model: model, context: context, maxContext: tokens.count + 8)
        let logits = try #require(context.device.makeBuffer(length: model.config.vocabSize * 2,
                                                            options: .storageModeShared))
        var rows: [String: [Float]] = [:]
        var perToken: [String: Int] = [:]
        let wanted = Set(positions)
        for p in 0...positions.last! {
            if wanted.contains(p) { runner.capture = .init() }
            try await runner.produce(token: tokens[p], position: p, into: logits)
            guard let captured = runner.capture?.floats else { continue }
            runner.capture = nil
            for (key, value) in captured {
                let parts = key.split(separator: ".")
                guard parts.count == 2, parts[0].hasPrefix("layer"), let L = Int(parts[0].dropFirst(5)),
                      Self.names.contains(String(parts[1])) else { continue }
                let file = "\(parts[1])-\(L)"
                rows[file, default: []] += value
                perToken[file] = value.count
            }
        }
        var index: [String: Any] = [:]
        for (file, values) in rows {
            try values.withUnsafeBytes { try Data($0).write(to: out.appendingPathComponent(file + ".f32")) }
            index[file] = ["per_token": perToken[file] ?? 0]
        }
        index["positions"] = positions
        try JSONSerialization.data(withJSONObject: index, options: [.sortedKeys])
            .write(to: out.appendingPathComponent("index.json"))
        print("[u5-trace] positions=\(positions) tensors=\(rows.count)")
    }
}

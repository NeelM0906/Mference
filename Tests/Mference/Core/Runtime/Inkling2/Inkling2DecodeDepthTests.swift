import Foundation
import Metal
import Testing
@testable import Mference
import MferenceValidationSupport

/// Decode at depth against prefill on the real install (a G6b extension the
/// coordinator added on 2026-11-23). G6b checks one decode step, at position
/// 5. This runs one teacher-forced prompt twice in the same configuration
/// (the product's runner and prefill chunk):
///
/// - the decode path prefills the first `split` tokens, then feeds every
///   remaining prompt token one at a time (`produce`, the given token, not a
///   sampled one);
/// - the prefill path prefills the whole prompt and records the logits at
///   every position from `split` on (`positionLogits`, in windows of 256
///   positions with a fresh prefill each, so the buffer stays near 100 MB).
///
/// Per position it compares the two distributions with G6c's metrics: top-1
/// agreement, overall and where the prefill side's margin is at least 0.2;
/// |delta log-prob| of the prefill side's top-1; and the top-20 KL as
/// `compare_goldens.py` approximates it. Per visit it compares the routed
/// expert sets, with each flip's selection margin on the prefill side.
/// Segments split at 1,024 (the window wraps) and 2,048 (the global KV's
/// growth step).
///
/// The default cases are `lp06-raw-1300:64` (decode from position 64, across
/// the window's wrap) and `lp07-raw-2600:1915` (the last 700, across 2,048).
/// `MFERENCE_INKLING2_DECODE_DEPTH_CASES` takes a comma-separated list of
/// `id:split`, with ids from `inputs/generate_prompts_tf.json`. The context
/// is 4,096.
///
/// No bound is asserted yet; the bound comes from this measurement. The test
/// requires every position compared and every value finite. It writes
/// `<id>-<split>.json` to `MFERENCE_INKLING2_DECODE_DEPTH_OUT`. It loads the
/// install, so it runs only with `MFERENCE_INKLING2_GTURBO` set, and then
/// only inside `model-hold.sh`.
@Suite("Inkling2 decode at depth against prefill on the real install", .serialized)
struct Inkling2DecodeDepthTests {
    static let environment = ProcessInfo.processInfo.environment
    static var install: URL? { environment["MFERENCE_INKLING2_GTURBO"].map { URL(fileURLWithPath: $0) } }
    static var output: URL? { environment["MFERENCE_INKLING2_DECODE_DEPTH_OUT"].map { URL(fileURLWithPath: $0) } }
    static let windowPositions = 256
    static let flipMarginLimit: Float = 0.0258

    typealias Top = [(id: Int, logprob: Double)]

    /// compare_goldens.py's per-position measures of `b` against `a`.
    static func compare(_ a: Top, _ b: Top) -> (agree: Bool, margin: Double, dlp: Double, kl: Double) {
        let ta = Dictionary(uniqueKeysWithValues: a.map { ($0.id, $0.logprob) })
        let tb = Dictionary(uniqueKeysWithValues: b.map { ($0.id, $0.logprob) })
        let floorA = a.last!.logprob, floorB = b.last!.logprob
        let top1 = a[0].id
        var kl = 0.0
        for i in Set(ta.keys).union(tb.keys) {
            let pa = ta[i] ?? floorA, pb = tb[i] ?? floorB
            kl += exp(pa) * (pa - pb)
        }
        return (top1 == b[0].id, a[0].logprob - a[1].logprob, abs(a[0].logprob - (tb[top1] ?? floorB)), max(kl, 0))
    }

    /// Nearest-rank percentile, as compare_goldens.py takes it.
    static func percentile(_ values: [Double], _ q: Double) -> Double {
        let s = values.sorted()
        return s[max(0, Int((q * Double(s.count)).rounded(.up)) - 1)]
    }

    struct Stats: Encodable {
        let from: Int
        let to: Int
        let positions: Int
        let top1_agreement: Double
        let top1_at_margin_0_2_agree: Int
        let top1_at_margin_0_2_positions: Int
        let max_dlp: Double
        let mean_dlp: Double
        let p95_dlp: Double
        let mean_kl: Double
        let p95_kl: Double
        let max_kl: Double
    }
    struct Worst: Encodable {
        let pos: Int
        let dlp: Double
        let kl: Double
        let prefill_margin: Double
        let top1_agree: Bool
    }
    struct Experts: Encodable {
        let token_visits: Int
        let differing: Int
        let min_visit_agreement: Double
        let max_flip_margin_prefill: Float
        let flips_at_or_above_0_0258: Int
    }
    struct Summary: Encodable {
        let id: String
        let split: Int
        let tokens: Int
        let chunk: Int
        let overall: Stats
        let segments: [Stats]
        let worst: [Worst]
        let expert_sets: Experts
        let decode_seconds: Double
        let prefill_windows: Int
    }

    static func stats(_ rows: [(pos: Int, agree: Bool, margin: Double, dlp: Double, kl: Double)],
                      from: Int, to: Int) -> Stats? {
        let r = rows.filter { (from..<to).contains($0.pos) }
        guard !r.isEmpty else { return nil }
        let confident = r.filter { $0.margin >= 0.2 }
        return Stats(from: from, to: to, positions: r.count,
                     top1_agreement: Double(r.filter { $0.agree }.count) / Double(r.count),
                     top1_at_margin_0_2_agree: confident.filter { $0.agree }.count,
                     top1_at_margin_0_2_positions: confident.count,
                     max_dlp: r.map { $0.dlp }.max()!, mean_dlp: r.map { $0.dlp }.reduce(0, +) / Double(r.count),
                     p95_dlp: percentile(r.map { $0.dlp }, 0.95),
                     mean_kl: r.map { $0.kl }.reduce(0, +) / Double(r.count), p95_kl: percentile(r.map { $0.kl }, 0.95),
                     max_kl: r.map { $0.kl }.max()!)
    }

    @Test(.enabled(if: install != nil && output != nil,
                   "needs MFERENCE_INKLING2_GTURBO and MFERENCE_INKLING2_DECODE_DEPTH_OUT"))
    func decodeAtDepthAgainstPrefill() async throws {
        let installURL = try #require(Self.install), outURL = try #require(Self.output)
        let promptsURL = Inkling2G6cTests.repositoryRoot.appendingPathComponent(
            "Scripts/parity/inkling2/inputs/generate_prompts_tf.json")
        let prompts = try JSONDecoder().decode([Inkling2G6cTests.Prompt].self, from: Data(contentsOf: promptsURL))
        let byID = Dictionary(uniqueKeysWithValues: prompts.map { ($0.id, $0.ids) })
        let cases = try (Self.environment["MFERENCE_INKLING2_DECODE_DEPTH_CASES"]
            ?? "lp06-raw-1300:64,lp07-raw-2600:1915").split(separator: ",").map { item -> (String, Int) in
                let parts = item.split(separator: ":")
                return (String(parts[0]), try #require(parts.count == 2 ? Int(parts[1]) : nil, "case \(item)"))
            }

        let context = try MetalContext()
        let device = context.device
        let model = try Model.load(directoryURL: installURL, device: device, expecting: .inkling2Mini,
                                   integrityPolicy: .trustedReceiptWhenValid)
        let c = model.config
        let topK = c.topKExperts, experts = c.numExperts
        let vocab = c.unpaddedVocabSize, vocabRows = c.vocabSize
        let softcap = Double(c.finalLogitSoftcap)
        let configuration = RuntimeConfiguration(
            prefillChunkTokens: RuntimeConfiguration.familyPrefillChunkTokens(for: c.family) ?? 128)
        let runner = try Inkling2ForwardRunner(model: model, context: context, maxContext: 4096,
                                               runtimeConfiguration: configuration)
        let biases = try (0..<c.numLayers).map { layer -> [Float] in
            let t = try model.inkling2RouterSelectionBias(layer: layer)
            return Inkling2BufferMatrix.floats(t.buffer, offset: Int(t.offset), count: experts)
        }
        let schedule = Inkling2ReferenceConfig.mini.schedule
        let out = try #require(device.makeBuffer(length: vocabRows * 2, options: .storageModeShared))
        try FileManager.default.createDirectory(at: outURL, withIntermediateDirectories: true)

        func top(_ halves: UnsafePointer<Float16>) throws -> Top {
            let row = Inkling2LogProbRow(halves: halves, count: vocab, softcap: softcap)
            try #require(row.rawFinite && row.logSumExp.isFinite, "non-finite logits")
            return row.top.map { (id: $0, logprob: row.logprob($0)) }
        }

        for (id, split) in cases {
            let ids = try #require(byID[id], "unknown prompt \(id)")
            let n = ids.count
            try #require(split > 0 && split < n && n <= 4096, "\(id): split \(split) of \(n)")
            func trace() throws -> Inkling2ForwardRunner.VisitTrace {
                .init(outputs: nil,
                      routes: try #require(device.makeBuffer(length: 48 * n * topK * 4, options: .storageModeShared)),
                      rows: n,
                      routerLogits: try #require(device.makeBuffer(length: 48 * n * experts * 4,
                                                                   options: .storageModeShared)))
            }

            // The decode path: prefill [0, split), then one token at a time.
            let decodeTrace = try trace()
            runner.reset()
            runner.visitTrace = decodeTrace
            let started = Date()
            _ = try await runner.prefillChunked(tokens: ids[..<split], startPosition: 0, outputMode: .logits,
                                                config: configuration.prefillConfig, into: out, onProgress: { _ in })
            var decodeTops: [Top] = []
            for p in split..<n {
                try await runner.produce(token: ids[p], position: p, into: out)
                decodeTops.append(try top(out.contents().bindMemory(to: Float16.self, capacity: vocab)))
            }
            let decodeSeconds = Date().timeIntervalSince(started)
            runner.visitTrace = nil

            // The prefill path: the whole prompt, logits from `split` on.
            let prefillTrace = try trace()
            var prefillTops: [Top] = []
            var windows = 0
            var lower = split
            while lower < n {
                let upper = min(n, lower + Self.windowPositions)
                let buffer = try #require(device.makeBuffer(length: (upper - lower) * vocabRows * 2,
                                                            options: .storageModeShared))
                runner.reset()
                runner.positionLogits = .init(buffer: buffer, positions: lower..<upper)
                runner.visitTrace = lower == split ? prefillTrace : nil
                _ = try await runner.prefillChunked(tokens: ids[...], startPosition: 0, outputMode: .logits,
                                                    config: configuration.prefillConfig, into: out,
                                                    onProgress: { _ in })
                runner.positionLogits = nil
                runner.visitTrace = nil
                for p in lower..<upper {
                    prefillTops.append(try top((buffer.contents() + (p - lower) * vocabRows * 2)
                        .bindMemory(to: Float16.self, capacity: vocab)))
                }
                windows += 1
                lower = upper
            }
            try #require(prefillTops.count == n - split && decodeTops.count == n - split)

            // Per position.
            let rows = (split..<n).map { p -> (pos: Int, agree: Bool, margin: Double, dlp: Double, kl: Double) in
                let m = Self.compare(prefillTops[p - split], decodeTops[p - split])
                return (p, m.agree, m.margin, m.dlp, m.kl)
            }
            #expect(rows.allSatisfy { $0.dlp.isFinite && $0.kl.isFinite }, "\(id)")
            let overall = try #require(Self.stats(rows, from: split, to: n))
            let segments = [(split, 1024), (1024, 2048), (2048, n)].compactMap { Self.stats(rows, from: $0.0, to: $0.1) }
            let worst = rows.sorted { $0.dlp > $1.dlp }.prefix(5).map {
                Worst(pos: $0.pos, dlp: $0.dlp, kl: $0.kl, prefill_margin: $0.margin, top1_agree: $0.agree)
            }

            // Per visit: the routed expert sets, and each flip's prefill-side margin.
            let a = prefillTrace.routes.contents().bindMemory(to: UInt32.self, capacity: 48 * n * topK)
            let b = decodeTrace.routes.contents().bindMemory(to: UInt32.self, capacity: 48 * n * topK)
            let logits = try #require(prefillTrace.routerLogits).contents()
                .bindMemory(to: Float.self, capacity: 48 * n * experts)
            var differing = 0, highFlips = 0
            var maxFlipMargin: Float = 0, minAgreement = 1.0
            for v in 0..<48 {
                var visitDiffering = 0
                for p in split..<n {
                    let row = v * n + p
                    let setA = Set((0..<topK).map { a[row * topK + $0] }), setB = Set((0..<topK).map { b[row * topK + $0] })
                    guard setA != setB else { continue }
                    visitDiffering += 1
                    let margin = Inkling2G6cTests.selectionMargin(logits: logits + row * experts,
                                                                  bias: biases[schedule[v].layer], topK: topK)
                    maxFlipMargin = max(maxFlipMargin, margin)
                    if !(margin < Self.flipMarginLimit) { highFlips += 1 }
                }
                differing += visitDiffering
                minAgreement = min(minAgreement, 1 - Double(visitDiffering) / Double(n - split))
            }
            let summary = Summary(id: id, split: split, tokens: n, chunk: configuration.prefillChunkTokens,
                                  overall: overall, segments: segments, worst: Array(worst),
                                  expert_sets: Experts(token_visits: 48 * (n - split), differing: differing,
                                                       min_visit_agreement: minAgreement,
                                                       max_flip_margin_prefill: maxFlipMargin,
                                                       flips_at_or_above_0_0258: highFlips),
                                  decode_seconds: decodeSeconds, prefill_windows: windows)
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            try encoder.encode(summary).write(to: outURL.appendingPathComponent("\(id)-\(split).json"))
            print(String(format: "inkling2 decode depth %@ from %d (%d positions): top-1 %.4f, max dlp %.4f, "
                         + "mean dlp %.5f, p95 dlp %.4f, mean KL %.6f, p95 KL %.5f, max KL %.4f; expert sets differ "
                         + "on %d of %d token-visits, largest flip margin %.4f, %d at or above 0.0258; decode %.1f s; "
                         + "peak footprint %llu MiB",
                         id, split, n - split, overall.top1_agreement, overall.max_dlp, overall.mean_dlp,
                         overall.p95_dlp, overall.mean_kl, overall.p95_kl, overall.max_kl, differing,
                         48 * (n - split), maxFlipMargin, highFlips, decodeSeconds,
                         Inkling2G6cTests.footprint().peak >> 20))
            for s in segments {
                print(String(format: "inkling2 decode depth %@ segment [%d, %d): %d positions, top-1 %.4f, "
                             + "max dlp %.4f, mean dlp %.5f, mean KL %.6f, max KL %.4f",
                             id, s.from, s.to, s.positions, s.top1_agreement, s.max_dlp, s.mean_dlp, s.mean_kl,
                             s.max_kl))
            }
        }
    }
}

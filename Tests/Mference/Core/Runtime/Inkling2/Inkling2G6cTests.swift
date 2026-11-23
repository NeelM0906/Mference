import Accelerate
import Foundation
import Metal
import Testing
@testable import Mference
import MferenceValidationSupport

/// Log-probabilities of one row of the runner's logits as K0's goldens
/// record them (`goldens/README.md`; `g6c_verdict.py`): the final softcap
/// `cap * tanh(x / cap)` on the FP16 values the head wrote (the sampler's
/// path: the install's head leaves the cap to it), a log-softmax over the
/// first `count` values (the real vocabulary, padding rows excluded), and
/// the top `k` by logit, ties to the lower id. Accelerate throughout: the
/// tests run as a debug build and a G6c run takes about 1,100 rows of 200,058.
struct Inkling2LogProbRow {
    /// The capped logits.
    let logits: [Double]
    let logSumExp: Double
    /// Ids of the `k` largest logits, largest first, ties to the lower id.
    let top: [Int]
    /// Every value read was finite.
    let rawFinite: Bool

    func logprob(_ id: Int) -> Double { logits[id] - logSumExp }

    init(halves: UnsafePointer<Float16>, count n: Int, softcap: Double, k: Int = 20) {
        precondition(n > k && k > 0)
        var floats = [Float](repeating: 0, count: n)
        floats.withUnsafeMutableBytes { dst in
            var source = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: halves), height: 1,
                                       width: vImagePixelCount(n), rowBytes: n * 2)
            var target = vImage_Buffer(data: dst.baseAddress, height: 1, width: vImagePixelCount(n), rowBytes: n * 4)
            _ = vImageConvert_Planar16FtoPlanarF(&source, &target, vImage_Flags(kvImageNoFlags))
        }
        var x = [Double](repeating: 0, count: n)
        vDSP_vspdp(floats, 1, &x, 1, vDSP_Length(n))
        // A sum of finite FP16 values is finite: one check before the cap
        // (which would turn an infinity into the cap).
        var rawSum = 0.0
        vDSP_sveD(x, 1, &rawSum, vDSP_Length(n))
        rawFinite = rawSum.isFinite
        x.withUnsafeMutableBufferPointer { b in
            let p = b.baseAddress!
            if softcap > 0 {
                var inverse = 1 / softcap, cap = softcap
                var count = Int32(n)
                vDSP_vsmulD(p, 1, &inverse, p, 1, vDSP_Length(n))
                vvtanh(p, p, &count)
                vDSP_vsmulD(p, 1, &cap, p, 1, vDSP_Length(n))
            }
        }
        var maximum = 0.0
        vDSP_maxvD(x, 1, &maximum, vDSP_Length(n))
        var shifted = [Double](repeating: 0, count: n)
        var negated = -maximum
        vDSP_vsaddD(x, 1, &negated, &shifted, 1, vDSP_Length(n))
        var count = Int32(n)
        shifted.withUnsafeMutableBufferPointer { vvexp($0.baseAddress!, $0.baseAddress!, &count) }
        var sum = 0.0
        vDSP_sveD(shifted, 1, &sum, vDSP_Length(n))
        logits = x
        logSumExp = maximum + Foundation.log(sum)

        // One pass, a sorted list of k: an equal later value never displaces
        // or passes an earlier one.
        var best: [Int] = []
        best.reserveCapacity(k + 1)
        x.withUnsafeBufferPointer { b in
            for i in 0..<n {
                let v = b[i]
                if best.count == k, !(v > b[best[k - 1]]) { continue }
                var j = best.count
                while j > 0, b[best[j - 1]] < v { j -= 1 }
                best.insert(i, at: j)
                if best.count > k { best.removeLast() }
            }
        }
        top = best
    }
}

@Suite("Inkling2 log-prob rows as the goldens record them")
struct Inkling2LogProbRowTests {
    @Test
    func cappedLogSoftmaxAndTopKTiesToTheLowerId() throws {
        let n = 64
        var values = (0..<n).map { Float16(Float(($0 * 37) % 23) * 0.1 - 2) }   // at most 0.2
        values[5] = 60          // capped to 30 tanh 2
        values[9] = 3; values[40] = 3; values[12] = 3   // a three-way tie
        values += [Float16(500)]   // past `count`: never read
        let row = values.withUnsafeBufferPointer { Inkling2LogProbRow(halves: $0.baseAddress!, count: n,
                                                                       softcap: 30, k: 4) }
        let capped = values[..<n].map { 30 * tanh(Double(Float($0)) / 30) }
        let lse = Foundation.log(capped.map { exp($0) }.reduce(0, +))
        #expect(row.logits.count == n)
        #expect(zip(row.logits, capped).allSatisfy { abs($0 - $1) < 1e-12 })
        #expect(abs(row.logSumExp - lse) < 1e-12)
        #expect(abs(row.logprob(5) - (30 * tanh(2.0) - lse)) < 1e-12)
        #expect(abs((0..<n).map { exp(row.logprob($0)) }.reduce(0, +) - 1) < 1e-12)
        // 60 first, then the tie at 3 in id order.
        #expect(row.top == [5, 9, 12, 40])
        // No cap: the raw values.
        let raw = values.withUnsafeBufferPointer { Inkling2LogProbRow(halves: $0.baseAddress!, count: n,
                                                                       softcap: 0, k: 2) }
        #expect(raw.logits[5] == 60 && raw.top == [5, 9] && raw.rawFinite)
        values[7] = .nan
        let bad = values.withUnsafeBufferPointer { Inkling2LogProbRow(halves: $0.baseAddress!, count: n,
                                                                       softcap: 30, k: 2) }
        #expect(!bad.rawFinite)
    }
}

/// The runner's position-logits hook (G6c) on the Mini-width toy, over a
/// prefill of two chunks ([0, 4) and [4, 6)) with the window starting inside
/// the first: every row against the CPU reference's logits for that token,
/// teacher-forced on the runner's own visit outputs and routes as in
/// `Inkling2ForwardRunnerToyTests` (the random toy amplifies a different
/// kernel's rounding about twofold every six visits, so only the head is
/// compared); the last row against the production head's output bit for
/// bit; and the first chunk's rows against prefills of the prefix alone,
/// which run the same kernels (chunks of 2 to 4 rows), bit for bit.
/// Production shape (about 1.7 GB of buffers, a slow CPU reference), so full
/// tier.
@Suite("Inkling2ForwardRunner position logits on the toy", .serialized)
struct Inkling2PositionLogitsToyTests {
    /// `Inkling2ForwardRunnerToyTests.logitTolerance`: the runner's last-token
    /// logits against the reference measured at most 3.6e-4.
    static let tolerance: Float = 0.002

    @Test(.fullTier)
    func everyRowIsTheHeadForItsToken() async throws {
        let context = try MetalContext()
        let toy = try Inkling2RunnerToy(device: context.device)
        let c = Inkling2RunnerToy.config
        let d = c.hiddenSize, topK = c.topK, traceRows = 8
        let runner = try Inkling2ForwardRunner(weights: toy.runnerWeights, context: context,
                                               maxContext: 64, maxChunkTokens: 4)
        let device = context.device
        let rowBytes = c.vocabPadded * 2
        let tokens: [Int32] = [17, 401, 3, 250, 99, 7]
        let chunks = [0..<4, 4..<6]
        let positions = 1..<6
        let hook = try #require(device.makeBuffer(length: positions.count * rowBytes, options: .storageModeShared))
        let out = try #require(device.makeBuffer(length: rowBytes, options: .storageModeShared))
        let outputs = try #require(device.makeBuffer(length: 48 * traceRows * d * 4, options: .storageModeShared))
        let routes = try #require(device.makeBuffer(length: 48 * traceRows * topK * 4, options: .storageModeShared))
        func prefill(_ count: Int) async throws {
            runner.reset()
            // The runner caps the spans at its 4-token chunk.
            _ = try await runner.prefillChunked(tokens: tokens[..<count], startPosition: 0, outputMode: .logits,
                                                config: .production(chunkTokens: 64), into: out,
                                                onProgress: { _ in })
        }
        func bits(_ buffer: MTLBuffer, row: Int) -> [UInt16] {
            let p = (buffer.contents() + row * rowBytes).bindMemory(to: UInt16.self, capacity: c.vocabUnpadded)
            return (0..<c.vocabUnpadded).map { p[$0] }
        }
        func floats(_ v: [UInt16]) -> [Float] { v.map { Float(Float16(bitPattern: $0)) } }
        func relative(_ got: [Float], _ want: [Float]) -> Float {
            guard got.count == want.count, got.allSatisfy(\.isFinite), want.allSatisfy(\.isFinite) else {
                return .infinity
            }
            return zip(got, want).map { abs($0 - $1) }.max()! / max(want.map(abs).max()!, 1e-6)
        }

        runner.positionLogits = .init(buffer: hook, positions: positions)
        runner.visitTrace = .init(outputs: outputs, routes: routes, rows: traceRows)
        try await prefill(tokens.count)
        runner.positionLogits = nil
        runner.visitTrace = nil
        let rows = positions.map { bits(hook, row: $0 - positions.lowerBound) }
        #expect(rows.last! == bits(out, row: 0), "the last row is the production head's")

        // Each chunk teacher-forced through the reference, logits for every token.
        let reference = try Inkling2ReferenceModel(config: c, weights: toy.referenceWeights)
        reference.precision = .metalRunner
        let ids = routes.contents().bindMemory(to: UInt32.self, capacity: 48 * traceRows * topK)
        for chunk in chunks {
            reference.forcedRoutes = (0..<48).map { visit in
                chunk.map { t in (0..<topK).map { Int(ids[(visit * traceRows + t) * topK + $0]) } }
            }
            reference.forcedVisitOutputs = (0..<48).map { visit in
                chunk.map { t in Inkling2BufferMatrix.floats(outputs, offset: (visit * traceRows + t) * d * 4, count: d) }
            }
            let want = reference.forward(tokens: Array(tokens[chunk]), logits: .all)
            try #require(want.count == chunk.count)
            for position in chunk where positions.contains(position) {
                let error = relative(floats(rows[position - positions.lowerBound]), want[position - chunk.lowerBound])
                print("inkling2 toy position logits at \(position) against the reference: \(error)")
                #expect(error < Self.tolerance, "position \(position)")
            }
        }

        // The first chunk's rows: prefills of the prefix alone end there.
        for position in positions where position < chunks[0].upperBound {
            try await prefill(position + 1)
            #expect(rows[position - positions.lowerBound] == bits(out, row: 0), "position \(position)")
        }
    }
}

/// G6c: the runner's teacher-forced output on the real install, written in
/// ink-golden's `generate` layout for `Scripts/parity/inkling2/g6c_verdict.py`
/// (K0's schema, option A). For each prompt of
/// `Scripts/parity/inkling2/inputs/generate_prompts_tf.json` (a log-prob
/// golden's prompt plus all but its last generated id, as one ids prompt):
/// one production prefill (chunks of 128, the install's expert store),
/// then `<out>/tf/<id>/golden.json` and `logprobs.json` with the top 20
/// log-probs at every recorded position (the last 80 where the entry has
/// `logprobs_last`). For trace01, trace02 and trace03-tool-result (the
/// prompt ids of `goldens/generate{,2,3}/metal/<id>`): the per-visit running
/// sums and expert sets in `<out>/traces/<id>/`. Then:
///
///     python3 -I Scripts/parity/inkling2/g6c_verdict.py <out>/tf --traces <out>/traces
///
/// Loads the install, so it runs only with `MFERENCE_INKLING2_GTURBO`,
/// `MFERENCE_INKLING2_GOLDENS_DIR` and `MFERENCE_INKLING2_G6C_OUT` set, and
/// then only inside `model-hold.sh`. `MFERENCE_INKLING2_G6C_PROMPTS` (comma
/// separated ids) picks the prompts, in that order; all by default.
@Suite("Inkling2 G6c: the runner's teacher-forced output", .serialized)
struct Inkling2G6cTests {
    static let environment = ProcessInfo.processInfo.environment
    static var install: URL? { environment["MFERENCE_INKLING2_GTURBO"].map { URL(fileURLWithPath: $0) } }
    static var goldens: URL? { environment["MFERENCE_INKLING2_GOLDENS_DIR"].map { URL(fileURLWithPath: $0) } }
    static var output: URL? { environment["MFERENCE_INKLING2_G6C_OUT"].map { URL(fileURLWithPath: $0) } }
    static let traceGoldens = ["trace01": "generate/metal", "trace02": "generate2/metal",
                               "trace03-tool-result": "generate3/metal"]
    /// K0's order: by length, the two long prompts last.
    static let defaultOrder = ["trace01", "trace02", "lp10-multilingual", "g01-capital", "lp09-chat-default",
                               "g06-chat-none", "lp08-code", "lp11-multi-turn", "lp04-chat-thinking",
                               "g05-wetlands", "trace03-tool-result", "lp05-tool-result",
                               "lp06-raw-1300", "lp07-raw-2600"]

    static var repositoryRoot: URL {
        var root = URL(fileURLWithPath: #filePath)
        for _ in 0..<6 { root.deleteLastPathComponent() }
        return root
    }

    struct Prompt: Decodable {
        let id: String
        let kind: String
        let ids: [Int32]
        let logprobs_last: Int?
    }
    struct GoldenIDs: Decodable { let prompt_ids: [Int32] }

    struct Pair: Encodable {
        let id: Int
        let logprob: Double
        func encode(to encoder: Encoder) throws {
            var c = encoder.unkeyedContainer()
            try c.encode(id)
            try c.encode(logprob)
        }
    }
    struct LogProbRecord: Encodable {
        let pos: Int
        let next: Int
        let next_logprob: Double
        let top: [Pair]
    }
    struct GoldenRecord: Encodable {
        let id: String
        let kind: String
        let prompt_ids: [Int32]
        let generated_ids: [Int]
        let logprobs_file: String?
        let logprobs_first_pos: Int?
        let prefill_s: Double
        let runner: String
    }
    struct TraceVisit: Encodable {
        let visit: Int
        let slot: Int
        let layer: Int
        let experts: [[Int]]
    }
    struct TraceRecord: Encodable {
        let hidden_file: String
        let hidden_shape: [Int]
        let hidden_tensor: String
        let visits: [TraceVisit]
    }

    /// This process's physical footprint now and at its peak.
    static func footprint() -> (current: UInt64, peak: UInt64) {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let status = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        guard status == KERN_SUCCESS else { return (0, 0) }
        return (info.phys_footprint, UInt64(max(0, info.ledger_phys_footprint_peak)))
    }

    private func write<T: Encodable>(_ value: T, to url: URL) throws {
        try JSONEncoder().encode(value).write(to: url)
    }

    @Test(.enabled(if: install != nil && goldens != nil && output != nil,
                   "needs MFERENCE_INKLING2_GTURBO, MFERENCE_INKLING2_GOLDENS_DIR and MFERENCE_INKLING2_G6C_OUT"))
    func writeTheRunnersTeacherForcedOutput() async throws {
        let installURL = try #require(Self.install), goldensURL = try #require(Self.goldens)
        let outURL = try #require(Self.output)
        let promptsURL = Self.repositoryRoot.appendingPathComponent(
            "Scripts/parity/inkling2/inputs/generate_prompts_tf.json")
        let prompts = try JSONDecoder().decode([Prompt].self, from: Data(contentsOf: promptsURL))
        let byID = Dictionary(uniqueKeysWithValues: prompts.map { ($0.id, $0) })
        let order = Self.environment["MFERENCE_INKLING2_G6C_PROMPTS"]?.split(separator: ",").map(String.init)
            ?? Self.defaultOrder
        for id in order {
            try #require(byID[id] != nil || Self.traceGoldens[id] != nil, "unknown prompt \(id)")
        }

        let context = try MetalContext()
        let device = context.device
        // `.pread`: the Model never opens an expert file; the runner's store does.
        let model = try Model.load(directoryURL: installURL, device: device, expecting: .inkling2Mini)
        let c = model.config
        let d = c.hiddenSize, topK = c.topKExperts
        let vocab = c.unpaddedVocabSize, vocabRows = c.vocabSize
        let softcap = Double(c.finalLogitSoftcap)
        try #require(softcap == 30 && vocab == 200_058)
        // The product's runner and prefill: what ForwardRunnerFactory builds.
        let configuration = RuntimeConfiguration()
        let runner = try Inkling2ForwardRunner(model: model, context: context, maxContext: 4096,
                                               runtimeConfiguration: configuration)
        let out = try #require(device.makeBuffer(length: vocabRows * 2, options: .storageModeShared))
        let schedule = Inkling2ReferenceConfig.mini.schedule
        print("inkling2 G6c: model loaded, footprint \(Self.footprint().current >> 20) MiB, "
              + "chunk \(configuration.prefillChunkTokens)")

        for id in order {
            let tf = byID[id]
            var ids: [Int32]
            if let tf {
                try #require(tf.kind == "ids")
                ids = tf.ids
            } else {
                ids = []
            }
            if let golden = Self.traceGoldens[id] {
                let traceIDs = try JSONDecoder().decode(GoldenIDs.self, from: Data(contentsOf: goldensURL
                    .appendingPathComponent(golden).appendingPathComponent(id).appendingPathComponent("golden.json")))
                    .prompt_ids
                try #require(tf == nil || traceIDs == ids, "\(id): the trace's prompt is the teacher-forced one")
                ids = traceIDs
            }
            let n = ids.count
            try #require(n > 0 && n <= 4096)
            let first = tf.map { n - 1 - ($0.logprobs_last ?? (n - 1)) }
            try #require(first.map { (0..<n).contains($0) } ?? true)

            var logitRows: MTLBuffer?
            if let first {
                let buffer = try #require(device.makeBuffer(length: (n - first) * vocabRows * 2,
                                                            options: .storageModeShared))
                runner.positionLogits = .init(buffer: buffer, positions: first..<n)
                logitRows = buffer
            }
            var trace: Inkling2ForwardRunner.VisitTrace?
            if Self.traceGoldens[id] != nil {
                trace = .init(outputs: try #require(device.makeBuffer(length: 48 * n * d * 4,
                                                                      options: .storageModeShared)),
                              routes: try #require(device.makeBuffer(length: 48 * n * topK * 4,
                                                                     options: .storageModeShared)),
                              rows: n)
                runner.visitTrace = trace
            }
            runner.reset()
            let started = Date()
            _ = try await runner.prefillChunked(tokens: ids[...], startPosition: 0, outputMode: .logits,
                                                config: configuration.prefillConfig, into: out,
                                                onProgress: { _ in })
            let seconds = Date().timeIntervalSince(started)
            runner.positionLogits = nil
            runner.visitTrace = nil

            if let first, let logitRows {
                let dir = outURL.appendingPathComponent("tf").appendingPathComponent(id)
                try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
                // The hook's last row is the production head's, bit for bit.
                let lastRow = logitRows.contents() + (n - 1 - first) * vocabRows * 2
                #expect(memcmp(lastRow, out.contents(), vocab * 2) == 0, "\(id): last row")
                var records: [LogProbRecord] = []
                var argmax = -1
                for position in first..<n {
                    let p = (logitRows.contents() + (position - first) * vocabRows * 2)
                        .bindMemory(to: Float16.self, capacity: vocab)
                    let row = Inkling2LogProbRow(halves: p, count: vocab, softcap: softcap)
                    let top = row.top.map { Pair(id: $0, logprob: row.logprob($0)) }
                    try #require(row.rawFinite && row.logSumExp.isFinite && top.allSatisfy { $0.logprob.isFinite },
                                 "\(id): non-finite log-probs at \(position)")
                    if position == n - 1 { argmax = row.top[0] }
                    let next = position < n - 1 ? Int(ids[position + 1]) : row.top[0]
                    records.append(LogProbRecord(pos: position, next: next, next_logprob: row.logprob(next), top: top))
                }
                try write(records, to: dir.appendingPathComponent("logprobs.json"))
                try write(GoldenRecord(id: id, kind: "ids", prompt_ids: ids, generated_ids: [argmax],
                                       logprobs_file: "logprobs.json", logprobs_first_pos: first,
                                       prefill_s: seconds, runner: "Inkling2ForwardRunner"),
                          to: dir.appendingPathComponent("golden.json"))
                print("inkling2 G6c \(id): \(n) tokens, log-probs at \(first)..<\(n), next \(argmax)")
            }
            if let trace {
                let dir = outURL.appendingPathComponent("traces").appendingPathComponent(id)
                try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
                // [48][n][d] F32 little-endian, straight from the trace buffer.
                try Data(bytesNoCopy: trace.outputs.contents(), count: 48 * n * d * 4, deallocator: .none)
                    .write(to: dir.appendingPathComponent("hidden_after_visit.f32"))
                let routes = trace.routes.contents().bindMemory(to: UInt32.self, capacity: 48 * n * topK)
                // Eight distinct experts of 128 per visit and token.
                let routesValid = (0..<(48 * n)).allSatisfy { r in
                    let set = Set((0..<topK).map { routes[r * topK + $0] })
                    return set.count == topK && set.allSatisfy { $0 < UInt32(c.numExperts) }
                }
                #expect(routesValid, "\(id): expert ids")
                let visits = (0..<48).map { v in
                    TraceVisit(visit: v, slot: schedule[v].cacheSlot, layer: schedule[v].layer,
                               experts: (0..<n).map { t in (0..<topK).map { Int(routes[(v * n + t) * topK + $0]) } })
                }
                try write(TraceRecord(hidden_file: "hidden_after_visit.f32", hidden_shape: [48, n, d],
                                      hidden_tensor: "the running block sum after each visit (l_out)",
                                      visits: visits),
                          to: dir.appendingPathComponent("trace.json"))
                try write(GoldenRecord(id: id, kind: "ids", prompt_ids: ids, generated_ids: [],
                                       logprobs_file: nil, logprobs_first_pos: nil, prefill_s: seconds,
                                       runner: "Inkling2ForwardRunner"),
                          to: dir.appendingPathComponent("golden.json"))
                print("inkling2 G6c \(id): trace of \(n) tokens written")
            }
            let footprint = Self.footprint()
            print("inkling2 G6c \(id): prefill \(n) tokens in \(String(format: "%.2f", seconds)) s, "
                  + "footprint \(footprint.current >> 20) MiB, peak \(footprint.peak >> 20) MiB")
        }
    }
}

import Darwin
import Foundation
import Metal
import Testing
@testable import Mference

/// Staged-read measurements on the real Inkling 2 Mini install (K2, for the
/// C2 table): the uncached and cached read rates of the staged layers, prefill
/// wall time for short prompts in each mode, and how much of a 2,048-token
/// chunk's staging the GPU waits for. (Decode's slot-cache hit rate and miss
/// cost are measured through the runner's own decode: the CLI with
/// `MFERENCE_INKLING2_DECODE_CACHE_LOG`.) Each stage loads the install, so it
/// runs only with `MFERENCE_INKLING2_GTURBO` set and
/// `MFERENCE_INKLING2_K2_STAGE` naming the stage, and then only inside
/// `model-hold.sh` (one stage per hold):
///
/// * `sweep-uncached`, `sweep-cached`: the staging ring alone over the plan's
///   staged layers, in visit order (cached: two passes, the first fills the
///   file cache);
/// * `short-uncached`, `short-cached`: the runner's prefill of 16, 128 and
///   512 tokens (cached: after a warm-up prefill fills the file cache);
/// * `chunk2048-uncached`, `chunk2048-cached`: a 2,048-token chunk and a
///   4,096-token prompt (two chunks) with the staging timeline;
/// * `all-cached`: prefill fills and decode misses both cached;
/// * `short-vs-sweep`: 16- and 64-token prompts prefilled through the slot
///   caches and with the whole-layer reads forced, in two runners: the
///   next-token logits byte for byte, then 32 greedy tokens each;
/// * `long-decode`: the store's defaults (option D below 48 GB): a 64-token
///   prompt, 1,024 greedy tokens with the slot cache, the staged files'
///   file-cache pages and the host's free memory every 128 tokens, then a
///   512-token prefill continuing the context (uncached slab fills, which
///   empty the slot caches) and 16 more tokens, each one's misses and time.
///
/// The install loads with the receipt's size checks, not full SHA-256: the
/// hash would read every layer file through the file cache first and turn
/// the uncached reads into cache hits.
///
/// Prints `[inkling2-k2-read]` lines; asserts only that runs complete.
@Suite("Inkling 2 staged reads on the real install", .serialized)
struct Inkling2StagedReadRealInstallTests {
    static let environment = ProcessInfo.processInfo.environment
    static let install = environment["MFERENCE_INKLING2_GTURBO"].map { URL(fileURLWithPath: $0) }
    static let stage = environment["MFERENCE_INKLING2_K2_STAGE"] ?? ""
    static let enabled = install != nil && !stage.isEmpty

    static func log(_ line: String) { print("[inkling2-k2-read] \(stage): \(line)") }

    /// This process's physical footprint (what Activity Monitor shows).
    static func footprintGB() -> Double {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let status = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        return status == KERN_SUCCESS ? Double(info.phys_footprint) / 1e9 : -1
    }

    static func swapMB() -> Double {
        var usage = xsw_usage()
        var size = MemoryLayout<xsw_usage>.size
        return sysctlbyname("vm.swapusage", &usage, &size, nil, 0) == 0 ? Double(usage.xsu_used) / 1_048_576 : -1
    }

    /// Host pages (GB): free, file-backed (the file cache), and compressed.
    static func hostMemoryGB() -> (free: Double, fileBacked: Double, compressed: Double) {
        var stats = vm_statistics64_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<vm_statistics64_data_t>.size / MemoryLayout<integer_t>.size)
        let status = withUnsafeMutablePointer(to: &stats) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &count)
            }
        }
        guard status == KERN_SUCCESS else { return (-1, -1, -1) }
        let page = Double(getpagesize()) / 1e9
        return (Double(stats.free_count) * page, Double(stats.external_page_count) * page,
                Double(stats.compressor_page_count) * page)
    }

    /// The fraction of a layer file's pages in the file cache (`mincore` on
    /// a mapping that is never touched, so it pages nothing in).
    static func cachedFraction(_ file: Inkling2ExpertLayerFile) -> Double {
        let fd = open(file.path, O_RDONLY)
        guard fd >= 0 else { return -1 }
        defer { close(fd) }
        let page = Int(getpagesize())
        let length = (Int(file.offset) + file.bytes + page - 1) / page * page
        guard let base = mmap(nil, length, PROT_READ, MAP_SHARED, fd, 0), base != MAP_FAILED else { return -1 }
        defer { munmap(base, length) }
        var vector = [CChar](repeating: 0, count: length / page)
        guard mincore(base, length, &vector) == 0 else { return -1 }
        return Double(vector.filter { $0 & 1 != 0 }.count) / Double(vector.count)
    }

    static func state(_ label: String) {
        log(String(format: "%@: footprint %.2f GB, swap %.2f MB", label, footprintGB(), swapMB()))
    }

    /// The staged layers of this host's plan, as the runner places them.
    static func stagedFiles(_ model: Model) throws -> [(layer: Int, file: Inkling2ExpertLayerFile)] {
        let c = model.config
        let layout = try model.inkling2ExpertLayout()
        let files = (0..<c.numLayers).map { layer in
            Inkling2ExpertLayerFile(
                path: model.directoryURL.appendingPathComponent("packed_experts")
                    .appendingPathComponent(model.packedExpertsLayout.layers[layer].file).path,
                offset: 0, bytes: c.numExperts * Int(layout.layers[layer].expertStride))
        }
        let plan = Inkling2ExpertResidencyPlan.make(
            layerBytes: files.map(\.bytes), loopedLayers: c.inkling2.loopCoreStart..<c.inkling2.loopCoreEnd,
            physicalMemory: ProcessInfo.processInfo.physicalMemory)
        return (0..<c.numLayers).filter { plan.placements[$0] == .staged }.map { ($0, files[$0]) }
    }

    /// Prints a pass of staged visits from the ring's timeline.
    static func report(_ label: String, _ all: [Inkling2StagedExpertLayers.TimelineEntry]) {
        let entries = all.filter { $0.started > 0 && $0.finished >= $0.started }
        if entries.count != all.count { log("\(label): \(all.count - entries.count) visits without a completed read") }
        guard let first = entries.first, let last = entries.last else { return }
        let bytes = entries.reduce(0) { $0 + $1.bytes }
        let span = Double(last.finished - first.started) / 1e9
        let reading = entries.reduce(0.0) { $0 + Double($1.finished - $1.started) / 1e9 }
        let stall = entries.reduce(0.0) { $0 + $1.gpuStallSeconds }
        log(String(format: "%@: %d staged visits, %.2f GB; reads %.3f s (%.2f GB/s while reading), first read to last %.3f s; GPU waited %.3f s",
                   label, entries.count, Double(bytes) / 1e9, reading, Double(bytes) / 1e9 / reading, span, stall))
        let perLayer = entries.map { String(format: "L%d %.0f ms%@", $0.layer, Double($0.finished - $0.started) / 1e6,
                                            $0.gpuStallSeconds > 0 ? String(format: " (wait %.0f)", $0.gpuStallSeconds * 1000) : "") }
        log("\(label) per visit: \(perLayer.joined(separator: ", "))")
    }

    /// Natural text for prompts: the repository's own docs, tokenized.
    static func promptTokens(_ model: Model, count: Int) async throws -> [Int32] {
        let tokenizer = try await MFTokenizer.load(forModelDirectory: model.directoryURL)
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let text = try ["README.md", "AGENTS.md", "docs/ENGINE.md", "docs/FAMILY_CONTRACT.md"]
            .map { try String(contentsOf: root.appendingPathComponent($0), encoding: .utf8) }.joined(separator: "\n\n")
        var tokens = tokenizer.encode(text)
        while tokens.count < count { tokens += tokens }
        return Array(tokens.prefix(count))
    }

    @Test(.enabled(if: enabled && stage.hasPrefix("sweep-"), "needs MFERENCE_INKLING2_GTURBO and a sweep stage"))
    func sweep() throws {
        let mode: Inkling2StagedReadMode = Self.stage == "sweep-cached" ? .cached : .uncached
        let context = try MetalContext()
        let model = try Model.load(directoryURL: try #require(Self.install), device: context.device, expecting: .inkling2Mini,
                                   integrityPolicy: .sizeCheckTrustedReceipt)
        let staged = try Self.stagedFiles(model)
        Self.log("staged layers \(staged.map(\.layer)), in the file cache before: "
                 + staged.map { String(format: "%.0f%%", Self.cachedFraction($0.file) * 100) }.joined(separator: " "))
        Self.state("before")
        let ring = try Inkling2StagedExpertLayers(device: context.device,
                                                  files: Dictionary(uniqueKeysWithValues: staged.map { ($0.layer, $0.file) }),
                                                  slabs: 2, readMode: mode)
        ring.recordsTimeline = true
        for pass in 0..<(mode == .cached ? 2 : 1) {
            ring.resetTimeline()
            let cb = context.queue.makeCommandBuffer()!
            for (layer, _) in staged {
                let ticket = try ring.stage(layer: layer)
                ring.encodeWait(ticket, commandBuffer: cb)
                ring.encodeRelease(ticket, commandBuffer: cb)
            }
            cb.commit()
            cb.waitUntilCompleted()
            ring.drain()
            #expect(ring.error == nil && cb.status == .completed)
            Self.report("\(mode.rawValue) pass \(pass + 1)", ring.timeline())
        }
        Self.log("in the file cache after: "
                 + staged.map { String(format: "%.0f%%", Self.cachedFraction($0.file) * 100) }.joined(separator: " "))
        Self.state("after")
    }

    /// The runner on the install with the staging timeline on, in `mode`.
    static func runner(_ context: MetalContext, mode: Inkling2StagedReadMode)
        throws -> (Model, Inkling2ForwardRunner, Inkling2ExpertStore) {
        let model = try Model.load(directoryURL: try #require(install), device: context.device, expecting: .inkling2Mini,
                                   integrityPolicy: .sizeCheckTrustedReceipt)
        let weights = try Inkling2RunnerWeights(model: model)
        let store = try #require(weights.experts as? Inkling2ExpertStore)
        store.readMode = mode
        store.staging?.recordsTimeline = true
        let runner = try Inkling2ForwardRunner(weights: weights, context: context, maxContext: 4096, maxChunkTokens: 2048)
        return (model, runner, store)
    }

    /// One prefill from position 0; returns wall seconds and the timeline.
    static func prefill(_ runner: Inkling2ForwardRunner, _ store: Inkling2ExpertStore, _ tokens: [Int32], out: MTLBuffer)
        async throws -> (Double, [Inkling2StagedExpertLayers.TimelineEntry]) {
        runner.reset()
        store.staging?.resetTimeline()
        let start = DispatchTime.now().uptimeNanoseconds
        _ = try await runner.prefillChunked(tokens: tokens[...], startPosition: 0, outputMode: .logits,
                                            config: .production(chunkTokens: 2048), into: out, onProgress: { _ in })
        let wall = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
        store.drainStaging()
        #expect(store.stagingError == nil)
        return (wall, store.staging?.timeline() ?? [])
    }

    @Test(.enabled(if: enabled && (stage.hasPrefix("short-") || stage.hasPrefix("chunk2048-")),
                   "needs MFERENCE_INKLING2_GTURBO and a prefill stage"))
    func prefillWall() async throws {
        let mode: Inkling2StagedReadMode = Self.stage.hasSuffix("-cached") ? .cached : .uncached
        let context = try MetalContext()
        Self.state("before load")
        let (model, runner, store) = try Self.runner(context, mode: mode)
        Self.state("loaded")
        let out = try #require(context.device.makeBuffer(length: model.config.vocabSize * 2, options: .storageModeShared))
        // chunk2048: one chunk of 2,048, then a 4K prompt (two chunks).
        let sizes = Self.stage.hasPrefix("chunk2048-") ? [2048, 4096] : [16, 128, 512]
        let tokens = try await Self.promptTokens(model, count: sizes.max()!)
        // Warm-up (pipelines; in cached mode it also fills the file cache).
        let (warm, warmTimeline) = try await Self.prefill(runner, store, Array(tokens.prefix(16)), out: out)
        Self.log(String(format: "warm-up prefill of 16 tokens: %.3f s", warm))
        Self.report("warm-up", warmTimeline)
        for n in sizes {
            let (wall, timeline) = try await Self.prefill(runner, store, Array(tokens.prefix(n)), out: out)
            Self.log(String(format: "%@ prefill of %d tokens: wall %.3f s (%.0f tok/s)", mode.rawValue, n, wall,
                            Double(n) / wall))
            Self.report("\(n) tokens", timeline)
            Self.state("after \(n) tokens")
        }
        if let staged = try? Self.stagedFiles(model) {
            Self.log("in the file cache at the end: "
                     + staged.map { String(format: "%.0f%%", Self.cachedFraction($0.file) * 100) }.joined(separator: " "))
        }
    }

    @Test(.enabled(if: enabled && stage == "long-decode", "needs MFERENCE_INKLING2_GTURBO and the long-decode stage"))
    func longDecodeThenPrefill() async throws {
        let context = try MetalContext()
        let (model, runner, store) = try Self.runner(context, mode: .forHost())
        let c = model.config
        let staged = try Self.stagedFiles(model)
        Self.log("store: read mode \(store.readMode.rawValue), decode slots \(store.decodeSlotLayout), decode misses "
                 + "\(store.decodeMissReadMode?.rawValue ?? "same"), slabs \(store.staging?.slabs.count ?? 0) x "
                 + String(format: "%.2f GB, bound %.2f GB", Double(store.staging?.slabBytes ?? 0) / 1e9,
                          Double(store.boundBytes) / 1e9))
        let out = try #require(context.device.makeBuffer(length: c.vocabSize * 2, options: .storageModeShared))
        let text = try await Self.promptTokens(model, count: 64 + 512)
        func cachedMB() -> Double {
            staged.reduce(0.0) { $0 + max(0, Self.cachedFraction($1.file)) * Double($1.file.bytes) / 1e6 }
        }
        func status(_ label: String) {
            let m = Self.hostMemoryGB()
            Self.log(String(format: "%@: staged files in the file cache %.0f MB; host free %.2f GB, file-backed %.2f GB, compressed %.2f GB; footprint %.2f GB; swap %.2f MB",
                            label, cachedMB(), m.free, m.fileBacked, m.compressed, Self.footprintGB(), Self.swapMB()))
        }
        func argmax() -> Int32 {
            let p = out.contents().bindMemory(to: Float16.self, capacity: c.unpaddedVocabSize)
            var best = 0
            for i in 1..<c.unpaddedVocabSize where p[i] > p[best] { best = i }
            return Int32(best)
        }
        status("loaded")
        _ = try await runner.prefillChunked(tokens: text[0..<64], startPosition: 0, outputMode: .logits,
                                            config: .production(chunkTokens: 2048), into: out, onProgress: { _ in })
        status("after the 64-token prompt")
        var position = 64, token = argmax()
        // Times `produce` alone (not the argmax or the status lines).
        var windowSeconds = 0.0, windowStats = runner.decodeCacheStats
        for step in 1...1024 {
            let t0 = DispatchTime.now().uptimeNanoseconds
            try await runner.produce(token: token, position: position, into: out)
            windowSeconds += Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e9
            token = argmax()
            position += 1
            if step % 128 == 0 {
                let stats = runner.decodeCacheStats
                let hits = stats.hits - windowStats.hits, misses = stats.misses - windowStats.misses
                Self.log(String(format: "tokens %d-%d: %.1f ms/token, hit rate %.1f%%, %.1f misses and %.1f ms of slot loads per token",
                                step - 127, step, windowSeconds * 1000 / 128,
                                100 * Double(hits) / Double(max(1, hits + misses)), Double(misses) / 128,
                                (stats.seconds - windowStats.seconds) * 1000 / 128))
                status("after \(step) tokens")
                windowSeconds = 0
                windowStats = stats
            }
        }
        // A follow-up turn: 512 tokens prefilled after the generation. The
        // staged files' cached pages immediately before and after it show
        // whether the uncached sweep throws the decode misses' pages away.
        status("immediately before the 512-token prefill")
        let started = DispatchTime.now().uptimeNanoseconds
        _ = try await runner.prefillChunked(tokens: text[64..<576], startPosition: position, outputMode: .logits,
                                            config: .production(chunkTokens: 2048), into: out, onProgress: { _ in })
        Self.log(String(format: "512-token prefill after the generation: %.3f s", Double(DispatchTime.now().uptimeNanoseconds - started) / 1e9))
        status("immediately after the 512-token prefill")
        position += 512
        token = argmax()
        var after: [String] = []
        for _ in 0..<16 {
            let before = runner.decodeCacheStats, t0 = DispatchTime.now().uptimeNanoseconds
            try await runner.produce(token: token, position: position, into: out)
            let stats = runner.decodeCacheStats
            after.append(String(format: "%.0f ms/%d misses/%.0f ms loads", Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6,
                                stats.misses - before.misses, (stats.seconds - before.seconds) * 1000))
            token = argmax()
            position += 1
        }
        Self.log("the 16 tokens after the prefill: " + after.joined(separator: ", "))
        status("end")
        #expect(store.stagingError == nil)
    }

    /// Everything through the file cache (`all-cached`): prefill slab fills
    /// and decode misses cached. A warm-up prefill fills the cache with the
    /// staged layers; then prefill wall time for 16, 128 and 512 tokens, and
    /// 128 decode tokens after the 512 with their slot-load cost, with the
    /// staged files' cached pages, the host's memory and swap around them.
    /// The number of other test jobs running is printed, since the question
    /// is what that cache does under ordinary load.
    @Test(.enabled(if: enabled && stage == "all-cached", "needs MFERENCE_INKLING2_GTURBO and the all-cached stage"))
    func allCached() async throws {
        let jobs = (try? Self.countProcesses(matching: "swiftpm-testing-helper")) ?? -1
        Self.log("other test helpers running at the start: \(jobs)")
        let context = try MetalContext()
        let (model, runner, store) = try Self.runner(context, mode: .cached)
        store.decodeMissReadMode = .cached
        let c = model.config
        let staged = try Self.stagedFiles(model)
        func status(_ label: String) {
            let cached = staged.reduce(0.0) { $0 + max(0, Self.cachedFraction($1.file)) * Double($1.file.bytes) / 1e6 }
            let m = Self.hostMemoryGB()
            Self.log(String(format: "%@: staged files in the file cache %.0f MB; host free %.2f GB, file-backed %.2f GB, compressed %.2f GB; footprint %.2f GB; swap %.2f MB",
                            label, cached, m.free, m.fileBacked, m.compressed, Self.footprintGB(), Self.swapMB()))
        }
        let out = try #require(context.device.makeBuffer(length: c.vocabSize * 2, options: .storageModeShared))
        let tokens = try await Self.promptTokens(model, count: 512)
        status("loaded")
        let (warm, _) = try await Self.prefill(runner, store, Array(tokens.prefix(16)), out: out)
        Self.log(String(format: "warm-up prefill of 16 tokens (fills the cache): %.3f s", warm))
        status("after the warm-up")
        for n in [16, 128, 512] {
            let (wall, timeline) = try await Self.prefill(runner, store, Array(tokens.prefix(n)), out: out)
            Self.log(String(format: "cached prefill of %d tokens: wall %.3f s (%.0f tok/s)", n, wall, Double(n) / wall))
            Self.report("\(n) tokens", timeline)
        }
        status("after the prefills")
        // Decode after the 512-token prefill (the runner's position is 512).
        var token: Int32 = 0
        do {
            let p = out.contents().bindMemory(to: Float16.self, capacity: c.unpaddedVocabSize)
            var best = 0
            for i in 1..<c.unpaddedVocabSize where p[i] > p[best] { best = i }
            token = Int32(best)
        }
        var seconds = 0.0
        let before = runner.decodeCacheStats
        for step in 0..<128 {
            let t0 = DispatchTime.now().uptimeNanoseconds
            try await runner.produce(token: token, position: 512 + step, into: out)
            seconds += Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e9
            let p = out.contents().bindMemory(to: Float16.self, capacity: c.unpaddedVocabSize)
            var best = 0
            for i in 1..<c.unpaddedVocabSize where p[i] > p[best] { best = i }
            token = Int32(best)
        }
        let stats = runner.decodeCacheStats
        let hits = stats.hits - before.hits, misses = stats.misses - before.misses
        Self.log(String(format: "128 decode tokens, all cached: %.1f ms/token, hit rate %.1f%%, %.1f misses and %.1f ms of slot loads per token (%.3f ms per miss)",
                        seconds * 1000 / 128, 100 * Double(hits) / Double(max(1, hits + misses)), Double(misses) / 128,
                        (stats.seconds - before.seconds) * 1000 / 128,
                        misses > 0 ? (stats.seconds - before.seconds) * 1000 / Double(misses) : 0))
        status("end")
        Self.log("other test helpers running at the end: \((try? Self.countProcesses(matching: "swiftpm-testing-helper")) ?? -1)")
        #expect(store.stagingError == nil)
    }

    /// Processes whose command line contains `pattern`, less this one.
    static func countProcesses(matching pattern: String) throws -> Int {
        let pgrep = Process()
        pgrep.executableURL = URL(fileURLWithPath: "/usr/bin/pgrep")
        pgrep.arguments = ["-f", pattern]
        let pipe = Pipe()
        pgrep.standardOutput = pipe
        try pgrep.run()
        pgrep.waitUntilExit()
        let pids = String(decoding: pipe.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
            .split(separator: "\n").compactMap { Int32($0) }
        return pids.filter { $0 != getpid() }.count
    }

    @Test(.enabled(if: enabled && stage == "short-vs-sweep", "needs MFERENCE_INKLING2_GTURBO and the short-vs-sweep stage"))
    func shortChunksMatchTheSweep() async throws {
        let context = try MetalContext()
        let (model, runner, _) = try Self.runner(context, mode: .forHost())
        let c = model.config
        let tokens = try await Self.promptTokens(model, count: 64)
        let out = try #require(context.device.makeBuffer(length: c.vocabSize * 2, options: .storageModeShared))
        func argmax() -> Int32 {
            let p = out.contents().bindMemory(to: Float16.self, capacity: c.unpaddedVocabSize)
            var best = 0
            for i in 1..<c.unpaddedVocabSize where p[i] > p[best] { best = i }
            return Int32(best)
        }
        /// The prompt's next-token logits (raw FP16 bytes), then 32 greedy tokens.
        func generate(_ prompt: [Int32], shortMax: Int) async throws -> (logits: [UInt16], tokens: [Int32], seconds: Double) {
            runner.shortPrefillMaxTokens = shortMax
            runner.reset()
            let started = DispatchTime.now().uptimeNanoseconds
            _ = try await runner.prefillChunked(tokens: prompt[...], startPosition: 0, outputMode: .logits,
                                                config: .production(chunkTokens: 2048), into: out, onProgress: { _ in })
            let seconds = Double(DispatchTime.now().uptimeNanoseconds - started) / 1e9
            let logits = Array(UnsafeBufferPointer(start: out.contents().bindMemory(to: UInt16.self, capacity: c.vocabSize),
                                                   count: c.vocabSize))
            var generated: [Int32] = [], token = argmax(), position = prompt.count
            for _ in 0..<32 {
                generated.append(token)
                try await runner.produce(token: token, position: position, into: out)
                token = argmax()
                position += 1
            }
            return (logits, generated, seconds)
        }
        for n in [16, 64] {
            let prompt = Array(tokens.prefix(n))
            let before = runner.shortPrefillStats
            let short = try await generate(prompt, shortMax: 64)
            let stats = runner.shortPrefillStats
            let sweep = try await generate(prompt, shortMax: 0)
            let again = try await generate(prompt, shortMax: 64)      // warm: the slots hold the last turn's experts
            Self.log(String(format: "%d-token prompt: prefill through the slots %.3f s cold, %.3f s warm; whole-layer reads %.3f s; "
                            + "staged visits %d (%d read whole), %d hits, %d misses; logits identical: %@, text identical: %@",
                            n, short.seconds, again.seconds, sweep.seconds, stats.visits - before.visits,
                            stats.fallbacks - before.fallbacks, stats.hits - before.hits, stats.misses - before.misses,
                            short.logits == sweep.logits ? "yes" : "NO", short.tokens == sweep.tokens ? "yes" : "NO"))
            #expect(short.logits == sweep.logits, "\(n)-token prompt: next-token logits")
            #expect(short.tokens == sweep.tokens, "\(n)-token prompt: generated tokens")
            #expect(again.logits == sweep.logits)
        }
        Self.state("end")
    }
}

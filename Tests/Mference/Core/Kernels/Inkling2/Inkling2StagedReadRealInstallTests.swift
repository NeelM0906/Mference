import Darwin
import Foundation
import Metal
import Testing
@testable import Mference
import MferenceValidationSupport

/// Staged-read measurements on the real Inkling 2 Mini install (K2, for the
/// C2 table): the uncached and cached read rates of the staged layers, prefill
/// wall time for short prompts in each mode, how much of a 2,048-token chunk's
/// staging the GPU waits for, and the decode slot cache's hit rate and miss
/// cost on real routes. Each stage loads the install, so it runs only with
/// `MFERENCE_INKLING2_GTURBO` set and `MFERENCE_INKLING2_K2_STAGE` naming the
/// stage, and then only inside `model-hold.sh` (one stage per hold):
///
/// * `sweep-uncached`, `sweep-cached`: the staging ring alone over the plan's
///   staged layers, in visit order (cached: two passes, the first fills the
///   file cache);
/// * `short-uncached`, `short-cached`: the runner's prefill of 16, 128 and
///   512 tokens (cached: after a warm-up prefill fills the file cache);
/// * `chunk2048-uncached`: one 2,048-token chunk with the staging timeline;
/// * `decode64-uncached`: 64 greedy tokens through the runner with the routes
///   traced, then the staged visits replayed through `prepareDecodeVisit`.
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
        let model = try Model.load(directoryURL: try #require(Self.install), device: context.device, expecting: .inkling2Mini)
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
        let model = try Model.load(directoryURL: try #require(install), device: context.device, expecting: .inkling2Mini)
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
        let sizes = Self.stage.hasPrefix("chunk2048-") ? [2048] : [16, 128, 512]
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

    @Test(.enabled(if: enabled && stage == "decode64-uncached", "needs MFERENCE_INKLING2_GTURBO and the decode stage"))
    func decodeReplay() async throws {
        let context = try MetalContext()
        let (model, runner, store) = try Self.runner(context, mode: .uncached)
        let c = model.config
        let prompt = try await Self.promptTokens(model, count: 64)
        let generate = 64, rows = prompt.count + generate, topK = c.topKExperts
        let out = try #require(context.device.makeBuffer(length: c.vocabSize * 2, options: .storageModeShared))
        let traceOut = try #require(context.device.makeBuffer(length: 48 * rows * c.hiddenSize * 4, options: .storageModeShared))
        let routes = try #require(context.device.makeBuffer(length: 48 * rows * topK * 4, options: .storageModeShared))
        runner.visitTrace = .init(outputs: traceOut, routes: routes, rows: rows)
        _ = try await Self.prefill(runner, store, prompt, out: out)
        // Greedy decode through the runner as it stands (it stages whole
        // layers per decode visit); only the routes are kept.
        var token = Int32(0), position = prompt.count
        func argmax() -> Int32 {
            let p = out.contents().bindMemory(to: Float16.self, capacity: c.unpaddedVocabSize)
            var best = 0
            for i in 1..<c.unpaddedVocabSize where p[i] > p[best] { best = i }
            return Int32(best)
        }
        token = argmax()
        let decodeStart = DispatchTime.now().uptimeNanoseconds
        for _ in 0..<generate {
            try await runner.produce(token: token, position: position, into: out)
            token = argmax()
            position += 1
        }
        store.drainStaging()
        Self.log(String(format: "runner decode of %d tokens (whole-layer staging): %.2f s", generate,
                        Double(DispatchTime.now().uptimeNanoseconds - decodeStart) / 1e9))
        runner.visitTrace = nil

        // Replay the decoded tokens' staged visits through the slot caches,
        // in visit order, with nothing else running.
        let ids = routes.contents().bindMemory(to: UInt32.self, capacity: 48 * rows * topK)
        let schedule = Inkling2ReferenceConfig.mini.schedule
        var hits = 0, misses = 0, seconds = 0.0, missesByToken: [Int] = [], msByToken: [Double] = []
        for t in prompt.count..<rows {
            var tokenMisses = 0, tokenSeconds = 0.0
            for visit in 0..<48 where store.isStaged(layer: schedule[visit].layer) {
                let routed = (0..<topK).map { Int(ids[(visit * rows + t) * topK + $0]) }
                let start = DispatchTime.now().uptimeNanoseconds
                let binding = try store.prepareDecodeVisit(layer: schedule[visit].layer, expertIDs: routed)
                tokenSeconds += Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
                let cb = context.queue.makeCommandBuffer()!
                store.endDecodeVisit(binding, commandBuffer: cb)
                cb.commit()
                await cb.completed()
                hits += binding.hits
                misses += binding.misses
                tokenMisses += binding.misses
            }
            seconds += tokenSeconds
            missesByToken.append(tokenMisses)
            msByToken.append(tokenSeconds * 1000)
        }
        let staged = (0..<48).filter { store.isStaged(layer: schedule[$0].layer) }.count
        Self.log(String(format: "slot-cache replay of %d tokens x %d staged visits: hit rate %.1f%% (%d hits, %d misses); reads %.2f s, %.2f ms per miss, %.1f ms per token",
                        generate, staged, Double(hits) * 100 / Double(hits + misses), hits, misses, seconds,
                        misses > 0 ? seconds * 1000 / Double(misses) : 0, seconds * 1000 / Double(generate)))
        Self.log("misses per token: \(missesByToken)")
        Self.log("read ms per token: \(msByToken.map { String(format: "%.1f", $0) })")
        Self.log("slots per staged layer: \((0..<c.numLayers).filter { store.isStaged(layer: $0) }.map { "L\($0) \(store.decodeSlots(layer: $0))" })")
        Self.state("end")
    }
}

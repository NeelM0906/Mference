import Foundation
import Metal
import Testing
@testable import Mference
import MferenceValidationSupport

/// GPU time of the Inkling 2 routed-expert path at the model's shape (D 2304,
/// F 1152, 128 experts, top-8) on synthetic weights: router, route plan,
/// grouped gate-up and down, reduce, for chunks of 512, 1,024 and 2,048 rows,
/// and the one-token decode step. Env-gated: `MFERENCE_INKLING2_MOE_KBENCH=1`.
/// Prints one line per measurement and asserts only that the work completed.
@Suite(.serialized) struct Inkling2MoEKernelBenchmarks {
    private static var enabled: Bool {
        ProcessInfo.processInfo.environment["MFERENCE_INKLING2_MOE_KBENCH"] == "1"
    }

    private static let shape = Inkling2MoEShape.inkling2Mini
    private static let repeats = 7

    /// A full layer of 128 experts: eight distinct random blobs repeated, so
    /// every expert has its own addresses without minutes of fixture work.
    private static func pool(_ device: MTLDevice, _ encoding: Inkling2MoEEncoding) throws -> Inkling2ExpertPool {
        let blobBytes = shape.expertBytes(encoding)
        let stride = (blobBytes + 16383) / 16384 * 16384
        guard let buffer = device.makeBuffer(length: stride * shape.experts, options: .storageModeShared) else {
            throw PrefillGroupedRoutedMoEError.allocationFailed("bench pool")
        }
        var rng = SeedTree(0xB2).key("bench-\(encoding)")
        let base = buffer.contents().bindMemory(to: UInt8.self, capacity: buffer.length)
        let distinct = (0..<8).map { _ in Inkling2SyntheticPool.randomBlocks(encoding: encoding, bytes: blobBytes, rng: &rng) }
        for e in 0..<shape.experts {
            distinct[e % 8].withUnsafeBufferPointer { (base + e * stride).update(from: $0.baseAddress!, count: blobBytes) }
        }
        let scales: [Float] = (0..<(2 * shape.experts)).map { _ in encoding == .nvfp4 ? rng.uniform(0.006, 0.02) : 1 }
        let scaleBuffer = scales.withUnsafeBytes {
            device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)!
        }
        return try Inkling2ExpertPool(buffer: buffer, encoding: encoding, shape: shape, expertStride: stride,
                                      gateUpScales: scaleBuffer, downScales: scaleBuffer,
                                      downScalesOffset: shape.experts * 4)
    }

    /// Uniform routes, or a skewed load where expert popularity falls as 1/(i+1)^0.7.
    private static func routes(tokens: Int, skewed: Bool, seed: UInt64) -> (ids: [UInt32], weights: [Float]) {
        guard skewed else {
            return Inkling2MoETestSupport.randomRoutes(tokens: tokens, experts: shape.experts, topK: shape.topK, seed: seed)
        }
        var rng = SeedTree(seed).key("skewed")
        let popularity = (0..<shape.experts).map { 1 / pow(Float($0 + 1), 0.7) }
        let total = popularity.reduce(0, +)
        var ids: [UInt32] = [], weights: [Float] = []
        for _ in 0..<tokens {
            var chosen: [UInt32] = []
            while chosen.count < shape.topK {
                var u = rng.uniform(0, total), e = 0
                while e < shape.experts - 1 && u >= popularity[e] { u -= popularity[e]; e += 1 }
                if !chosen.contains(UInt32(e)) { chosen.append(UInt32(e)) }
            }
            ids += chosen
            weights += (0..<shape.topK).map { _ in rng.uniform(0.2, 2) }
        }
        return (ids, weights)
    }

    private static func best(_ context: MetalContext, _ body: (MTLComputeCommandEncoder) -> Void) -> Double {
        var best = Double.infinity
        for _ in 0..<repeats {
            let cb = Inkling2MoETestSupport.run(context, body)
            #expect(cb.status == .completed, "\(String(describing: cb.error))")
            best = min(best, cb.gpuEndTime - cb.gpuStartTime)
        }
        return best
    }

    /// The loop passes' per-visit gate and mix kernels at one token (decode),
    /// Mini shapes (d 2,304, gate rank 64; synthetic F32 weights): GPU time
    /// per dispatch, 32 dispatches per command buffer so the GPU stays
    /// clocked up. Each loop-pass visit runs four message gates (q, k, v and
    /// the MoE input) and two depth mixes; the real install's decode
    /// timeline shows loop-pass visits at about 6 ms against 0.9 ms in pass 0.
    @Test(.enabled(if: enabled))
    func loopPassKernelsAtOneToken() throws {
        let context = try MetalContext()
        let kernels = try Inkling2RunnerKernels(device: context.device)
        let device = context.device
        let d = 2304, rank = 64, steps = 32
        func floats(_ count: Int, seed: UInt64) -> MTLBuffer {
            var rng = SeedTree(seed).key("loop-pass-bench")
            let values = (0..<count).map { _ in Float.random(in: -0.05...0.05, using: &rng) }
            return device.makeBuffer(bytes: values, length: count * 4, options: .storageModeShared)!
        }
        let x = floats(4 * d, seed: 1), message = floats(4 * d, seed: 2), out = floats(4 * d, seed: 3)
        let down = floats(rank * 2 * d, seed: 4), gain = floats(d * rank, seed: 5), forget = floats(d * rank, seed: 6)
        let arena = floats(9 * d, seed: 7), queries = floats(4 * d, seed: 8)
        func perDispatch(_ body: (MTLComputeCommandEncoder) -> Void) -> Double {
            _ = Inkling2MoETestSupport.run(context) { enc in for _ in 0..<steps { body(enc) } }
            return Self.best(context) { enc in for _ in 0..<steps { body(enc) } } / Double(steps) * 1000
        }
        let gate = perDispatch { enc in
            kernels.encodeMessageGate(enc, x: x, xStride: d, message: message, messageStride: d,
                                      down: down, downOffset: 0, gain: gain, gainOffset: 0, forget: forget,
                                      forgetOffset: 0, rank: rank, out: out, outStride: d, d: d, tokens: 1)
        }
        let mix4 = perDispatch { enc in
            kernels.encodeDepthMix(enc, arena: arena, stateOffsets: (0..<8).map { $0 * d }, stateStride: d,
                                   queries: queries, queryCount: 4, out: out, outStride: d, outQueryStride: d,
                                   d: d, tokens: 1)
        }
        let mix1 = perDispatch { enc in
            kernels.encodeDepthMix(enc, arena: arena, stateOffsets: (0..<8).map { $0 * d }, stateStride: d,
                                   queries: queries, queryCount: 1, out: out, outStride: d, outQueryStride: d,
                                   d: d, tokens: 1)
        }
        let gateBytes = Double((rank * 2 * d + 2 * d * rank) * 4)
        print(String(format: "[inkling2-moe-kbench] loop pass at one token: message gate %.3f ms per dispatch (%.1f GB/s of its %.2f MB of weights), four per visit %.2f ms; depth mix over 8 states with 4 queries %.3f ms, with 1 query %.3f ms",
                     gate, gateBytes / (gate / 1000) / 1e9, gateBytes / 1e6, 4 * gate, mix4, mix1))
    }

    @Test(.enabled(if: enabled))
    func groupedExperts() throws {
        let context = try MetalContext()
        let kernels = try Inkling2MoEKernels(context: context)
        let device = context.device
        print("[inkling2-moe-kbench] device=\(device.name) apple10=\(device.supportsApple10TensorOps)")
        let pools = try Inkling2MoEEncoding.allCases.map { try Self.pool(device, $0) }
        let maxTokens = 2048
        let scratch = try Inkling2MoEPrefillScratch(device: device, shape: Self.shape, maxTokens: maxTokens)
        print("[inkling2-moe-kbench] scratch at 2048 rows: \(Inkling2MoEPrefillScratch.bytes(shape: Self.shape, maxTokens: maxTokens) / 1_048_576) MiB")
        let (hidden, _) = Inkling2MoETestSupport.halfBuffer(device, rows: maxTokens, stride: Self.shape.d, d: Self.shape.d,
                                                            seed: 0xB3, range: 1)
        let out = device.makeBuffer(length: maxTokens * Self.shape.d * 4, options: .storageModeShared)!
        // Router inputs: FP32 rows and a random F32 router.
        var rng = SeedTree(0xB4).key("router")
        let x = (0..<(maxTokens * Self.shape.d)).map { _ in rng.uniform(-1, 1) }
        let routerWeight = (0..<(128 * Self.shape.d)).map { _ in rng.uniform(-0.05, 0.05) }
        let bias = (0..<128).map { _ in rng.uniform(-0.1, 0.1) }
        func buffer(_ v: [Float]) -> MTLBuffer {
            v.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)! }
        }
        let xBuffer = buffer(x), wBuffer = buffer(routerWeight), bBuffer = buffer(bias)
        let routerIds = device.makeBuffer(length: maxTokens * 32, options: .storageModeShared)!
        let routerWeights = device.makeBuffer(length: maxTokens * 32, options: .storageModeShared)!

        // Warm-up: bring the GPU to its sustained clock before the first number.
        let warm = Self.routes(tokens: maxTokens, skewed: false, seed: 0xB8)
        Inkling2MoETestSupport.write(warm.ids, to: scratch.ids)
        Inkling2MoETestSupport.write(warm.weights, to: scratch.routeWeights)
        let warmStart = Date()
        while Date().timeIntervalSince(warmStart) < 3 {
            _ = Inkling2MoETestSupport.run(context) {
                kernels.encodePrefill($0, hidden: hidden, hiddenStride: Self.shape.d, pool: pools[0],
                                      scratch: scratch, tokens: maxTokens, out: out, outStride: Self.shape.d)
            }
        }

        for tokens in [512, 1024, 2048] {
            let router = Self.best(context) {
                kernels.encodeRouter($0, shape: Self.shape, x: xBuffer, xStride: Self.shape.d, weight: wBuffer,
                                     bias: bBuffer, ids: routerIds, routeWeights: routerWeights, tokens: tokens)
            }
            print(String(format: "[inkling2-moe-kbench] rows=%d router %.3f ms", tokens, router * 1000))
            for skewed in [false, true] {
                let r = Self.routes(tokens: tokens, skewed: skewed, seed: 0xB5 + UInt64(tokens))
                Inkling2MoETestSupport.write(r.ids, to: scratch.ids)
                Inkling2MoETestSupport.write(r.weights, to: scratch.routeWeights)
                for pool in pools {
                    let plan = Self.best(context) { kernels.encodeRoutePlan($0, scratch: scratch, tokens: tokens) }
                    let experts = Self.best(context) {
                        kernels.encodeGroupedExperts($0, hidden: hidden, hiddenStride: Self.shape.d, pool: pool,
                                                     scratch: scratch, tokens: tokens)
                    }
                    let reduce = Self.best(context) {
                        kernels.encodeReduce($0, scratch: scratch, tokens: tokens, out: out, outStride: Self.shape.d)
                    }
                    let total = Self.best(context) {
                        kernels.encodePrefill($0, hidden: hidden, hiddenStride: Self.shape.d, pool: pool,
                                              scratch: scratch, tokens: tokens, out: out, outStride: Self.shape.d)
                    }
                    let blockCounts = Inkling2MoETestSupport.read(scratch.workCount, as: UInt32.self, count: 2)
                    let work = Int(blockCounts[0]), tails = Int(blockCounts[1])
                    let flops = 2.0 * Double(tokens * Self.shape.topK) * Double(Self.shape.macsPerPair)
                    print(String(format: "[inkling2-moe-kbench] rows=%d routes=%@ enc=%@ blocks=%d+%d tail plan %.3f ms experts %.3f ms (%.2f TFLOP/s) reduce %.3f ms total %.3f ms (%.2f TFLOP/s)",
                                 tokens, skewed ? "skewed" : "uniform", pool.encoding == .nvfp4 ? "nvfp4" : "q8_0", work, tails,
                                 plan * 1000, experts * 1000, flops / experts / 1e12, reduce * 1000,
                                 total * 1000, flops / total / 1e12))
                    if !skewed, kernels.availableVariants.contains(.tensorOps) {
                        kernels.variant = .tensorOps
                        let tensor = Self.best(context) {
                            kernels.encodePrefill($0, hidden: hidden, hiddenStride: Self.shape.d, pool: pool,
                                                  scratch: scratch, tokens: tokens, out: out, outStride: Self.shape.d)
                        }
                        kernels.variant = .simdgroup
                        print(String(format: "[inkling2-moe-kbench] rows=%d enc=%@ tensor-ops total %.3f ms (%.2f TFLOP/s)",
                                     tokens, pool.encoding == .nvfp4 ? "nvfp4" : "q8_0", tensor * 1000, flops / tensor / 1e12))
                    }
                    if !skewed {
                        for mode in [Inkling2MoEInputQuantization.q8_0Blocks, .vendorRequired] {
                            let quantized = Self.best(context) {
                                kernels.encodePrefill($0, hidden: hidden, hiddenStride: Self.shape.d, pool: pool,
                                                      scratch: scratch, tokens: tokens, out: out, outStride: Self.shape.d,
                                                      inputQuantization: mode)
                            }
                            print(String(format: "[inkling2-moe-kbench] rows=%d enc=%@ input %@ total %.3f ms (+%.3f ms)",
                                         tokens, pool.encoding == .nvfp4 ? "nvfp4" : "q8_0",
                                         mode == .q8_0Blocks ? "q8_0" : "vendor", quantized * 1000,
                                         (quantized - total) * 1000))
                        }
                    }
                }
            }
        }
    }

    /// The one-token decode step under sustained load: 32 steps per command
    /// buffer, each with its own 8 random experts of a full 128-expert layer,
    /// as a decode loop keeps the GPU busy (a lone 0.3 ms command buffer runs
    /// at idle clocks). Also the gate-up half on its own.
    @Test(.enabled(if: enabled))
    func decodeStep() throws {
        let context = try MetalContext()
        let kernels = try Inkling2MoEKernels(context: context)
        let device = context.device
        let steps = 32
        let (x, _) = Inkling2MoETestSupport.halfBuffer(device, rows: 1, stride: Self.shape.d, d: Self.shape.d, seed: 0xB6, range: 1)
        let ids = device.makeBuffer(length: steps * 32, options: .storageModeShared)!
        let weights = device.makeBuffer(length: steps * 32, options: .storageModeShared)!
        let r = Inkling2MoETestSupport.randomRoutes(tokens: steps, experts: Self.shape.experts, topK: 8, seed: 0xB7)
        Inkling2MoETestSupport.write(r.ids, to: ids)
        Inkling2MoETestSupport.write(r.weights, to: weights)
        let activation = device.makeBuffer(length: 8 * Self.shape.f * 2, options: .storageModeShared)!
        let out = device.makeBuffer(length: Self.shape.d * 4, options: .storageModeShared)!
        for encoding in Inkling2MoEEncoding.allCases {
            let pool = try Self.pool(device, encoding)
            func batch(_ gateUpOnly: Bool) -> (MTLComputeCommandEncoder) -> Void {
                return { encoder in
                    for step in 0..<steps {
                        if gateUpOnly {
                            kernels.encodeDecodeGateUp(encoder, shape: Self.shape, x: x, ids: ids, idsOffset: step * 32,
                                                       pool: pool, activation: activation)
                        } else {
                            kernels.encodeDecode(encoder, shape: Self.shape, x: x, ids: ids, idsOffset: step * 32,
                                                 routeWeights: weights, routeWeightsOffset: step * 32, pool: pool,
                                                 activation: activation, out: out)
                        }
                    }
                }
            }
            let warmStart = Date()
            while Date().timeIntervalSince(warmStart) < 2 { _ = Inkling2MoETestSupport.run(context, batch(false)) }
            let t = Self.best(context, batch(false)) / Double(steps)
            let gateUp = Self.best(context, batch(true)) / Double(steps)
            let bytes = Double(8 * Self.shape.expertBytes(encoding))
            print(String(format: "[inkling2-moe-kbench] decode enc=%@ %.3f ms per step (gate-up %.3f, down %.3f; %.0f GB/s of expert weights)",
                         encoding == .nvfp4 ? "nvfp4" : "q8_0", t * 1000, gateUp * 1000, (t - gateUp) * 1000,
                         bytes / t / 1e9))
        }
    }

    /// Staging a full-size layer in each read mode: synthetic layer files of
    /// the install's sizes (NVFP4 574.6 MB, Q8_0 1,084.2 MB), written past
    /// the file cache, read back into a slab four times per mode, uncached
    /// first (from the SSD), then cached (the first read fills the cache);
    /// deleted afterwards. The steady state of a 32 GB prefill re-reads 12
    /// such layers per chunk.
    @Test(.enabled(if: enabled))
    func stagingThroughput() throws {
        let context = try MetalContext()
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("inkling2-staging-bench-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        var files: [Int: Inkling2ExpertLayerFile] = [:]
        var rng = SeedTree(0xBA).key("staging")
        let chunk = Inkling2SyntheticPool.randomBlocks(encoding: .q8_0, bytes: 34 * 987_000, rng: &rng)
        // Written through an `F_NOCACHE` descriptor from a page-aligned
        // buffer, so the write leaves the pages out of the file cache and the
        // uncached reads below come from the SSD.
        let block = UnsafeMutableRawPointer.allocate(byteCount: 32 << 20, alignment: 16_384)
        defer { block.deallocate() }
        chunk.withUnsafeBytes { block.copyMemory(from: $0.baseAddress!, byteCount: 32 << 20) }
        for (layer, bytes) in [(0, 128 * 4_489_216), (1, 128 * 8_470_528)] {
            let url = directory.appendingPathComponent("layer_\(layer).bin")
            let fd = open(url.path, O_CREAT | O_WRONLY | O_TRUNC, 0o600)
            #expect(fd >= 0 && fcntl(fd, F_NOCACHE, 1) != -1)
            var written = 0
            while written < bytes {
                let n = write(fd, block, min(32 << 20, bytes - written))
                guard n > 0 else { break }
                written += n
            }
            close(fd)
            #expect(written == bytes)
            files[layer] = Inkling2ExpertLayerFile(path: url.path, offset: 0, bytes: bytes)
        }
        // Each read mode in turn: uncached (`F_NOCACHE`, the default below
        // 48 GB) first, then through the file cache.
        for mode in [Inkling2StagedReadMode.uncached, .cached] {
        let staged = try Inkling2StagedExpertLayers(device: context.device, files: files, slabs: 2, readMode: mode)
        for layer in [0, 1] {
            // The first read warms the cache in cached mode; four reads each.
            for _ in 0..<4 {
                let cb = context.queue.makeCommandBuffer()!
                let ticket = try staged.stage(layer: layer)
                staged.encodeWait(ticket, commandBuffer: cb)
                staged.encodeRelease(ticket, commandBuffer: cb)
                cb.commit()
                cb.waitUntilCompleted()
            }
        }
        staged.drain()
        #expect(staged.error == nil)
        print(String(format: "[inkling2-moe-kbench] staging (%@) %.2f GB in %.3f s (%.1f GB/s, first reads included)",
                     mode.rawValue, Double(staged.bytesRead) / 1e9, staged.secondsReading,
                     Double(staged.bytesRead) / 1e9 / staged.secondsReading))

        // Decode-cache misses at full size: both files as staged layers of a
        // store (each cache then holds the whole layer, so every visit of
        // fresh experts is 8 misses).
        let specs = [
            Inkling2ExpertLayerSpec(file: files[0]!, encoding: .nvfp4, expertStride: 4_489_216, gateUpScales: nil, downScales: nil),
            Inkling2ExpertLayerSpec(file: files[1]!, encoding: .q8_0, expertStride: 8_470_528, gateUpScales: nil, downScales: nil),
        ]
        let plan = Inkling2ExpertResidencyPlan(placements: [.staged, .staged], stagingSlabs: 2, residentBytes: 0, stagingBytes: 0)
        let store = try Inkling2ExpertStore(device: context.device, shape: Self.shape, layers: specs, plan: plan,
                                            readMode: mode)
        // The runner prefills before it decodes, so the slabs are touched;
        // here a first pass touches them, then a staged prefill visit empties
        // the caches, and the second pass is timed.
        for pass in 0..<2 {
        if pass == 1 {
            let cb = context.queue.makeCommandBuffer()!
            _ = try store.beginPrefillVisit(layer: 0, commandBuffer: cb)
            try store.endPrefillVisit(layer: 0, commandBuffer: cb)
            cb.commit()
            cb.waitUntilCompleted()
        }
        for (layer, label) in [(0, "nvfp4"), (1, "q8_0")] {
            var times: [Double] = []
            for visit in 0..<(Self.shape.experts / 8) {
                let ids = (0..<8).map { visit * 8 + $0 }
                let start = DispatchTime.now().uptimeNanoseconds
                let binding = try store.prepareDecodeVisit(layer: layer, expertIDs: ids)
                times.append(Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6)
                #expect(binding.misses == 8)
            }
            times.sort()
            print(String(format: "[inkling2-moe-kbench] decode cache %@ (%@, %@): 8 misses per visit, median %.2f ms, best %.2f ms, worst %.2f ms",
                         label, mode.rawValue, pass == 0 ? "first touch of the slabs" : "slabs touched",
                         times[times.count / 2], times[0], times[times.count - 1]))
        }
        }
        store.drainStaging()
        }
    }

    /// Decode reading a layer file mapped and wrapped whole
    /// (`makeBuffer(bytesNoCopy:)`), warm in the file cache, against the same
    /// bytes in an anonymous buffer. Warm, the two cost the same; but binding
    /// a mapping makes Metal page in and pin the whole file (K4's probe: a
    /// cold 1 GiB file read in full for a 4-byte access), so the store does
    /// not decode staged layers this way (`prepareDecodeVisit` instead). Times one
    /// step per command buffer (GPU time and wall time from commit to
    /// completion, which includes Metal making the buffer resident) and 32
    /// steps per command buffer.
    @Test(.enabled(if: enabled))
    func decodeOnMappedLayer() throws {
        let context = try MetalContext()
        let kernels = try Inkling2MoEKernels(context: context)
        let device = context.device
        let encoding = Inkling2MoEEncoding.nvfp4
        let anonymous = try Self.pool(device, encoding)
        let bytes = anonymous.expertStride * Self.shape.experts
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("inkling2-mapped-\(UUID().uuidString).bin")
        defer { try? FileManager.default.removeItem(at: url) }
        FileManager.default.createFile(atPath: url.path, contents: nil)
        let handle = try FileHandle(forWritingTo: url)
        let chunk = 32 << 20
        var written = 0
        while written < bytes {
            let n = min(chunk, bytes - written)
            try handle.write(contentsOf: Data(bytesNoCopy: anonymous.buffer.contents() + written, count: n, deallocator: .none))
            written += n
        }
        try handle.close()
        let fd = open(url.path, O_RDONLY)
        #expect(fd >= 0)
        defer { close(fd) }
        guard let base = mmap(nil, bytes, PROT_READ, MAP_SHARED, fd, 0), base != MAP_FAILED else {
            Issue.record("mmap failed"); return
        }
        defer { munmap(base, bytes) }
        nonisolated(unsafe) let mapped = base
        guard let buffer = device.makeBuffer(bytesNoCopy: mapped, length: bytes, options: .storageModeShared,
                                             deallocator: nil) else { Issue.record("wrap failed"); return }
        let mappedPool = try anonymous.rebound(to: buffer)

        let steps = 32
        let (x, _) = Inkling2MoETestSupport.halfBuffer(device, rows: 1, stride: Self.shape.d, d: Self.shape.d, seed: 0xBB, range: 1)
        let ids = device.makeBuffer(length: steps * 32, options: .storageModeShared)!
        let weights = device.makeBuffer(length: steps * 32, options: .storageModeShared)!
        let r = Inkling2MoETestSupport.randomRoutes(tokens: steps, experts: Self.shape.experts, topK: 8, seed: 0xBC)
        Inkling2MoETestSupport.write(r.ids, to: ids)
        Inkling2MoETestSupport.write(r.weights, to: weights)
        let activation = device.makeBuffer(length: 8 * Self.shape.f * 2, options: .storageModeShared)!
        let out = device.makeBuffer(length: Self.shape.d * 4, options: .storageModeShared)!
        for (label, pool) in [("anonymous", anonymous), ("mapped", mappedPool)] {
            func step(_ s: Int) -> (MTLComputeCommandEncoder) -> Void {
                { kernels.encodeDecode($0, shape: Self.shape, x: x, ids: ids, idsOffset: s * 32, routeWeights: weights,
                                       routeWeightsOffset: s * 32, pool: pool, activation: activation, out: out) }
            }
            let warmStart = Date()
            while Date().timeIntervalSince(warmStart) < 2 { _ = Inkling2MoETestSupport.run(context, step(0)) }
            var gpu = Double.infinity, wall = Double.infinity
            for s in 0..<steps {
                let begin = DispatchTime.now().uptimeNanoseconds
                let cb = Inkling2MoETestSupport.run(context, step(s))
                wall = min(wall, Double(DispatchTime.now().uptimeNanoseconds - begin) / 1e9)
                gpu = min(gpu, cb.gpuEndTime - cb.gpuStartTime)
            }
            let batched = Self.best(context) { encoder in for s in 0..<steps { step(s)(encoder) } } / Double(steps)
            print(String(format: "[inkling2-moe-kbench] decode on %@ layer: one step per buffer GPU %.3f ms, wall %.3f ms; batched %.3f ms per step",
                         label, gpu * 1000, wall * 1000, batched * 1000))
        }
    }

    /// A/B of the pipelined and plain simdgroup loops, alternating in one
    /// process so machine load affects both alike: one 2,048-row (and
    /// 1,024-row) visit per encoding, five rounds of best-of-3 each.
    @Test(.enabled(if: enabled))
    func pipelineAB() throws {
        let context = try MetalContext()
        let pipelined = try Inkling2MoEKernels(context: context, pipelined: true)
        let plain = try Inkling2MoEKernels(context: context, pipelined: false)
        let device = context.device
        let pools = try Inkling2MoEEncoding.allCases.map { try Self.pool(device, $0) }
        let scratch = try Inkling2MoEPrefillScratch(device: device, shape: Self.shape, maxTokens: 2048)
        let (hidden, _) = Inkling2MoETestSupport.halfBuffer(device, rows: 2048, stride: Self.shape.d, d: Self.shape.d,
                                                            seed: 0xBD, range: 1)
        let out = device.makeBuffer(length: 2048 * Self.shape.d * 4, options: .storageModeShared)!
        for tokens in [1024, 2048] {
            let r = Inkling2MoETestSupport.randomRoutes(tokens: tokens, experts: Self.shape.experts, topK: 8, seed: 0xBE)
            Inkling2MoETestSupport.write(r.ids, to: scratch.ids)
            Inkling2MoETestSupport.write(r.weights, to: scratch.routeWeights)
            for pool in pools {
                var a: [Double] = [], b: [Double] = []
                for _ in 0..<5 {
                    for (kernels, sink) in [(pipelined, 0), (plain, 1)] {
                        var best = Double.infinity
                        for _ in 0..<3 {
                            let cb = Inkling2MoETestSupport.run(context) {
                                kernels.encodePrefill($0, hidden: hidden, hiddenStride: Self.shape.d, pool: pool,
                                                      scratch: scratch, tokens: tokens, out: out, outStride: Self.shape.d)
                            }
                            best = min(best, cb.gpuEndTime - cb.gpuStartTime)
                        }
                        if sink == 0 { a.append(best) } else { b.append(best) }
                    }
                }
                print(String(format: "[inkling2-moe-kbench] A/B rows=%d enc=%@ pipelined %.3f ms (min of rounds) plain %.3f ms",
                             tokens, pool.encoding == .nvfp4 ? "nvfp4" : "q8_0", a.min()! * 1000, b.min()! * 1000))
            }
        }
    }
}

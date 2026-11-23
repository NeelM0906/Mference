import Foundation
import Metal
import Testing
@testable import Mference
import MferenceValidationSupport

/// The runner on the real Inkling 2 Mini install (G6b): every visit against
/// the CPU reference reading the same mapped bytes, teacher-forced as in the
/// toy tests, on K0's first golden prompt; then the next greedy token against
/// K0's Metal golden. Loads the 17.6 GB install, so it runs only with
/// `MFERENCE_INKLING2_GTURBO` set, and then only inside `model-hold.sh`.
@Suite("Inkling2ForwardRunner on the real install", .serialized)
struct Inkling2RealInstallTests {
    static var install: URL? {
        ProcessInfo.processInfo.environment["MFERENCE_INKLING2_GTURBO"].map { URL(fileURLWithPath: $0) }
    }

    /// `g01-capital` (`goldens/generate/metal`): "The capital of France is",
    /// then " Paris" from llama.cpp b12641 on Metal.
    static let prompt: [Int32] = [371, 151451, 149879, 151194, 149884]
    static let goldenNext = 152717

    private func relative(_ got: [Float], _ want: [Float]) -> Float {
        zip(got, want).map { abs($0 - $1) }.max()! / max(want.map(abs).max()!, 1e-6)
    }

    @Test(.enabled(if: install != nil, "needs MFERENCE_INKLING2_GTURBO"))
    func visitsMatchTheReferenceOnTheRealInstall() async throws {
        let url = try #require(Self.install)
        let context = try MetalContext()
        // `.pread`: the Model never opens an expert file; the runner's
        // expert store and the reference map them themselves.
        let model = try Model.load(directoryURL: url, device: context.device, expecting: .inkling2Mini)
        let c = model.config
        let d = c.hiddenSize, topK = c.topKExperts, rows = 8
        var weights = try Inkling2RunnerWeights(model: model)
        // Compared with the reference's capped logits; generation leaves the
        // cap to the sampler.
        weights.finalLogitSoftcap = Float(c.finalLogitSoftcap)
        let runner = try Inkling2ForwardRunner(weights: weights, context: context, maxContext: 64, maxChunkTokens: 8)
        // The reference reads the experts where the runner's store holds
        // them: the resident layers' buffers (no second copy) and, for the
        // staged layers, the store's read-only file mappings, which only the
        // CPU touches here (the runner itself stages those layers).
        let store = try #require(weights.experts as? Inkling2ExpertStore)
        let referenceExperts: (Int) throws -> Inkling2ReferenceExpertLayer = { layer in
            let pool = store.decodePool(layer: layer)
            let n = pool.shape.experts
            let nvfp4 = pool.encoding == .nvfp4
            return Inkling2ReferenceExpertLayer(
                buffer: pool.buffer, offset: pool.offset, stride: pool.expertStride,
                gateUpOffset: pool.gateUpOffset, downOffset: pool.downOffset,
                gateUpScales: nvfp4 ? Inkling2BufferMatrix.floats(pool.gateUpScales, offset: pool.gateUpScalesOffset,
                                                                  count: n) : nil,
                downScales: nvfp4 ? Inkling2BufferMatrix.floats(pool.downScales, offset: pool.downScalesOffset,
                                                                count: n) : nil)
        }
        // The store's resident copies equal the files (first and last
        // expert of each resident layer, read back with pread).
        for layer in 0..<c.numLayers where store.plan.placements[layer] == .resident {
            let pool = store.decodePool(layer: layer)
            let path = url.appendingPathComponent("packed_experts")
                .appendingPathComponent(model.packedExpertsLayout.layers[layer].file).path
            let fd = open(path, O_RDONLY)
            #expect(fd >= 0, "layer \(layer) file")
            defer { close(fd) }
            for e in [0, c.numExperts - 1] {
                var bytes = [UInt8](repeating: 0, count: pool.expertStride)
                let got = bytes.withUnsafeMutableBytes { pread(fd, $0.baseAddress!, pool.expertStride,
                                                               off_t(e * pool.expertStride)) }
                #expect(got == pool.expertStride)
                let resident = UnsafeRawBufferPointer(start: pool.buffer.contents() + pool.offset + e * pool.expertStride,
                                                      count: pool.expertStride)
                #expect(resident.elementsEqual(bytes), "layer \(layer) expert \(e)")
            }
        }
        let reference = try Inkling2ReferenceModel(
            config: .mini, weights: Inkling2ReferenceWeights(model: model, experts: referenceExperts))
        reference.captureTrace = true
        reference.precision = .metalRunner
        let device = context.device
        let out = try #require(device.makeBuffer(length: c.vocabSize * 2, options: .storageModeShared))
        let outputs = try #require(device.makeBuffer(length: 48 * rows * d * 4, options: .storageModeShared))
        let routes = try #require(device.makeBuffer(length: 48 * rows * topK * 4, options: .storageModeShared))
        runner.visitTrace = .init(outputs: outputs, routes: routes, rows: rows)
        let ids = routes.contents().bindMemory(to: UInt32.self, capacity: 48 * rows * topK)
        let schedule = Inkling2ReferenceConfig.mini.schedule

        // The prompt, then a decode step on the golden's next token.
        var position = 0
        for (index, chunk) in [Self.prompt, [Int32(Self.goldenNext)]].enumerated() {
            let n = chunk.count
            let started = Date()
            if n == 1 {
                try await runner.produce(token: chunk[0], position: position, into: out)
            } else {
                _ = try await runner.prefillChunked(tokens: chunk[...], startPosition: position, outputMode: .logits,
                                                    config: .production(chunkTokens: 32), into: out,
                                                    onProgress: { _ in })
            }
            let gpuSeconds = Date().timeIntervalSince(started)
            reference.forcedRoutes = (0..<48).map { visit in
                (0..<n).map { t in (0..<topK).map { Int(ids[(visit * rows + position + t) * topK + $0]) } }
            }
            let gpuOutputs = (0..<48).map { visit in
                (0..<n).map { t in
                    Inkling2BufferMatrix.floats(outputs, offset: (visit * rows + position + t) * d * 4, count: d)
                }
            }
            reference.forcedVisitOutputs = gpuOutputs
            let want = reference.forward(tokens: chunk, logits: .last)[0]
            let trace = try #require(reference.lastTrace)
            var visitErrors: [Float] = [], shortfall: Float = 0, flips = 0
            for visit in 0..<48 {
                let blockStart = schedule[visit].layer % 6 == 0
                var err: Float = 0
                for t in 0..<n {
                    let input = blockStart ? [Float](repeating: 0, count: d) : gpuOutputs[visit - 1][t]
                    let got = zip(gpuOutputs[visit][t], input).map { $0 - $1 }
                    let ref = zip(trace.visitOutputs[visit][t], input).map { $0 - $1 }
                    err = max(err, relative(got, ref), relative(gpuOutputs[visit][t], trace.visitOutputs[visit][t]))
                    let s = trace.routeShortfalls[visit][t]
                    if s > 0 { flips += 1 }
                    shortfall = max(shortfall, s)
                }
                visitErrors.append(err)
            }
            let p = out.contents().bindMemory(to: Float16.self, capacity: c.unpaddedVocabSize)
            let real = (0..<c.unpaddedVocabSize).map { Float(p[$0]) }
            let argmax = real.indices.max { real[$0] < real[$1] }!
            let wantArgmax = want.indices.max { want[$0] < want[$1] }!
            print("inkling2 real step \(index) (\(n) at \(position), GPU \(String(format: "%.3f", gpuSeconds)) s): "
                  + "visit error \(visitErrors.max()!), logit error \(relative(real, want)), "
                  + "argmax \(argmax) vs reference \(wantArgmax), \(flips) near-tie flips, shortfall \(shortfall)")
            print("inkling2 real per-visit errors: \(visitErrors.map { String(format: "%.4f", $0) })")
            #expect(visitErrors.max()! < 0.02, "step \(index)")
            #expect(relative(real, want) < 0.01, "step \(index)")
            #expect(shortfall <= 0.005, "step \(index)")
            if index == 0 {
                print("inkling2 real next token after the prompt: \(argmax) (K0 Metal golden \(Self.goldenNext))")
                #expect(argmax == Self.goldenNext)
            }
            position += n
        }
    }
}

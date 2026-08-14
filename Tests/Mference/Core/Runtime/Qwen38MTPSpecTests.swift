import Foundation
import Metal
import Testing
@testable import Mference
@testable import MferenceRepackCore

/// Qwen 3.8 MTP speculative decoding against the toy fixture with the
/// draft layer attached by the real `MTPAttachTool`:
///
///  - the attach step produces a loadable install whose `mtp.*` tensors
///    resolve at runtime,
///  - speculative greedy decode is byte-identical to plain decode for
///    organic drafts (near-zero acceptance: the rollback path), injected
///    always-correct drafts (full acceptance: no rollback), injected
///    partially-correct drafts (mixed acceptance), and several draft depths,
///  - the `MFERENCE_MTP=0` kill switch disables the speculator.
///
/// Serialized: two cases mutate process environment before runner creation.
@Suite(.serialized) struct Qwen38MTPSpecTests {
    private static let vocab = 1024

    /// One attached toy install shared by the suite (attach is idempotent
    /// per directory; each call builds a fresh toy).
    private static func makeAttachedDirectory() throws -> URL {
        let dir = try Qwen38ToySynthetic.write()
        let shard = try Qwen38ToySynthetic.writeMTPShard()
        defer { try? FileManager.default.removeItem(at: shard) }
        let result = try MTPAttachTool.run(gturboDirectory: dir.path,
                                           shardPath: shard.path)
        #expect(result.tensorCount == 15)
        return dir
    }

    private func makeRunner(_ dir: URL, maxContext: Int = 96) throws
        -> (MetalContext, Qwen38ForwardRunner) {
        let ctx = try MetalContext()
        let model = try Model.load(directoryURL: dir,
                                   device: ctx.device,
                                   expecting: .qwen38Toy())
        let runner = try Qwen38ForwardRunner(model: model,
                                             context: ctx,
                                             maxContext: maxContext)
        return (ctx, runner)
    }

    private func makeLogits(_ ctx: MetalContext) throws -> MTLBuffer {
        guard let buf = ctx.device.makeBuffer(
            length: Self.vocab * MemoryLayout<Float16>.stride,
            options: .storageModeShared) else {
            throw ModelError.residentBufferWrapFailed
        }
        return buf
    }

    /// Greedy generation mirroring `runRawCompletion`'s fused-greedy loop:
    /// chunked prefill seeds the first token, then one `produce` per token.
    private func generate(_ runner: Qwen38ForwardRunner,
                          logits: MTLBuffer,
                          prompt: [Int32],
                          newTokens: Int) async throws -> [Int32] {
        runner.reset()
        let result = try await runner.prefillChunked(
            tokens: prompt[...],
            startPosition: 0,
            outputMode: .greedyIfAvailable,
            config: .production(chunkTokens: 32),
            into: logits,
            onProgress: { _ in })
        guard case .greedyToken(let seed) = result.seed else {
            Issue.record("expected a greedy prefill seed")
            return []
        }
        var out: [Int32] = [Int32(bitPattern: seed)]
        var position = prompt.count
        while out.count < newTokens {
            try await runner.produce(token: out.last!, position: position, into: logits)
            position += 1
            out.append(Int32(bitPattern: runner.lastGreedyToken))
        }
        return out
    }

    private static func prompt(_ count: Int) -> [Int32] {
        (0..<count).map { Int32(($0 * 37 + 11) % vocab) }
    }

    @Test func attachProducesLoadableInstall() throws {
        let dir = try Self.makeAttachedDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        // The attach must leave a fresh install receipt behind — the Mac app
        // gates "installed" on verified-install.json, so a stale or missing
        // receipt strands an otherwise valid install on the download screen.
        let receiptURL = dir.appendingPathComponent("verified-install.json")
        let receiptData = try Data(contentsOf: receiptURL)
        let receipt = try JSONSerialization.jsonObject(with: receiptData) as? [String: Any]
        let receiptFiles = receipt?["files"] as? [String: [String: Any]] ?? [:]
        let weightsEntry = receiptFiles["model_weights.bin"]
        let weightsSize = try FileManager.default
            .attributesOfItem(atPath: dir.appendingPathComponent("model_weights.bin").path)[.size] as? Int
        #expect(weightsEntry?["size"] as? Int == weightsSize,
                "receipt must record the post-attach weights size")
        let ctx = try MetalContext()
        // Model.load re-verifies the manifest sha256 of the rewritten
        // weights file, so a load success also validates the attach output.
        let model = try Model.load(directoryURL: dir, device: ctx.device,
                                   expecting: .qwen38Toy())
        let fc = try model.resident(name: "mtp.fc.weight")
        #expect(fc.dtype == 0)
        #expect(fc.shape.0 == 64 && fc.shape.1 == 128)
        #expect(fc.scaleLength > 0 && fc.biasLength > 0)
        let norm = try model.resident(name: "mtp.norm.weight")
        #expect(norm.dtype == 1)
        // Zero-centered source values became full form: all near 1.0.
        let values = norm.buffer.contents().advanced(by: Int(norm.offset))
            .bindMemory(to: UInt16.self, capacity: 64)
        for i in 0..<64 {
            let value = Float(bitPattern: UInt32(values[i]) << 16)
            #expect(abs(value - 1.0) < 0.5, "norm[\(i)] = \(value)")
        }
        // Attaching twice must be rejected.
        let shard = try Qwen38ToySynthetic.writeMTPShard()
        defer { try? FileManager.default.removeItem(at: shard) }
        #expect(throws: (any Error).self) {
            _ = try MTPAttachTool.run(gturboDirectory: dir.path, shardPath: shard.path)
        }
    }

    @Test("speculative decode is byte-identical to plain decode",
          arguments: [1, 2, 3])
    func specMatchesPlain_organicDrafts(k: Int) async throws {
        let dir = try Self.makeAttachedDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let (ctxPlain, plainRunner) = try makeRunner(dir)
        plainRunner.mtp = nil
        let (ctxSpec, specRunner) = try makeRunner(dir)
        let mtp = try #require(specRunner.mtp)
        #expect(k <= mtp.draftCapacity)
        mtp.draftCount = k

        let logitsPlain = try makeLogits(ctxPlain)
        let logitsSpec = try makeLogits(ctxSpec)
        for promptLength in [3, 40] {
            let prompt = Self.prompt(promptLength)
            let want = try await generate(plainRunner, logits: logitsPlain,
                                          prompt: prompt, newTokens: 40)
            let got = try await generate(specRunner, logits: logitsSpec,
                                         prompt: prompt, newTokens: 40)
            #expect(want == got, "k=\(k) prompt=\(promptLength)")
        }
        #expect(mtp.stats.rounds > 0)
    }

    @Test("byte identity under forced full and partial acceptance")
    func specMatchesPlain_injectedDrafts() async throws {
        let dir = try Self.makeAttachedDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let (ctxPlain, plainRunner) = try makeRunner(dir)
        plainRunner.mtp = nil
        let logitsPlain = try makeLogits(ctxPlain)
        let prompt = Self.prompt(9)
        let newTokens = 48
        let want = try await generate(plainRunner, logits: logitsPlain,
                                      prompt: prompt, newTokens: newTokens)

        // The injector needs the position of each round inside the expected
        // stream; the produce loop below keeps the box current.
        final class Cursor { var produced = 0 }
        let cursor = Cursor()

        // scenario: given the correct continuation for the round, the drafts
        // to inject (correct prefix length varies -> full/partial/none).
        let scenarios: [(String, (_ correct: [Int32], _ round: Int) -> [Int32])] = [
            ("full", { correct, _ in correct }),
            ("none", { correct, _ in correct.map { ($0 &+ 1) % Int32(Self.vocab) } }),
            ("mixed", { correct, round in
                var drafts = correct
                let keep = round % (drafts.count + 1)
                for i in keep..<drafts.count {
                    drafts[i] = (drafts[i] &+ 1) % Int32(Self.vocab)
                }
                return drafts
            }),
        ]

        for (name, scenario) in scenarios {
            let (ctxSpec, specRunner) = try makeRunner(dir)
            let mtp = try #require(specRunner.mtp)
            mtp.draftOverride = { round, k in
                // The round's bonus token is want[cursor.produced - 1]; its
                // verified continuation is the next k expected tokens.
                let start = cursor.produced
                let correct = (0..<k).map { offset -> Int32 in
                    let index = start + offset
                    return index < want.count ? want[index] : 7
                }
                return scenario(correct, round)
            }
            let logitsSpec = try makeLogits(ctxSpec)

            specRunner.reset()
            let result = try await specRunner.prefillChunked(
                tokens: prompt[...], startPosition: 0,
                outputMode: .greedyIfAvailable,
                config: .production(chunkTokens: 32),
                into: logitsSpec, onProgress: { _ in })
            guard case .greedyToken(let seed) = result.seed else {
                Issue.record("expected a greedy prefill seed")
                return
            }
            var got: [Int32] = [Int32(bitPattern: seed)]
            var position = prompt.count
            while got.count < newTokens {
                cursor.produced = got.count
                try await specRunner.produce(token: got.last!, position: position,
                                             into: logitsSpec)
                position += 1
                got.append(Int32(bitPattern: specRunner.lastGreedyToken))
            }
            #expect(got == want, "scenario \(name)")
            if name == "full" {
                #expect(mtp.stats.rollbacks == 0, "full acceptance must not roll back")
                #expect(mtp.stats.acceptedTokens == mtp.stats.draftedTokens)
            }
            if name == "none" {
                #expect(mtp.stats.acceptedTokens == 0)
                #expect(mtp.stats.rollbacks == mtp.stats.rounds)
            }
        }
    }

    @Test("deeper draft depth via MFERENCE_MTP_K")
    func specMatchesPlain_k4() async throws {
        setenv("MFERENCE_MTP_K", "4", 1)
        defer { unsetenv("MFERENCE_MTP_K") }
        let dir = try Self.makeAttachedDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let (ctxPlain, plainRunner) = try makeRunner(dir)
        plainRunner.mtp = nil
        let (ctxSpec, specRunner) = try makeRunner(dir)
        let mtp = try #require(specRunner.mtp)
        #expect(mtp.draftCapacity == 4)

        let logitsPlain = try makeLogits(ctxPlain)
        let logitsSpec = try makeLogits(ctxSpec)
        let prompt = Self.prompt(12)
        let want = try await generate(plainRunner, logits: logitsPlain,
                                      prompt: prompt, newTokens: 40)
        let got = try await generate(specRunner, logits: logitsSpec,
                                     prompt: prompt, newTokens: 40)
        #expect(want == got)
    }

    @Test("MFERENCE_MTP=0 disables the speculator")
    func killSwitch() async throws {
        setenv("MFERENCE_MTP", "0", 1)
        defer { unsetenv("MFERENCE_MTP") }
        let dir = try Self.makeAttachedDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let (_, runner) = try makeRunner(dir)
        #expect(runner.mtp == nil)
    }

    @Test("plain qwen38 installs without MTP tensors stay unaffected")
    func noMTPTensorsMeansNoSpeculator() async throws {
        let dir = try Qwen38ToySynthetic.write()
        defer { try? FileManager.default.removeItem(at: dir) }
        let (ctx, runner) = try makeRunner(dir)
        #expect(runner.mtp == nil)
        let logits = try makeLogits(ctx)
        let tokens = try await generate(runner, logits: logits,
                                        prompt: Self.prompt(5), newTokens: 8)
        #expect(tokens.count == 8)
    }
}

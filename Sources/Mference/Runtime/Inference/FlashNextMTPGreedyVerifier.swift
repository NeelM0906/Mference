import Foundation
import Metal

/// Internal exact, sequential target verifier for native-MTP qualification.
/// Deliberately NOT a fast batched verifier or a CLI/server generation path.
/// Never evaluate rejected tokens through a numerically different prefill path.
/// This is the reference against which a future accelerated verifier must gate.
final class FlashNextMTPGreedyVerifier {
    struct Result {
        let tokens: [Int32]
        let accepted: Int
        let reachedStop: Bool
    }

    let target: FlashNextForwardRunner
    let primer: FlashNextMTPPrimer
    let logits: MTLBuffer
    private let context: MetalContext
    private let scratch: RawCompletionScratch
    private let draftScratch: RawCompletionScratch
    private let vocab: Int
    private let maxContext: Int
    private var ready = false
    /// Fault-injection seam; no work when absent.
    var didVerifyToken: ((Int) throws -> Void)?

    init(model: Model, context: MetalContext, maxContext: Int,
         policy: FlashNextMTPDraftRunner.ExpertPolicy) throws {
        self.maxContext = maxContext
        self.context = context
        vocab = model.config.vocabSize
        target = try FlashNextForwardRunner(model: model, context: context, maxContext: maxContext)
        primer = try FlashNextMTPPrimer(model: model, context: context, maxContext: maxContext, policy: policy)
        scratch = try RawCompletionScratch(context: context, vocab: vocab,
            logitSoftcap: Float(model.config.finalLogitSoftcap))
        draftScratch = try RawCompletionScratch(context: context, vocab: vocab,
            logitSoftcap: Float(model.config.finalLogitSoftcap))
        logits = scratch.logits
        let consumer = primer
        target.consumeTargetHiddenRows = { try consumer.consume($0) }
    }

    /// Return the target-selected seed followed by native draft proposals.
    /// Neither target nor primer advances. The first element is guaranteed by
    /// the target and MUST NOT be counted as a successful draft prediction.
    /// This is a correctness path, not accelerated generation.
    func propose(maxDraftTokens: Int) throws -> [Int32] {
        try Task.checkCancellation()
        guard ready, maxDraftTokens >= 0, target.continuationPosition < maxContext else {
            throw FlashNextForwardRunnerError.invalidInput("invalid native proposal state or count")
        }
        let count = min(maxDraftTokens, maxContext - target.continuationPosition - 1)
        let seed = try sampleToken()
        guard count > 0 else { return [seed] }
        do {
            let proposals = try primer.proposals(after: seed, count: count, into: draftScratch.logits) { _ in
                try self.sampleToken(using: self.draftScratch)
            }
            return [seed] + proposals
        } catch {
            // The primer restores ordinary proposal failures itself. If it
            // cannot prove paired cursors are clean after recovery, reset all.
            do {
                _ = try primer.checkpoint()
                guard primer.targetPosition == target.continuationPosition,
                      primer.draftPosition == target.continuationPosition - 1 else {
                    throw FlashNextForwardRunnerError.invalidInput("native proposal recovery cursor mismatch")
                }
            } catch {
                reset()
                throw error
            }
            throw error
        }
    }

    func reset() {
        target.reset()
        primer.reset()
        ready = false
    }

    func prefill(_ tokens: [Int32]) async throws {
        guard !tokens.isEmpty, tokens.count < maxContext else {
            throw FlashNextForwardRunnerError.invalidInput("verification prefix must leave generation capacity")
        }
        reset()
        do {
            _ = try await target.prefillChunked(tokens: tokens[...], startPosition: 0,
                outputMode: .logits, config: .production(chunkTokens: 32), into: logits, onProgress: { _ in })
            ready = true
        } catch {
            reset()
            throw error
        }
    }

    /// Proposals begin at the current next-token position. Commit a matching
    /// prefix plus the target's correction (or a bonus if every proposal agrees).
    /// Budget/context/stop tokens can end earlier. All returned tokens have been
    /// consumed by BOTH target and primer; continuation starts after them.
    /// Sampling/stop-string buffering belongs to future generation integration.
    func verify(_ proposals: [Int32], budget: Int, stopTokens: Set<Int32> = []) async throws -> Result {
        guard ready, budget > 0, target.continuationPosition < maxContext,
              proposals.allSatisfy({ $0 >= 0 && Int($0) < vocab }) else {
            throw FlashNextForwardRunnerError.invalidInput("invalid verification state, budget or proposal")
        }
        let targetBase = try target.captureDecodeCheckpoint()
        let primerBase = try primer.checkpoint()
        let words = logits.contents().assumingMemoryBound(to: UInt16.self)
        let saved = Array(UnsafeBufferPointer(start: words, count: vocab))
        var tokens: [Int32] = []
        var accepted = 0
        do {
            while tokens.count < budget && target.continuationPosition < maxContext {
                try Task.checkCancellation()
                // Use the actual generation sampler, including FP16 softmax
                // rounding and its tie rule, not an assumed raw-logit argmax.
                let token = try sampleToken()
                guard token >= 0, Int(token) < vocab else {
                    throw FlashNextForwardRunnerError.invalidInput("invalid verification sample")
                }
                let matched = tokens.count < proposals.count && proposals[tokens.count] == token
                try await target.produce(token: token, position: target.continuationPosition, into: logits)
                tokens.append(token)
                if matched { accepted += 1 }
                try didVerifyToken?(tokens.count)
                if stopTokens.contains(token) { return .init(tokens: tokens, accepted: accepted, reachedStop: true) }
                if !matched { break }
            }
            return .init(tokens: tokens, accepted: accepted, reachedStop: false)
        } catch {
            // Roll back both recurrent/cache owners and the caller-visible head.
            // If recovery itself fails, reset the pair and refuse continuation.
            do {
                try target.restoreDecodeCheckpoint(targetBase)
                try primer.restore(primerBase)
                saved.withUnsafeBufferPointer { words.update(from: $0.baseAddress!, count: vocab) }
            } catch {
                reset()
                throw error
            }
            throw error
        }
    }

    private func sampleToken(using alternate: RawCompletionScratch? = nil) throws -> Int32 {
        let selected = alternate ?? scratch
        let values = selected.logits.contents().assumingMemoryBound(to: Float16.self)
        var hasFinite = false
        for i in 0..<vocab {
            guard !values[i].isNaN, values[i] != .infinity else {
                throw FlashNextForwardRunnerError.invalidInput("invalid native verification/sample logits")
            }
            hasFinite = hasFinite || values[i].isFinite
        }
        guard hasFinite else { throw FlashNextForwardRunnerError.invalidInput("no finite native sample logits") }
        guard let cb = context.queue.makeCommandBuffer() else {
            throw FlashNextForwardRunnerError.commandFailed("cannot sample verification token")
        }
        selected.sampler.sample(commandBuffer: cb, logits: selected.logits, probs: selected.probs,
            history: [], config: .init(temperature: 0), position: target.continuationPosition,
            outToken: selected.outToken)
        cb.commit()
        cb.waitUntilCompleted()
        if let error = cb.error { throw error }
        return Int32(bitPattern: selected.outToken.contents().assumingMemoryBound(to: UInt32.self).pointee)
    }
}

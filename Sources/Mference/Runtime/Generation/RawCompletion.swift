import Foundation
import Metal

/// Streaming callbacks from `runRawCompletion`. `.prefill` reports monotonic
/// producer-defined prompt progress; scalar replay reports per token, while a
/// prefill-capable producer may report per internal chunk. `.token` fires per
/// decoded non-stop token; `.tail` carries the detokenizer flush remainder at a
/// stop or Gemma channel/tool boundary.
public enum RawDecodeProgress: Sendable {
    case prefill(done: Int, total: Int)
    case token(index: Int, id: Int32, delta: String)
    case tail(String)
}

public enum RawCompletionStart: Sendable, Equatable {
    case reset
    case resume(cachedPromptTokens: Int)
}

public struct RawDecodeResult: Sendable {
    public let prefillTokens: Int
    public let cachedPromptTokens: Int
    public let computedPrefillTokens: Int
    public let prefillSeconds: Double
    public let newTokens: Int
    public let decodeSeconds: Double
    public let reason: StopReason
    public let kvPosition: Int
    public let kvBackedTokenIDs: [Int32]
    public let uncommittedBoundaryTokenIDs: [Int32]
    public let prefillExecution: PrefillExecutionReport?
}

/// Preallocated per-generation buffers (two 512 KiB vocab buffers plus a token
/// slot) and sampler. A warm session reuses them for every token, avoiding
/// per-token Metal buffer allocation.
///
/// `@unchecked Sendable`: the buffers and sampler are exclusively owned by one
/// generation at a time — the single-in-flight guard upstream is the contract.
public struct RawCompletionScratch: @unchecked Sendable {
    let logits: MTLBuffer
    let probs: MTLBuffer
    let outToken: MTLBuffer
    let sampler: Sampler
    /// Caller-owned logits, probability and output-token buffer capacities.
    var diagnosticBufferBytes: UInt64 { uniqueBufferBytes([logits, probs, outToken]) + sampler.diagnosticBufferBytes }

    public init(context: MetalContext, vocab: Int, logitSoftcap: Float = 30.0) throws {
        guard let logits = context.device.makeBuffer(length: vocab * MemoryLayout<Float16>.size,
                                                     options: .storageModeShared),
              let probs = context.device.makeBuffer(length: vocab * MemoryLayout<Float16>.size,
                                                    options: .storageModeShared),
              let outToken = context.device.makeBuffer(length: MemoryLayout<UInt32>.size,
                                                       options: .storageModeShared)
        else {
            throw ModelError.residentBufferWrapFailed
        }
        self.logits = logits
        self.probs = probs
        self.outToken = outToken
        self.sampler = try Sampler(context: context, vocab: vocab,
                                   logitSoftcap: logitSoftcap)
    }
}

extension GenerationConfig {
    /// A pure-greedy config can use the fused head's GPU argmax
    /// (`RealForwardRunner.lastGreedyToken`) instead of sampling from the
    /// logits buffer. Anything else needs real logits.
    public var isPureGreedy: Bool {
        temperature == 0 && (repeatLastN == 0 || (repetitionPenalty == 1 && presencePenalty == 0 && frequencyPenalty == 0))
    }

}

/// Raw-completion prefill + decode loop shared by the CLI and the server.
/// Consumes pre-encoded `promptIds` (BOS + verbatim encode upstream — no chat
/// template). Stop handling, detokenizer flush ordering, and history append
/// ordering are shared by both front ends.
///
/// When the producer runs the fused lm_head (`RealForwardRunner` default) the
/// logits buffer is never written; the loop then requires a pure-greedy config
/// and reads `lastGreedyToken`. Callers with sampling configs must construct
/// the runner with `forceLogitsHead: true`.
public func runRawCompletion(producer: any LogitProducer,
                             tokenizer: MFTokenizer,
                             promptIds: [Int32],
                             config: GenerationConfig,
                             context: MetalContext,
                             scratch: RawCompletionScratch,
                             prefillConfig: PrefillRuntimeConfig = .defaultChunked,
                             start: RawCompletionStart = .reset,
                             prefillCheckpoint: (position: Int, capture: () throws -> Void)? = nil,
                             shouldStop: () -> Bool = { false },
                             onProgress: (RawDecodeProgress) -> Void) async throws -> RawDecodeResult {
    try config.validate()
    guard !promptIds.isEmpty else {
        throw GeneratorError.emptyPrompt
    }
    let fusedRunner = producer as? any FusedHeadLogitProducer
    let fusedGreedy = fusedRunner?.usesFusedGreedyHead == true
    guard !fusedGreedy || config.isPureGreedy else {
        throw PrefillError.unsupportedPrefillSeed(
            "the fused-head producer cannot serve this sampling configuration; use a logits head")
    }

    let cachedPromptTokens: Int
    switch start {
    case .reset:
        cachedPromptTokens = 0
    case .resume(let count):
        guard count > 0, count < promptIds.count else {
            throw GeneratorError.invalidContinuation(
                "cached prompt token count must be greater than zero and less than the effective prompt")
        }
        guard producer is any ContinuableLogitProducer else {
            throw GeneratorError.invalidContinuation(
                "producer does not support continuation")
        }
        cachedPromptTokens = count
    }
    if let checkpoint = prefillCheckpoint {
        guard checkpoint.position >= cachedPromptTokens, checkpoint.position <= promptIds.count else {
            throw GeneratorError.invalidContinuation("prefill checkpoint is outside the computed prompt")
        }
    }
    let computedPrefillTokens = promptIds.count - cachedPromptTokens

    var detok = MFDetokenizer(tokenizer: tokenizer)
    var history = Array(promptIds.prefix(cachedPromptTokens))
    history.reserveCapacity(promptIds.count + config.maxNewTokens)

    if let context = producer as? any ContextWindowReporting,
       promptIds.count + config.maxNewTokens > context.maxContext {
        throw GeneratorError.contextOverflow(prompt: promptIds.count,
                                             maxNew: config.maxNewTokens,
                                             maxContext: context.maxContext)
    }
    switch start {
    case .reset:
        producer.reset()
    case .resume:
        let continuable = producer as! any ContinuableLogitProducer
        try continuable.prepareForContinuation(expectedPosition: cachedPromptTokens)
    }
    let prefillStart = Date()
    var position = cachedPromptTokens
    var prefillSeed: PrefillSeed?
    var prefillExecution: PrefillExecutionReport?
    if prefillCheckpoint?.position == position {
        try Task.checkCancellation()
        try prefillCheckpoint?.capture()
    }
    var boundaries: [Int] = []
    if let checkpoint = prefillCheckpoint, checkpoint.position > position,
       checkpoint.position < promptIds.count { boundaries.append(checkpoint.position) }
    boundaries.append(promptIds.count)
    for end in boundaries {
        let segmentStart = position
        let prefillTokens = promptIds[segmentStart..<end]
        switch prefillConfig.mode {
        case .chunked where producer is any ChunkedPrefillRunner:
            let chunked = producer as! any ChunkedPrefillRunner
            let mode: PrefillOutputMode = fusedGreedy ? .greedyIfAvailable : .logits
            let result = try await chunked.prefillChunked(tokens: prefillTokens,
                startPosition: position, outputMode: mode, config: prefillConfig,
                into: scratch.logits) { done in
                onProgress(.prefill(done: segmentStart + done, total: promptIds.count))
            }
            if mode == .logits, result.seed != .logitsWritten {
                throw PrefillError.unsupportedPrefillSeed(
                    "RawCompletion chunked prefill requested logits but producer returned \(result.seed)")
            }
            if case .greedyToken = result.seed, !config.isPureGreedy {
                throw PrefillError.unsupportedPrefillSeed(
                    "RawCompletion chunked prefill returned a greedy token for a sampling config")
            }
            position = result.newPosition
            prefillSeed = result.seed
            if let execution = result.execution {
                if segmentStart == cachedPromptTokens { prefillExecution = execution }
                else { prefillExecution?.append(execution) }
            } else { prefillExecution = nil }
            history.append(contentsOf: prefillTokens)
        case .chunked:
            throw PrefillError.chunkedUnsupported(PrefillError.chunkedRequiresChunkedRunnerReason)
        case .off:
            var execution = prefillExecution ?? PrefillExecutionReport()
            let headless = producer as? any HeadlessSequentialPrefillRunner
            let exactPrefill = producer as? any ExactPrefillLogitProducer
            for t in prefillTokens {
                try Task.checkCancellation()
                if position + 1 < promptIds.count, let headless {
                    try await headless.produceWithoutLogits(token: t, position: position)
                } else if let exactPrefill {
                    try await exactPrefill.produceExactPrefill(token: t, position: position, into: scratch.logits)
                } else {
                    try await producer.produce(token: t, position: position, into: scratch.logits)
                }
                position += 1
                execution.recordReplay(1, reason: "prefill_disabled")
                history.append(t)
                onProgress(.prefill(done: position, total: promptIds.count))
            }
            prefillExecution = execution
        }
        if prefillCheckpoint?.position == position {
            try Task.checkCancellation()
            try prefillCheckpoint?.capture()
        }
    }

    let decodeStart = Date()
    let prefillSeconds = decodeStart.timeIntervalSince(prefillStart)
    var stopMatcher = StreamingStopMatcher(stops: config.stopStrings)
    var generated = 0
    var reason: StopReason = .maxTokens
    var uncommittedBoundaryTokenIDs: [Int32] = []

    while true {
        try Task.checkCancellation()

        let tokenID: Int32
        if generated == 0, let seed = prefillSeed {
            switch seed {
            case .greedyToken(let token):
                tokenID = Int32(bitPattern: token)
            case .logitsWritten:
                tokenID = try sampleOnce(scratch: scratch, context: context,
                                         history: history, config: config, position: generated)
            }
        } else if fusedGreedy {
            tokenID = Int32(bitPattern: fusedRunner!.lastGreedyToken)
        } else {
            tokenID = try sampleOnce(scratch: scratch, context: context,
                                     history: history, config: config, position: generated)
        }
        generated += 1
        uncommittedBoundaryTokenIDs = [tokenID]

        if tokenizer.stopTokenIDs.contains(tokenID) || config.extraStopTokens.contains(tokenID) {
            if tokenID == tokenizer.endOfTurnID {
                reason = .endOfTurn
            } else if tokenID == tokenizer.toolResponseID {
                reason = .toolCalls
            } else {
                reason = .eos
            }
            let tail = stopMatcher.push(detok.flush()) + stopMatcher.finish()
            if !tail.isEmpty { onProgress(.tail(tail)) }
            break
        }

        if tokenizer.dialect == .gemma,
           tokenID == tokenizer.channelStartID || tokenID == tokenizer.channelEndID
            || tokenID == tokenizer.toolCallStartID || tokenID == tokenizer.toolCallEndID {
            // Special tokens are removed before byte-fallback decoding. Flush
            // pending bytes in the old channel before delivering its boundary,
            // then start a fresh segment so they cannot reappear in the next.
            let tail = stopMatcher.push(detok.flush())
            if !tail.isEmpty { onProgress(.tail(tail)) }
            detok = MFDetokenizer(tokenizer: tokenizer)
        }
        let delta = detok.push(tokenID)
        let visible = stopMatcher.push(delta)
        onProgress(.token(index: generated - 1, id: tokenID, delta: visible))

        let hitStopString = stopMatcher.isStopped || shouldStop()
        let hitMax = generated >= config.maxNewTokens
        if hitStopString || hitMax {
            let tail = stopMatcher.push(detok.flush()) + stopMatcher.finish()
            if !tail.isEmpty { onProgress(.tail(tail)) }
            reason = hitStopString ? .stopString : .maxTokens
            break
        }

        history.append(tokenID)
        try await producer.produce(token: tokenID, position: position, into: scratch.logits)
        position += 1
        uncommittedBoundaryTokenIDs.removeAll(keepingCapacity: true)
    }

    return RawDecodeResult(prefillTokens: promptIds.count,
                           cachedPromptTokens: cachedPromptTokens,
                           computedPrefillTokens: computedPrefillTokens,
                           prefillSeconds: prefillSeconds,
                           newTokens: generated,
                           decodeSeconds: Date().timeIntervalSince(decodeStart),
                           reason: reason,
                           kvPosition: position,
                           kvBackedTokenIDs: history,
                           uncommittedBoundaryTokenIDs: uncommittedBoundaryTokenIDs,
                           prefillExecution: prefillExecution)
}

private func sampleOnce(scratch: RawCompletionScratch, context: MetalContext,
                        history: [Int32], config: GenerationConfig, position: Int) throws -> Int32 {
    let cb = context.queue.makeCommandBuffer()!
    cb.label = "sample position=\(position)"
    scratch.sampler.sample(commandBuffer: cb, logits: scratch.logits, probs: scratch.probs,
                           history: history, config: config, position: position,
                           outToken: scratch.outToken)
    cb.commit(); cb.waitUntilCompleted()
    try checkCommandBufferError(cb)
    return Int32(bitPattern: scratch.outToken.contents().load(as: UInt32.self))
}

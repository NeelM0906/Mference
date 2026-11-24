import Foundation
import LlamaCpp
import Mference
import Synchronization

/// A GGUF model loaded into llama.cpp: one model, one context, one sequence.
///
/// `prepare` renders the GGUF's own chat template and tokenizes with the
/// model's vocabulary; `generate` prefills (reusing whatever prefix of the
/// prompt the context already holds), samples one token at a time, and turns
/// the tokens into the same events, finish reasons and usage as the native
/// path (`StructuredAssistantDecoder`, `StreamingStopMatcher`).
///
/// With a vision projector, a request that carries images is tokenized by
/// `mtmd_tokenize` and prefilled chunk by chunk as llama.cpp's `mtmd-helper`
/// does (images encoded, then decoded as embeddings at M-RoPE positions);
/// it always decodes plainly, without MTP. A request without images takes
/// exactly the text-only path.
actor LlamaModel: EngineModel {
    nonisolated let descriptor: EngineModelDescriptor
    nonisolated let traits: EngineModelTraits

    private let handle: LlamaModelHandle
    private let contextOwner: LlamaContextHandle
    /// MTP speculative decoding (Deep), when the GGUF has an MTP head and
    /// draft decoding is on; nil runs plain decoding.
    private let drafter: LlamaDrafter?
    private var context: OpaquePointer?
    private let vocabulary: LlamaVocabulary
    private let tokenizer: MFTokenizer
    private let template: LlamaChatTemplate
    private let batchSize: Int
    private let abortFlag = LlamaAbortFlag()

    /// What the context's memory holds, for prompt-prefix reuse.
    private var promptCache = LlamaPromptCache()
    /// Every token the last generation sampled, the end-of-generation token
    /// included. Kept for the token-identity checks against upstream tools.
    private(set) var lastSampledTokens: [Int32] = []

    /// The vision projector (`mmproj`) file, when the model has one.
    private let projector: URL?
    private let projectorUsesGPU: Bool
    /// The loaded projector (`mtmd_context`), from the first image on.
    private var media: OpaquePointer?
    /// The batch media prompts prefill through (explicit positions).
    private var mediaBatch: OpaquePointer?
    /// The text that stands for an image in a rendered prompt. Random per
    /// load, as llama.cpp's server makes it, so no typed text can be taken
    /// for an image.
    private let mediaMarker = "<__media_\(UUID().uuidString.lowercased())__>"
    /// The media prompt `prepare` last tokenized, for `generate` to reuse.
    private var preparedMedia: LlamaMediaPrompt?

    /// The prompt text `prepare` tokenizes: the GGUF's template rendered for
    /// `request`, with one media marker per image.
    func renderedPrompt(_ request: EngineRequest) throws -> String {
        let rendered = try template.render(
            messages: request.messages,
            tools: request.tools,
            keywordArguments: LlamaChatTemplate.keywordArguments(
                reasoningEffort: request.reasoningEffort,
                preserveThinking: request.preserveThinking,
                readsEnableThinking: template.readsEnableThinking,
                readsReasoningEffort: template.readsReasoningEffort),
            mediaCounts: request.media.mapValues(\.count).filter { $0.value > 0 },
            mediaMarker: mediaMarker)
        return template.strippingImplicitSpecialTokens(rendered, addsBOS: vocabulary.addsBOS,
                                                       addsEOS: vocabulary.addsEOS)
    }

    init(descriptor: EngineModelDescriptor,
         handle: LlamaModelHandle,
         context: OpaquePointer,
         vocabulary: LlamaVocabulary,
         tokenizer: MFTokenizer,
         template: LlamaChatTemplate,
         batchSize: Int,
         mtp: OpaquePointer? = nil,
         draftLimit: Int = LlamaBackend.draftLimit,
         projector: URL? = nil,
         projectorUsesGPU: Bool = true) {
        self.descriptor = descriptor
        self.projector = projector
        self.projectorUsesGPU = projectorUsesGPU
        self.handle = handle
        let contextOwner = LlamaContextHandle(context, model: handle)
        self.contextOwner = contextOwner
        let drafter = mtp.map { LlamaDrafter($0, draftLimit: draftLimit, context: contextOwner) }
        self.drafter = drafter
        self.context = context
        self.vocabulary = vocabulary
        self.tokenizer = tokenizer
        self.template = template
        self.batchSize = batchSize
        traits = EngineModelTraits(
            chatDialect: tokenizer.dialect,
            generationDefaults: .defaults,
            maxContext: Int(llama_n_ctx(context)),
            acceptsReasoningEffort: template.readsEnableThinking || template.readsReasoningEffort,
            // Qwen 3.8's source template, which reads the effort level and
            // rejects developer messages, as the server's Qwen 3.8 rules do.
            supportsQwenReasoningEffort: template.readsReasoningEffort,
            reusesPromptPrefixes: drafter == nil,
            usesDraftDecoding: drafter != nil,
            supportsVision: projector != nil)
        llama_set_abort_callback(context, { data in
            guard let data else { return false }
            return Unmanaged<LlamaAbortFlag>.fromOpaque(data).takeUnretainedValue().isSet
        }, Unmanaged.passUnretained(abortFlag).toOpaque())
        if let draftContext = drafter?.pointer.flatMap(mference_mtp_draft_context) {
            llama_set_abort_callback(draftContext, { data in
                guard let data else { return false }
                return Unmanaged<LlamaAbortFlag>.fromOpaque(data).takeUnretainedValue().isSet
            }, Unmanaged.passUnretained(abortFlag).toOpaque())
        }
    }

    // MARK: EngineModel

    func prepare(_ request: EngineRequest) async throws -> EnginePreparedPrompt {
        guard context != nil else { throw EngineError.modelUnloaded(descriptor.id) }
        if request.reasoningEffort != nil, !traits.acceptsReasoningEffort {
            throw EngineError.invalidRequest(message: "reasoning_effort is not supported by this model",
                                             param: "reasoning_effort", code: "unsupported_value")
        }
        try request.validateMedia(supportsVision: traits.supportsVision)
        preparedMedia = nil
        let tokens: [Int32]
        if request.hasMedia {
            let prompt = try mediaPrompt(request)
            preparedMedia = prompt
            tokens = prompt.layout.promptTokens
        } else {
            tokens = vocabulary.tokenIDs(try renderedPrompt(request), addSpecial: true)
        }
        // (`renderedPrompt` already dropped any BOS/EOS tokenization adds.)
        guard !tokens.isEmpty else {
            throw EngineError.invalidRequest(message: "the rendered prompt is empty",
                                             param: "messages", code: "invalid_chat_template")
        }
        guard tokens.count < traits.maxContext else {
            throw EngineError.invalidRequest(message: "effective prompt exceeds the configured context",
                                             param: "messages", code: "context_length_exceeded")
        }
        return EnginePreparedPrompt(request: request, promptTokens: tokens,
                                    needsToolTemplate: request.needsToolTemplate)
    }

    func generate(_ prepared: EnginePreparedPrompt,
                  onEvent: @escaping @Sendable (EngineEvent) -> Void) async throws -> EngineCompletion {
        guard let context else { throw EngineError.modelUnloaded(descriptor.id) }
        abortFlag.clear()
        return try await withTaskCancellationHandler {
            try run(prepared, context: context, onEvent: onEvent)
        } onCancel: {
            abortFlag.set()
        }
    }

    /// The text of `tokens` with special tokens spelled out, as llama.cpp's
    /// tools print a generation, without a final end-of-generation token.
    func printedText(_ tokens: [Int32]) -> String {
        var tokens = tokens
        if let last = tokens.last, vocabulary.isEndOfGeneration(last) { tokens.removeLast() }
        return String(decoding: tokens.flatMap { vocabulary.pieceBytes($0, special: true) }, as: UTF8.self)
    }

    /// Forgets the prompt prefix the memory holds, so the next request
    /// prefills from scratch (used by the token-identity checks, which compare
    /// cold runs with the upstream tools).
    func discardCachedPrompt() {
        promptCache.invalidate()
    }

    func unload() async {
        abortFlag.set()
        preparedMedia = nil
        if let mediaBatch { llama_batch_ext_free(mediaBatch) }
        mediaBatch = nil
        if let media { mtmd_free(media) }
        media = nil
        context = nil
        drafter?.free()
        contextOwner.free()
        promptCache.invalidate()
        handle.free()
    }

    // MARK: Generation

    private func run(_ prepared: EnginePreparedPrompt,
                     context: OpaquePointer,
                     onEvent: @escaping @Sendable (EngineEvent) -> Void) throws -> EngineCompletion {
        let request = prepared.request
        let prompt = prepared.promptTokens
        let maxContext = traits.maxContext
        guard !prompt.isEmpty, prompt.count < maxContext else {
            throw EngineError.invalidRequest(message: "effective prompt exceeds the configured context",
                                             param: "messages", code: "context_length_exceeded")
        }
        try request.generationConfig.validate()
        let config = request.generationConfig
        let maxNewTokens = min(request.maximumCompletionTokens, maxContext - prompt.count)

        // Tokens become events the same way on both paths.
        let decoder = StructuredAssistantDecoder(
            tokenizer: tokenizer,
            allowedTools: Set(request.tools.map(\.name)),
            startsInThought: tokenizer.startsInThinking(reasoningEffort: request.reasoningEffort,
                                                        promptIDs: prompt),
            toolDefinitions: request.tools)
        var assembler = LlamaEventAssembler(stops: config.stopStrings, onEvent: onEvent)
        // Reasoning bypasses the stop matcher, as on the native path.
        var reasoning = ""
        decoder.onReasoning = { [onEvent] text in
            reasoning += text
            onEvent(.reasoning(text))
        }
        var utf8 = LlamaUTF8Stream()
        lastSampledTokens = []
        var generated = 0
        var hitLimit = false

        /// Records one sampled token; false once generation must end (end of
        /// generation, a stop string, or the token budget).
        func take(_ token: Int32) throws -> Bool {
            lastSampledTokens.append(token)
            generated += 1
            if vocabulary.isEndOfGeneration(token)
                || tokenizer.stopTokenIDs.contains(token)
                || config.extraStopTokens.contains(token) {
                return false
            }
            let delta = utf8.push(vocabulary.pieceBytes(token, special: false))
            try assembler.take(decoder.consume(tokenID: token, delta: delta))
            if assembler.isStopped { return false }
            guard generated < maxNewTokens else {
                hitLimit = true
                return false
            }
            return true
        }

        let reused: Int
        let clock = ContinuousClock()
        let prefillSeconds: Double
        let decodeStart: ContinuousClock.Instant
        var draftProposed: Int?
        var draftAccepted: Int?

        if let drafter, let mtp = drafter.pointer, !request.hasMedia {
            // MTP: llama.cpp's speculative loop owns both memories and starts
            // from empty ones (no prefix reuse).
            reused = 0
            // From here until a clean return, the memory may hold anything.
            promptCache.invalidate()
            let prefillStart = clock.now
            let status = prompt.withUnsafeBufferPointer { buffer in
                mference_mtp_begin(mtp, buffer.baseAddress, Int32(buffer.count),
                                   LlamaSampler.mtpSampling(config, contextSize: Int32(maxContext)))
            }
            if status == 2 || abortFlag.isSet { throw CancellationError() }
            guard status == 0 else { throw LlamaBackendError.decodeFailed(status) }
            llama_synchronize(context)
            prefillSeconds = (clock.now - prefillStart).seconds
            decodeStart = clock.now
            var proposed = 0
            var accepted = 0
            var buffer = [Int32](repeating: 0, count: drafter.stepCapacity)
            stepping: while true {
                try Task.checkCancellation()
                guard generated < maxNewTokens else {
                    hitLimit = true
                    break
                }
                var drafted: Int32 = 0
                let count = mference_mtp_step(mtp, Int32(maxNewTokens - generated),
                                              &buffer, Int32(buffer.count), &drafted)
                if count == -2 || abortFlag.isSet { throw CancellationError() }
                guard count > 0 else { throw LlamaBackendError.decodeFailed(count) }
                proposed += Int(drafted)
                // Of the step's tokens, all but the target's last one are
                // accepted drafts; count only those that reached the output.
                var taken = 0
                defer { accepted += min(taken, Int(count) - 1) }
                for token in buffer.prefix(Int(count)) {
                    taken += 1
                    guard try take(token) else { break stepping }
                }
            }
            draftProposed = proposed
            draftAccepted = accepted
        } else {
            // Decides what to reuse, trims or clears the memory to match, and
            // leaves the cache untrusted until a clean return.
            let memory = llama_get_memory(context)
            let prefillStart = clock.now
            if request.hasMedia {
                reused = try prefillMedia(request, context: context)
            } else {
                reused = promptCache.begin(
                    prompt: prompt,
                    trim: { llama_memory_seq_rm(memory, 0, llama_pos($0), -1) },
                    clear: { llama_memory_clear(memory, true) })
                var position = reused
                while position < prompt.count {
                    try Task.checkCancellation()
                    let end = min(position + batchSize, prompt.count)
                    try decode(Array(prompt[position..<end]), context: context)
                    position = end
                }
            }
            // llama_decode only queues GPU work; wait for it so the prefill time
            // is the prefill's, not the first sampled token's.
            llama_synchronize(context)
            prefillSeconds = (clock.now - prefillStart).seconds

            let sampler = LlamaSampler(config: config, vocabularySize: Int32(vocabulary.tokenCount),
                                       contextSize: Int32(maxContext),
                                       suppressedTokens: vocabulary.suppressedTokens)
            // Text tokens only: llama.cpp's server skips an image's tokens.
            for token in prompt where token != EngineMedia.placeholderToken { sampler.accept(token) }
            decodeStart = clock.now
            while true {
                try Task.checkCancellation()
                guard generated < maxNewTokens else {
                    hitLimit = true
                    break
                }
                guard try take(sampler.sample(context: context)) else { break }
                try decode([lastSampledTokens[lastSampledTokens.count - 1]], context: context)
            }
        }
        try assembler.take(decoder.consumeFlushedText(utf8.finish()))
        try assembler.take(decoder.finish())
        assembler.finish()
        let decodeSeconds = (clock.now - decodeStart).seconds
        try Task.checkCancellation()

        let finishReason: EngineFinishReason =
            !assembler.toolCalls.isEmpty ? .toolCalls : (hitLimit && !assembler.isStopped ? .length : .stop)
        let counts = decoder.payloadTokenCounts
        // The plain path's memory now holds exactly what the cache recorded;
        // the MTP loop's is not tracked, so the next request starts from empty
        // (as does one after a media request on an MTP model).
        if drafter == nil { promptCache.commit() } else { promptCache.invalidate() }
        return EngineCompletion(
            content: assembler.content,
            reasoningContent: reasoning.isEmpty ? nil : reasoning,
            toolCalls: assembler.toolCalls,
            finishReason: finishReason,
            usage: EngineUsage(promptTokens: prompt.count,
                               completionTokens: generated,
                               cachedPromptTokens: reused,
                               reasoningTokens: counts?.reasoning,
                               visibleTokens: assembler.isStopped ? nil : counts?.visible),
            timings: EngineTimings(prefillSeconds: prefillSeconds, decodeSeconds: decodeSeconds,
                                   draftTokensProposed: draftProposed,
                                   draftTokensAccepted: draftAccepted))
    }

    // MARK: Media

    /// The loaded projector, loading it on first use.
    private func mediaContext() throws -> OpaquePointer {
        if let media { return media }
        guard let projector, let model = handle.pointer, let context else {
            throw EngineError.invalidRequest(message: "this model does not accept images",
                                             param: "messages", code: "unsupported_content")
        }
        var params = mtmd_context_params_default()
        params.use_gpu = projectorUsesGPU
        params.print_timings = false
        let loaded = mediaMarker.withCString { marker -> OpaquePointer? in
            params.media_marker = marker
            return mtmd_init_from_file(projector.path, model, params)
        }
        guard let loaded else { throw LlamaBackendError.projectorFailed("could not load \(projector.path)") }
        guard mtmd_support_vision(loaded), let batch = llama_batch_ext_init(context) else {
            mtmd_free(loaded)
            throw LlamaBackendError.projectorFailed("\(projector.lastPathComponent) has no vision encoder")
        }
        media = loaded
        mediaBatch = batch
        return loaded
    }

    /// `request` rendered and tokenized with its images.
    private func mediaPrompt(_ request: EngineRequest) throws -> LlamaMediaPrompt {
        let media = try mediaContext()
        return try LlamaMediaPrompt(rendered: try renderedPrompt(request), media: request.orderedMedia,
                                    context: media)
    }

    /// Prefills a prompt with images, reusing what the memory holds of it, as
    /// llama.cpp's `mtmd_helper_eval_chunks` does: text at one position per
    /// token, each image encoded and then decoded as embeddings at its M-RoPE
    /// positions (`t`, `y`, `x`, `z`), the position advancing by the image's
    /// `n_pos`. Logits are kept for the last prompt token only. Returns the
    /// prompt tokens reused.
    private func prefillMedia(_ request: EngineRequest, context: OpaquePointer) throws -> Int {
        let prompt: LlamaMediaPrompt
        if let prepared = preparedMedia, prepared.mediaHashes == request.orderedMedia.map(\.hash),
           prepared.rendered == (try renderedPrompt(request)) {
            prompt = prepared
        } else {
            prompt = try mediaPrompt(request)
        }
        preparedMedia = nil
        guard let media, let batch = mediaBatch, let model = handle.pointer else {
            throw EngineError.modelUnloaded(descriptor.id)
        }
        let layout = prompt.layout
        let memory = llama_get_memory(context)
        let keys = layout.cacheKeys
        let reusedPositions = promptCache.begin(
            prompt: keys,
            media: layout.mediaStarts,
            trim: { llama_memory_seq_rm(memory, 0, llama_pos($0), -1) },
            clear: { llama_memory_clear(memory, true) })
        var plan = layout.remaining(afterReusing: reusedPositions)
        if plan == nil {
            // Ends inside an image (cannot happen with whole-image keys):
            // start over.
            llama_memory_clear(memory, true)
            promptCache.invalidate()
            plan = layout.remaining(afterReusing: 0)
        }
        guard let (pieces, reusedTokens) = plan else { throw LlamaBackendError.decodeFailed(-1) }

        let embeddingWidth = Int(llama_model_n_embd_inp(model))
        let usesMRoPE = mtmd_decode_use_mrope(media)
        for (pieceIndex, piece) in pieces.enumerated() {
            try Task.checkCancellation()
            let isLast = pieceIndex == pieces.count - 1
            switch piece {
            case .text(let segment, let skipping, let start):
                guard case .text(let all) = layout.segments[segment] else { continue }
                let tokens = Array(all[skipping...])
                var offset = 0
                while offset < tokens.count {
                    let end = min(offset + batchSize, tokens.count)
                    llama_batch_ext_clear(batch)
                    for index in offset..<end {
                        let entry = llama_batch_ext_add_token(batch, 0, tokens[index])
                        var position: [llama_pos] = [llama_pos(start + index), 0, 0, 0]
                        guard entry >= 0, llama_batch_ext_set_pos(batch, entry, &position) else {
                            throw LlamaBackendError.decodeFailed(-1)
                        }
                        if isLast, index == tokens.count - 1 {
                            _ = llama_batch_ext_set_output_logits(batch, entry, true)
                        }
                    }
                    try process(batch, context: context)
                    promptCache.append(Array(tokens[offset..<end]))
                    offset = end
                }
            case .media(let segment, let start):
                guard case .media(let count, let positions, let hash) = layout.segments[segment],
                      let chunk = prompt.chunk(segment) else { continue }
                guard mtmd_encode_chunk(media, chunk) == 0, let rows = mtmd_get_output_embd(media) else {
                    if abortFlag.isSet { throw CancellationError() }
                    throw LlamaBackendError.projectorFailed("an image could not be encoded")
                }
                let imageTokens = mtmd_input_chunk_get_tokens_image(chunk)
                let nonCausal = mtmd_decode_use_non_causal(media, chunk)
                if nonCausal { llama_set_causal_attn(context, false) }
                defer { if nonCausal { llama_set_causal_attn(context, true) } }
                var offset = 0
                while offset < count {
                    let end = min(offset + batchSize, count)
                    llama_batch_ext_clear(batch)
                    for index in offset..<end {
                        let entry = llama_batch_ext_add_embd(
                            batch, 0, llama_embd(data: rows + index * embeddingWidth, n_rows: 1,
                                                 n_embd: embeddingWidth))
                        var position: [llama_pos]
                        if usesMRoPE, let imageTokens {
                            let at = mtmd_image_tokens_get_decoder_pos(imageTokens, llama_pos(start), index)
                            position = [llama_pos(at.t), llama_pos(at.y), llama_pos(at.x), llama_pos(at.z)]
                        } else {
                            position = [llama_pos(start + index), 0, 0, 0]
                        }
                        guard entry >= 0, llama_batch_ext_set_pos(batch, entry, &position) else {
                            throw LlamaBackendError.decodeFailed(-1)
                        }
                    }
                    try process(batch, context: context)
                    offset = end
                }
                promptCache.appendMedia(Array(keys[start..<(start + positions)]), hash: hash)
            }
        }
        llama_batch_ext_clear(batch)
        return reusedTokens
    }

    private func process(_ batch: OpaquePointer, context: OpaquePointer) throws {
        let status = llama_process(context, LLAMA_PROCESS_TYPE_DECODE, batch)
        if status == 2 || abortFlag.isSet { throw CancellationError() }
        guard status == 0 else { throw LlamaBackendError.decodeFailed(status) }
    }

    private func decode(_ tokens: [Int32], context: OpaquePointer) throws {
        var tokens = tokens
        let status = tokens.withUnsafeMutableBufferPointer { buffer in
            llama_decode(context, llama_batch_get_one(buffer.baseAddress, Int32(buffer.count)))
        }
        if status == 2 || abortFlag.isSet { throw CancellationError() }
        guard status == 0 else { throw LlamaBackendError.decodeFailed(status) }
        promptCache.append(tokens)
    }
}

/// The tokens a llama.cpp context's memory holds, for prompt-prefix reuse.
///
/// A request calls `begin` once: it decides how much of the new prompt the
/// memory already holds (while the record is still trusted), trims or clears
/// the memory to match, and only then marks the record untrusted. Decoded
/// tokens are `append`ed, and `commit` trusts the record again after a clean
/// finish; a request that fails or is cancelled never commits, so the next one
/// starts from an empty memory. At least one prompt token is always left to
/// decode, for its logits.
///
/// An image's entries are short keys derived from its hash, so the record
/// also keeps each image's full SHA-256 by the position it starts at, and an
/// image is reused only when the memory holds that very image there.
struct LlamaPromptCache {
    private(set) var tokens: [Int32] = []
    private(set) var isTrusted = true
    /// The full hash of each image the memory holds, by its first position.
    private(set) var mediaHashes: [Int: String] = [:]

    /// Returns how many leading prompt tokens are reused. `media` gives the
    /// prompt's images (first position to full hash); reuse stops at the
    /// first one the memory does not hold. `trim(n)` drops everything from
    /// position `n` on and reports whether the memory could (a recurrent state
    /// cannot roll back); `clear()` empties it.
    mutating func begin(prompt: [Int32], media: [Int: String] = [:],
                        trim: (Int) -> Bool, clear: () -> Void) -> Int {
        defer { isTrusted = false }
        var common = 0
        if isTrusted {
            var limit = min(tokens.count, prompt.count - 1)
            for (start, hash) in media where start < limit && mediaHashes[start] != hash {
                limit = start
            }
            while common < limit, tokens[common] == prompt[common] { common += 1 }
            if common == tokens.count {
                return common
            }
            if common > 0, trim(common) {
                tokens.removeLast(tokens.count - common)
                mediaHashes = mediaHashes.filter { $0.key < common }
                return common
            }
        }
        clear()
        tokens = []
        mediaHashes = [:]
        return 0
    }

    mutating func append(_ decoded: [Int32]) {
        tokens += decoded
    }

    /// Records an image decoded at the next position: its keys and full hash.
    mutating func appendMedia(_ keys: [Int32], hash: String) {
        mediaHashes[tokens.count] = hash
        tokens += keys
    }

    mutating func commit() {
        isTrusted = true
    }

    /// Forgets the record; the next `begin` clears the memory.
    mutating func invalidate() {
        isTrusted = false
        tokens = []
        mediaHashes = [:]
    }
}

/// Collects decoder events into the completion and forwards them, applying
/// stop strings to visible text the same way the server's native path does.
struct LlamaEventAssembler {
    private var stopMatcher: StreamingStopMatcher
    private let onEvent: @Sendable (EngineEvent) -> Void
    private(set) var content = ""
    private(set) var toolCalls: [ParsedToolCall] = []

    init(stops: [String], onEvent: @escaping @Sendable (EngineEvent) -> Void) {
        stopMatcher = StreamingStopMatcher(stops: stops)
        self.onEvent = onEvent
    }

    var isStopped: Bool { stopMatcher.isStopped }

    mutating func take(_ events: [StructuredAssistantEvent]) {
        for event in events {
            switch event {
            case .content(let text):
                let visible = stopMatcher.push(text)
                if !visible.isEmpty {
                    content += visible
                    onEvent(.content(visible))
                }
            case .toolCall(let call):
                toolCalls.append(call)
                onEvent(.toolCall(call))
            }
        }
    }

    /// Releases text the stop matcher was holding back as a possible prefix.
    mutating func finish() {
        let tail = stopMatcher.finish()
        if !tail.isEmpty {
            content += tail
            onEvent(.content(tail))
        }
    }
}

/// Set from a cancellation handler on another thread; read by llama.cpp's
/// abort callback between graph nodes.
final class LlamaAbortFlag: Sendable {
    private let flag = Atomic<Bool>(false)
    var isSet: Bool { flag.load(ordering: .acquiring) }
    func set() { flag.store(true, ordering: .releasing) }
    func clear() { flag.store(false, ordering: .releasing) }
}

fileprivate extension Duration {
    var seconds: Double {
        let (seconds, attoseconds) = components
        return Double(seconds) + Double(attoseconds) / 1e18
    }
}

import Foundation
import LlamaCpp
import Synchronization

/// Failures of the llama.cpp embedder.
public enum LlamaEmbedderError: Error, Equatable, Sendable, CustomStringConvertible {
    case modelLoadFailed(String)
    case contextCreationFailed(String)
    /// The GGUF does not pool its output into one vector per text.
    case notAnEmbeddingModel(String)
    /// `llama_decode` (or `llama_encode`) failed with this status.
    case decodeFailed(Int32)
    /// `unload()` was called.
    case unloaded
    /// No media model was given, or it could not be loaded.
    case mediaUnavailable(String)
    /// The image or audio could not be encoded.
    case mediaFailed(String)

    public var description: String {
        switch self {
        case .modelLoadFailed(let path): "llama.cpp could not load the embedding model \(path)"
        case .contextCreationFailed(let path): "llama.cpp could not create an embedding context for \(path)"
        case .notAnEmbeddingModel(let path): "\(path) is not a pooled embedding model"
        case .decodeFailed(let status): "the embedding pass failed with status \(status)"
        case .unloaded: "the embedding model was unloaded"
        case .mediaUnavailable(let detail): "no media encoder: \(detail)"
        case .mediaFailed(let detail): "the media encoder failed: \(detail)"
        }
    }
}

/// One image or one stretch of audio for the media encoders.
public enum LlamaMediaInput: Sendable, Hashable {
    /// Packed 8-bit RGB, `width × height × 3` bytes, already at the size
    /// the encoder should see (`LlamaEmbedder.imageSize`).
    case image(width: Int, height: Int, rgb: [UInt8])
    /// Mono samples at `LlamaEmbedder.audioSampleRate`, at most
    /// `LlamaEmbedder.maxAudioSeconds` long (longer audio is cut).
    case audio(samples: [Float])
}

/// Text embeddings from a pooled GGUF embedding model (EmbeddingGemma 2)
/// through the vendored llama.cpp, linked in-process.
///
/// This is deliberately not an `EngineHost` model: it is a second, small
/// llama.cpp model and context (the text tower of EmbeddingGemma 2 is about
/// 310 MB at Q8_0) that lives beside the one resident chat model, so search
/// never evicts the model a chat is using. The weights are mapped
/// (`use_mmap`), never read into Swift memory. The model has no KV cache
/// (non-causal attention), so each call packs several texts into one batch
/// of separate sequences and reads one pooled vector per sequence.
///
/// With `mediaModelPath` (EmbeddingGemma 2's `mmproj` GGUF, about 555 MB
/// at Q8_0) it also embeds images and audio into the same space: llama.cpp's
/// multimodal library (mtmd) encodes them, and its rows go through the same
/// text model as `<bos> <|image> rows <image|> <eos>` (or the audio markers),
/// as the reference implementation frames them. The media encoder loads on
/// the first media call, so a text-only caller never pays for it.
///
/// Calls are synchronous and serialised by a lock; run them off the main
/// thread. Vectors come back as the model writes them, not normalised:
/// callers truncate (Matryoshka) and L2-normalise themselves.
public final class LlamaEmbedder: Sendable {
    /// Sample rate media audio must have (EmbeddingGemma 2's audio encoder).
    public static let audioSampleRate = 16_000
    /// Audio past this many seconds of one input is cut, as the reference
    /// model does.
    public static let maxAudioSeconds = 30
    /// Image token budgets the vision encoder was trained with.
    public static let imageTokenBudgets = [70, 140, 280, 560, 1_120]

    /// Tokens one text may use; longer texts are cut to this length.
    public let maxSequenceTokens: Int
    /// Texts packed into one batch at most.
    public let maxSequences: Int
    /// Length of each vector (`n_embd_out`).
    public let dimensions: Int
    public let modelPath: URL
    /// The media encoder GGUF (`mmproj`), if images and audio are wanted.
    public let mediaModelPath: URL?

    private struct Handles: @unchecked Sendable {
        var model: OpaquePointer?
        var context: OpaquePointer?
        var batch: llama_batch
        var batchCapacity: Int32
        /// mtmd context, loaded on the first media call.
        var media: OpaquePointer?
        /// Mixed token and embedding-row batch for media inputs.
        var mediaBatch: OpaquePointer?
        var useGPU: Bool
    }

    private let handles: Mutex<Handles>

    /// Loads `modelPath` and creates its context. Throws when the file is not
    /// a pooled embedding model.
    public init(modelPath: URL, mediaModelPath: URL? = nil, maxSequenceTokens: Int = 2_048,
                maxSequences: Int = 16) throws {
        precondition(maxSequenceTokens > 0 && maxSequences > 0)
        LlamaRuntime.initialize()
        self.modelPath = modelPath
        self.mediaModelPath = mediaModelPath
        self.maxSequenceTokens = maxSequenceTokens
        self.maxSequences = maxSequences

        var modelParams = llama_model_default_params()
        let gpuOverride = ProcessInfo.processInfo.environment["MFERENCE_LLAMA_GPU_LAYERS"].flatMap(Int32.init)
        modelParams.n_gpu_layers = LlamaBackend.gpuLayerCount(
            supportsFastPaths: LlamaBackend.metalSupportsFastPaths(), override: gpuOverride)
        let useGPU = modelParams.n_gpu_layers != 0
        modelParams.load_mode = LLAMA_LOAD_MODE_MMAP
        guard let model = llama_model_load_from_file(modelPath.path, modelParams) else {
            throw LlamaEmbedderError.modelLoadFailed(modelPath.path)
        }

        var contextParams = llama_context_default_params()
        // Non-causal attention needs a whole sequence in one micro-batch, so
        // the logical and physical batch sizes are the same.
        let batchTokens = UInt32(maxSequenceTokens)
        contextParams.n_ctx = batchTokens
        contextParams.n_batch = batchTokens
        contextParams.n_ubatch = batchTokens
        contextParams.n_seq_max = UInt32(maxSequences)
        // Any one input may use the whole context (an image of 1,120 tokens
        // or a long passage), not n_ctx / n_seq_max.
        contextParams.kv_unified = true
        contextParams.embeddings = true
        contextParams.pooling_type = LLAMA_POOLING_TYPE_UNSPECIFIED
        contextParams.no_perf = true
        guard let context = llama_init_from_model(model, contextParams) else {
            llama_model_free(model)
            throw LlamaEmbedderError.contextCreationFailed(modelPath.path)
        }
        guard llama_pooling_type(context) != LLAMA_POOLING_TYPE_NONE else {
            llama_free(context)
            llama_model_free(model)
            throw LlamaEmbedderError.notAnEmbeddingModel(modelPath.path)
        }
        dimensions = Int(llama_model_n_embd_out(model))
        let batch = llama_batch_init(Int32(batchTokens), 0, Int32(maxSequences))
        handles = Mutex(Handles(model: model, context: context, batch: batch, batchCapacity: Int32(batchTokens),
                                media: nil, mediaBatch: nil, useGPU: useGPU))
    }

    deinit {
        unload()
    }

    /// Frees the context, the batch and the model. Later calls throw
    /// `.unloaded`.
    public func unload() {
        handles.withLock { handles in
            if handles.context != nil || handles.model != nil {
                llama_batch_free(handles.batch)
            }
            if let mediaBatch = handles.mediaBatch { llama_batch_ext_free(mediaBatch) }
            if let media = handles.media { mtmd_free(media) }
            handles.mediaBatch = nil
            handles.media = nil
            if let context = handles.context { llama_free(context) }
            if let model = handles.model { llama_model_free(model) }
            handles.context = nil
            handles.model = nil
        }
    }

    /// One vector per text, in order. Each text is tokenized with the
    /// model's special tokens (BOS/EOS as the GGUF asks) and cut to
    /// `maxSequenceTokens`. An empty array gives an empty array.
    public func embed(_ texts: [String]) throws -> [[Float]] {
        guard !texts.isEmpty else { return [] }
        return try handles.withLock { handles -> [[Float]] in
            guard let model = handles.model, let context = handles.context else {
                throw LlamaEmbedderError.unloaded
            }
            let vocab = llama_model_get_vocab(model)
            let tokenized = texts.map { Self.tokenize($0, vocab: vocab, limit: maxSequenceTokens) }
            let groups = Self.pack(tokenCounts: tokenized.map(\.count),
                                   batchTokens: Int(handles.batchCapacity), maxSequences: maxSequences)
            var vectors = [[Float]](repeating: [], count: texts.count)
            let encoder = llama_model_has_encoder(model)
            for group in groups {
                var batch = handles.batch
                batch.n_tokens = 0
                for (sequence, textIndex) in group.enumerated() {
                    for (position, token) in tokenized[textIndex].enumerated() {
                        let slot = Int(batch.n_tokens)
                        batch.token[slot] = token
                        batch.pos[slot] = llama_pos(position)
                        batch.n_seq_id[slot] = 1
                        batch.seq_id[slot]![0] = llama_seq_id(sequence)
                        batch.logits[slot] = 1
                        batch.n_tokens += 1
                    }
                }
                if let memory = llama_get_memory(context) { llama_memory_clear(memory, true) }
                let status = encoder ? llama_encode(context, batch) : llama_decode(context, batch)
                guard status == 0 else { throw LlamaEmbedderError.decodeFailed(status) }
                for (sequence, textIndex) in group.enumerated() {
                    guard let pointer = llama_get_embeddings_seq(context, llama_seq_id(sequence)) else {
                        throw LlamaEmbedderError.decodeFailed(-1)
                    }
                    vectors[textIndex] = Array(UnsafeBufferPointer(start: pointer, count: dimensions))
                }
            }
            return vectors
        }
    }

    // MARK: Media

    /// Whether this embedder was given a media model.
    public var supportsMedia: Bool { mediaModelPath != nil }

    /// The vector of one image or stretch of audio (not normalised, like
    /// `embed`). Loads the media encoder on first use.
    public func embed(media input: LlamaMediaInput) throws -> [Float] {
        let vectors = try embed(media: [input])
        guard let vector = vectors.first ?? nil else { throw LlamaEmbedderError.mediaFailed("the input was not embedded") }
        return vector
    }

    /// Vectors for several images and stretches of audio, in order; nil for
    /// an input that could not be embedded (its neighbours still are).
    ///
    /// Images of the same size go through the vision encoder together (one
    /// GPU pass per group, up to `mediaBatchTokens` tokens), and the encoded
    /// inputs go through the text model as separate sequences of one batch.
    /// The vectors equal one-at-a-time embedding (the batch only shares the
    /// passes). Throws only when the model is gone or the media encoder
    /// cannot load.
    public func embed(media inputs: [LlamaMediaInput]) throws -> [[Float]?] {
        guard !inputs.isEmpty else { return [] }
        return try handles.withLock { handles -> [[Float]?] in
            guard let model = handles.model, let context = handles.context else {
                throw LlamaEmbedderError.unloaded
            }
            let media = try Self.mediaContext(&handles, model: model, context: context, path: mediaModelPath)
            guard let batch = handles.mediaBatch else { throw LlamaEmbedderError.unloaded }
            let rowSize = Int(llama_model_n_embd_inp(model))

            // 1. Tokenize each input: text tokens around one media chunk.
            struct Prepared {
                var chunks: OpaquePointer
                var parts: [Part] = []
                var media: OpaquePointer?
                var mediaTokens = 0
                var rows: [Float]?
                var sizeKey: String
            }
            enum Part { case tokens([llama_token]); case media }
            var prepared = [Prepared?](repeating: nil, count: inputs.count)
            defer { for item in prepared.compactMap({ $0 }) { mtmd_input_chunks_free(item.chunks) } }
            for (index, input) in inputs.enumerated() {
                guard let (chunks, key) = Self.tokenize(input, media: media) else { continue }
                var item = Prepared(chunks: chunks, sizeKey: key)
                for chunkIndex in 0..<mtmd_input_chunks_size(chunks) {
                    let chunk = mtmd_input_chunks_get(chunks, chunkIndex)
                    if mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_TEXT {
                        var count = 0
                        guard let tokens = mtmd_input_chunk_get_tokens_text(chunk, &count) else { continue }
                        item.parts.append(.tokens(Array(UnsafeBufferPointer(start: tokens, count: count))))
                    } else if item.media == nil {
                        item.media = chunk
                        item.mediaTokens = Int(mtmd_input_chunk_get_n_tokens(chunk))
                        item.parts.append(.media)
                    }
                }
                let total = item.parts.reduce(0) { total, part in
                    if case .tokens(let tokens) = part { return total + tokens.count }
                    return total + item.mediaTokens
                }
                guard item.media != nil, total < Int(handles.batchCapacity) else { continue }
                prepared[index] = item
            }

            // 2. Encode media, same-size images together.
            func copyRows(_ pointer: UnsafePointer<Float>?, count: Int) -> [Float]? {
                pointer.map { Array(UnsafeBufferPointer(start: $0, count: count * rowSize)) }
            }
            func encodeAlone(_ index: Int) {
                guard let item = prepared[index], let chunk = item.media else { return }
                guard mtmd_encode_chunk(media, chunk) == 0 else { return }
                prepared[index]?.rows = copyRows(mtmd_get_output_embd(media), count: item.mediaTokens)
            }
            let order = prepared.indices.filter { prepared[$0] != nil }
                .sorted { (prepared[$0]!.sizeKey, $0) < (prepared[$1]!.sizeKey, $1) }
            var group: [Int] = []
            func flush() {
                defer { group = [] }
                guard group.count > 1, let encoder = mtmd_batch_init(media) else {
                    for index in group { encodeAlone(index) }
                    return
                }
                defer { mtmd_batch_free(encoder) }
                for index in group { _ = mtmd_batch_add_chunk(encoder, prepared[index]!.media) }
                guard mtmd_batch_encode(encoder) == 0 else {
                    for index in group { encodeAlone(index) }
                    return
                }
                for index in group {
                    let item = prepared[index]!
                    prepared[index]?.rows = copyRows(mtmd_batch_get_output_embd(encoder, item.media),
                                                     count: item.mediaTokens)
                }
            }
            var groupTokens = 0
            for index in order {
                let item = prepared[index]!
                let joins = !group.isEmpty && prepared[group[0]]!.sizeKey == item.sizeKey && !item.sizeKey.isEmpty
                    && groupTokens + item.mediaTokens <= Self.mediaBatchTokens
                if !joins { flush(); groupTokens = 0 }
                group.append(index)
                groupTokens += item.mediaTokens
            }
            flush()

            // 3. The text model over the encoded inputs, several per batch.
            var vectors = [[Float]?](repeating: nil, count: inputs.count)
            let ready = prepared.indices.filter { prepared[$0]?.rows != nil }
            let lengths = ready.map { index -> Int in
                prepared[index]!.parts.reduce(0) { total, part in
                    if case .tokens(let tokens) = part { return total + tokens.count }
                    return total + prepared[index]!.mediaTokens
                }
            }
            for packed in Self.pack(tokenCounts: lengths, batchTokens: Int(handles.batchCapacity),
                                    maxSequences: maxSequences) {
                llama_batch_ext_clear(batch)
                var placed = true
                for (sequence, slot) in packed.enumerated() {
                    let item = prepared[ready[slot]]!
                    var position: llama_pos = 0
                    func place(_ index: Int32) -> Bool {
                        guard index >= 0 else { return false }
                        var pos: [llama_pos] = [position, 0, 0, 0]
                        guard llama_batch_ext_set_pos(batch, index, &pos),
                              llama_batch_ext_set_output_embd(batch, index, true) else { return false }
                        position += 1
                        return true
                    }
                    for part in item.parts where placed {
                        switch part {
                        case .tokens(let tokens):
                            for token in tokens where placed {
                                placed = place(llama_batch_ext_add_token(batch, llama_seq_id(sequence), token))
                            }
                        case .media:
                            item.rows!.withUnsafeBufferPointer { rows in
                                for row in 0..<item.mediaTokens where placed {
                                    let embd = llama_embd(data: rows.baseAddress! + row * rowSize, n_rows: 1,
                                                          n_embd: rowSize)
                                    placed = place(llama_batch_ext_add_embd(batch, llama_seq_id(sequence), embd))
                                }
                            }
                        }
                    }
                }
                guard placed else { continue }
                if let memory = llama_get_memory(context) { llama_memory_clear(memory, true) }
                let processed = llama_process(context, LLAMA_PROCESS_TYPE_ENCODE, batch)
                llama_batch_ext_clear(batch)
                guard processed == 0 else { continue }
                for (sequence, slot) in packed.enumerated() {
                    guard let pointer = llama_get_embeddings_seq(context, llama_seq_id(sequence)) else { continue }
                    vectors[ready[slot]] = Array(UnsafeBufferPointer(start: pointer, count: dimensions))
                }
            }
            return vectors
        }
    }

    /// Vision tokens encoded in one pass at most (mtmd's own default).
    public static let mediaBatchTokens = 1_024

    /// Chunks for one input (`<bos> <|image> … <image|> <eos>`), and a key
    /// that is equal for images the vision encoder can batch (same size);
    /// empty for audio. Nil when the input cannot be read.
    private static func tokenize(_ input: LlamaMediaInput, media: OpaquePointer) -> (OpaquePointer, String)? {
        let bitmap: OpaquePointer?
        let key: String
        switch input {
        case .image(let width, let height, let rgb):
            guard width > 0, height > 0, rgb.count == width * height * 3, mtmd_support_vision(media) else { return nil }
            bitmap = rgb.withUnsafeBufferPointer { mtmd_bitmap_init(UInt32(width), UInt32(height), $0.baseAddress) }
            key = "\(width)x\(height)"
        case .audio(let samples):
            guard !samples.isEmpty, mtmd_support_audio(media) else { return nil }
            let limit = maxAudioSeconds * Int(mtmd_get_audio_sample_rate(media))
            let kept = samples.count > limit ? Array(samples.prefix(limit)) : samples
            bitmap = kept.withUnsafeBufferPointer { mtmd_bitmap_init_from_audio($0.count, $0.baseAddress) }
            key = ""
        }
        guard let bitmap else { return nil }
        defer { mtmd_bitmap_free(bitmap) }
        guard let chunks = mtmd_input_chunks_init() else { return nil }
        let marker = mtmd_default_marker()!
        var text = mtmd_input_text(text: marker, text_len: strlen(marker), add_special: true, parse_special: true)
        let bitmaps: [OpaquePointer?] = [bitmap]
        let status = bitmaps.withUnsafeBufferPointer { mtmd_tokenize(media, chunks, &text, $0.baseAddress, 1) }
        guard status == 0 else {
            mtmd_input_chunks_free(chunks)
            return nil
        }
        return (chunks, key)
    }

    /// The loaded media context, loading it on first use.
    private static func mediaContext(_ handles: inout Handles, model: OpaquePointer, context: OpaquePointer,
                                     path: URL?) throws -> OpaquePointer {
        if let media = handles.media { return media }
        guard let path else { throw LlamaEmbedderError.mediaUnavailable("no media model was given") }
        var params = mtmd_context_params_default()
        params.use_gpu = handles.useGPU
        params.print_timings = false
        params.warmup = true
        // Images arrive at their final size (`imageSize`); mtmd must not
        // rescale them.
        params.image_min_tokens = 1
        params.image_max_tokens = Int32(imageTokenBudgets.last!)
        guard let media = mtmd_init_from_file(path.path, model, params) else {
            throw LlamaEmbedderError.mediaUnavailable("could not load \(path.path)")
        }
        guard let batch = llama_batch_ext_init(context) else {
            mtmd_free(media)
            throw LlamaEmbedderError.mediaUnavailable("could not create a batch")
        }
        handles.media = media
        handles.mediaBatch = batch
        return media
    }

    /// The size an image should be handed over at for `tokens` vision
    /// tokens (one of `imageTokenBudgets`): scaled to fill the budget with
    /// its aspect ratio kept, each side a multiple of 48 px (3 × 3 patches of
    /// 16 px), as the reference image processor sizes it. Nil for an image
    /// too thin to embed.
    public static func imageSize(width: Int, height: Int, tokens: Int) -> (width: Int, height: Int)? {
        guard width > 0, height > 0, tokens > 0 else { return nil }
        let patch = 16, pooling = 3
        let maxPatches = tokens * pooling * pooling
        let factor = (Double(maxPatches * patch * patch) / Double(height * width)).squareRoot()
        let side = pooling * patch
        var targetHeight = Int((factor * Double(height) / Double(side)).rounded(.down)) * side
        var targetWidth = Int((factor * Double(width) / Double(side)).rounded(.down)) * side
        if targetHeight == 0 && targetWidth == 0 { return nil }
        let maxSide = (maxPatches / (pooling * pooling)) * side
        if targetHeight == 0 {
            targetHeight = side
            targetWidth = min(width / height * side, maxSide)
        } else if targetWidth == 0 {
            targetWidth = side
            targetHeight = min(height / width * side, maxSide)
        }
        return (targetWidth, targetHeight)
    }

    /// Tokens for `text` with the model's special tokens, at most `limit`.
    /// An empty text still gets the special tokens (or one token), so every
    /// sequence has something to pool.
    static func tokenize(_ text: String, vocab: OpaquePointer?, limit: Int) -> [llama_token] {
        let utf8 = Array(text.utf8)
        var capacity = utf8.count + 8
        var tokens: [llama_token] = []
        while true {
            var buffer = [llama_token](repeating: 0, count: capacity)
            let count = utf8.withUnsafeBufferPointer { bytes in
                bytes.withMemoryRebound(to: CChar.self) { chars in
                    llama_tokenize(vocab, chars.baseAddress, Int32(chars.count), &buffer, Int32(capacity), true, false)
                }
            }
            if count >= 0 {
                tokens = Array(buffer.prefix(Int(count)))
                break
            }
            capacity = Int(-count)
        }
        if tokens.isEmpty { tokens = [llama_vocab_bos(vocab) >= 0 ? llama_vocab_bos(vocab) : 0] }
        guard tokens.count > limit else { return tokens }
        // Keep the closing special token, if the model appends one.
        let eos = llama_vocab_eos(vocab)
        if llama_vocab_get_add_eos(vocab), eos >= 0 {
            return Array(tokens.prefix(limit - 1)) + [eos]
        }
        return Array(tokens.prefix(limit))
    }

    /// Groups texts (by index, in order) so that each group fits one batch:
    /// at most `batchTokens` tokens and `maxSequences` texts. A text longer
    /// than `batchTokens` cannot occur (texts are cut to fit).
    public static func pack(tokenCounts: [Int], batchTokens: Int, maxSequences: Int) -> [[Int]] {
        var groups: [[Int]] = []
        var current: [Int] = []
        var used = 0
        for (index, count) in tokenCounts.enumerated() {
            if !current.isEmpty, used + count > batchTokens || current.count == maxSequences {
                groups.append(current)
                current = []
                used = 0
            }
            current.append(index)
            used += count
        }
        if !current.isEmpty { groups.append(current) }
        return groups
    }
}

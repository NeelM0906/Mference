import Foundation
import LlamaCpp

/// The layout of a prompt that carries images, as `mtmd_tokenize` split it:
/// text runs and images in order. Pure data, so prefix reuse and positions
/// are tested without a model.
///
/// Positions follow llama.cpp's `mtmd-helper` at the pin: a text token takes
/// one position, and an image takes `positions` of them (`max(h, w)` of its
/// token grid under M-RoPE) however many tokens it has, so after an image the
/// position runs ahead of the token count.
struct LlamaMediaLayout: Equatable {
    enum Segment: Equatable {
        case text([Int32])
        /// An image of `tokens` vision tokens spanning `positions` positions.
        case media(tokens: Int, positions: Int, hash: String)

        var positions: Int {
            switch self {
            case .text(let tokens): tokens.count
            case .media(_, let positions, _): positions
            }
        }
    }

    /// Work left after reusing a prefix: a text run from `skipping` tokens
    /// in, or a whole image, each starting at `position`.
    enum Piece: Equatable {
        case text(segment: Int, skipping: Int, position: Int)
        case media(segment: Int, position: Int)
    }

    let segments: [Segment]

    /// The prompt in context cells: text tokens, and
    /// `EngineMedia.placeholderToken` for each vision token.
    var promptTokens: [Int32] {
        segments.flatMap { segment -> [Int32] in
            switch segment {
            case .text(let tokens): tokens
            case .media(let tokens, _, _): Array(repeating: EngineMedia.placeholderToken, count: tokens)
            }
        }
    }

    /// One prompt-cache entry per position: the token for text, and for an
    /// image a run of negative keys derived from its hash, so the prompt cache
    /// (whose index is the position) reuses an image only whole and only when
    /// it is the same image.
    var cacheKeys: [Int32] {
        segments.flatMap { segment -> [Int32] in
            switch segment {
            case .text(let tokens): tokens
            case .media(_, let positions, let hash): Self.mediaKeys(hash: hash, positions: positions)
            }
        }
    }

    var totalPositions: Int { segments.reduce(0) { $0 + $1.positions } }

    /// Negative keys for an image's positions. The first comes from the
    /// hash's leading 31 bits, so two different images differ at once.
    static func mediaKeys(hash: String, positions: Int) -> [Int32] {
        let base = UInt32(hash.prefix(8), radix: 16) ?? 0
        return (0..<positions).map { offset in
            Int32(bitPattern: (base &+ UInt32(truncatingIfNeeded: offset) &* 0x9E37_79B1) | 0x8000_0000)
        }
    }

    /// What is left to decode when the memory already holds the first
    /// `reused` positions, and how many prompt tokens that reuse covers. Nil
    /// when `reused` ends inside an image, which cannot be resumed.
    func remaining(afterReusing reused: Int) -> (pieces: [Piece], reusedTokens: Int)? {
        var pieces: [Piece] = []
        var position = 0
        var reusedTokens = 0
        for (index, segment) in segments.enumerated() {
            let end = position + segment.positions
            switch segment {
            case .text(let tokens):
                let skip = max(0, min(tokens.count, reused - position))
                reusedTokens += skip
                if skip < tokens.count {
                    pieces.append(.text(segment: index, skipping: skip, position: position + skip))
                }
            case .media(let tokens, _, _):
                if end <= reused {
                    reusedTokens += tokens
                } else if position < reused {
                    return nil
                } else {
                    pieces.append(.media(segment: index, position: position))
                }
            }
            position = end
        }
        return (pieces, reusedTokens)
    }
}

/// A prompt tokenized by `mtmd_tokenize`: its layout and the chunks behind
/// it (an image chunk holds the preprocessed pixels the encoder reads).
/// Freed with this object; used only inside its `LlamaModel`.
final class LlamaMediaPrompt {
    let rendered: String
    let mediaHashes: [String]
    let layout: LlamaMediaLayout
    private let chunks: OpaquePointer

    /// Tokenizes `rendered`, in which each of `media` appears as the
    /// context's marker, in order. Throws `EngineError.invalidRequest` when
    /// the markers and images disagree (the text itself contains the marker)
    /// or an image cannot be preprocessed.
    init(rendered: String, media: [EngineMedia], context: OpaquePointer) throws {
        guard let chunks = mtmd_input_chunks_init() else { throw LlamaBackendError.projectorFailed("no chunks") }
        var bitmaps: [OpaquePointer?] = []
        defer { for bitmap in bitmaps { mtmd_bitmap_free(bitmap) } }
        for image in media {
            let bitmap = image.rgb.withUnsafeBytes { bytes in
                mtmd_bitmap_init(UInt32(image.width), UInt32(image.height),
                                 bytes.bindMemory(to: UInt8.self).baseAddress)
            }
            guard let bitmap else {
                mtmd_input_chunks_free(chunks)
                throw LlamaBackendError.projectorFailed("could not hold an image")
            }
            mtmd_bitmap_set_id(bitmap, image.hash)
            bitmaps.append(bitmap)
        }
        let status = rendered.withCString { text in
            var input = mtmd_input_text(text: text, text_len: strlen(text), add_special: true, parse_special: true)
            return bitmaps.withUnsafeBufferPointer { mtmd_tokenize(context, chunks, &input, $0.baseAddress, $0.count) }
        }
        guard status == 0 else {
            mtmd_input_chunks_free(chunks)
            throw EngineError.invalidRequest(
                message: status == 1 ? "the images do not match the image positions in the prompt"
                    : "an image could not be read",
                param: "messages", code: "unsupported_content")
        }
        var segments: [LlamaMediaLayout.Segment] = []
        var mediaIndex = 0
        for index in 0..<mtmd_input_chunks_size(chunks) {
            let chunk = mtmd_input_chunks_get(chunks, index)
            if mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_TEXT {
                var count = 0
                let tokens = mtmd_input_chunk_get_tokens_text(chunk, &count)
                segments.append(.text(tokens.map { Array(UnsafeBufferPointer(start: $0, count: count)) } ?? []))
            } else {
                let hash = mediaIndex < media.count ? media[mediaIndex].hash : ""
                mediaIndex += 1
                segments.append(.media(tokens: Int(mtmd_input_chunk_get_n_tokens(chunk)),
                                       positions: Int(mtmd_input_chunk_get_n_pos(chunk)), hash: hash))
            }
        }
        self.rendered = rendered
        self.mediaHashes = media.map(\.hash)
        self.layout = LlamaMediaLayout(segments: segments)
        self.chunks = chunks
    }

    func chunk(_ segment: Int) -> OpaquePointer? {
        mtmd_input_chunks_get(chunks, segment)
    }

    deinit { mtmd_input_chunks_free(chunks) }
}

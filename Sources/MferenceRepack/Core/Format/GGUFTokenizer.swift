import Foundation

/// Builds the tokenizer sidecars Mference loads (`tokenizer.json`,
/// `tokenizer_config.json`, `chat_template.jinja`) from a GGUF's
/// `tokenizer.*` metadata, so a GGUF install needs no vendor files.
///
/// The synthesized `tokenizer.json` reproduces the vendor's pipeline for the
/// pre-tokenizer it names (`preTokenizers`): vocabulary, merges, special
/// tokens and the EOS / padding ids all come from the GGUF, the split regex
/// and stage flags from the table. llama.cpp applies no Unicode normalizer
/// to a gpt2 BPE vocabulary and the vendor's tokenizer has none, so neither
/// does this one. Parity with `llama-tokenize` is checked in the fidelity gate.
enum GGUFTokenizer {
    /// `llama_token_type` values (llama.cpp `src/llama-vocab.h`).
    enum TokenType: Int64 {
        case undefined = 0, normal = 1, unknown = 2, control = 3, userDefined = 4, unused = 5, byte = 6
    }

    /// How a `tokenizer.ggml.pre` value maps to a Hugging Face fast
    /// tokenizer: the split regex in the HF form (llama.cpp quotes it as the
    /// "original regex from tokenizer.json" beside its own rewrite), BPE
    /// `ignore_merges`, and the three ByteLevel stages' flags.
    struct PreTokenizerSpec: Sendable {
        let regex: String
        let ignoreMerges: Bool
        let preTokenizerByteLevel: [String: Bool]
        let postProcessorByteLevel: [String: Bool]
        let decoderByteLevel: [String: Bool]
    }

    /// Supported pre-tokenizers. `inkling2` is llama.cpp's GPT-4o split
    /// (`LLAMA_VOCAB_PRE_TYPE_GPT4O`) with `ignore_merges` on (#32799, which
    /// makes llama.cpp match the vendor's tokenizer). The values here are the
    /// vendor's own `tokenizer.json` of `thinkingmachines/Inkling-2-Mini` @
    /// `b85c3cf`: no normalizer, this split, ByteLevel, `ignore_merges: true`.
    static let preTokenizers: [String: PreTokenizerSpec] = [
        "inkling2": PreTokenizerSpec(
            regex: #"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"#,
            ignoreMerges: true,
            preTokenizerByteLevel: ["add_prefix_space": false, "trim_offsets": true, "use_regex": false],
            postProcessorByteLevel: ["add_prefix_space": true, "trim_offsets": false, "use_regex": true],
            decoderByteLevel: ["add_prefix_space": true, "trim_offsets": true, "use_regex": true]),
    ]

    struct Sidecars: Sendable {
        let tokenizerJSON: Data
        let tokenizerConfigJSON: Data
        let chatTemplate: String
    }

    static func synthesize(metadata md: [String: GGUF.Value], path: String) throws -> Sidecars {
        func invalid(_ d: String) -> RepackError { .ggufInvalid(path: path, detail: "tokenizer: \(d)") }
        func strings(_ key: String) throws -> [String] {
            guard case .array(let a)? = md[key] else { throw invalid("missing \(key)") }
            return try a.map { v in
                guard let s = v.string else { throw invalid("\(key) holds a non-string") }
                return s
            }
        }
        func id(_ key: String) throws -> Int? {
            guard let v = md[key] else { return nil }
            guard let n = v.uint64 else { throw invalid("\(key) is not an unsigned integer") }
            return Int(n)
        }

        guard md["tokenizer.ggml.model"]?.string == "gpt2" else {
            throw invalid("model \(md["tokenizer.ggml.model"]?.string ?? "nil") is not gpt2 BPE")
        }
        let pre = md["tokenizer.ggml.pre"]?.string ?? ""
        guard let spec = preTokenizers[pre] else { throw invalid("unsupported pre-tokenizer '\(pre)'") }
        let tokens = try strings("tokenizer.ggml.tokens")
        guard case .array(let typeValues)? = md["tokenizer.ggml.token_type"], typeValues.count == tokens.count else {
            throw invalid("token_type missing or not one per token")
        }
        let types: [TokenType] = try typeValues.map { v in
            let raw: Int64
            switch v {
            case .i32(let x): raw = Int64(x)
            case .u32(let x): raw = Int64(x)
            default: throw invalid("token_type holds a non-integer")
            }
            guard let t = TokenType(rawValue: raw) else { throw invalid("token_type \(raw)") }
            return t
        }
        let merges: [[String]] = try strings("tokenizer.ggml.merges").map { m in
            let parts = m.split(separator: " ", omittingEmptySubsequences: false)
            guard parts.count == 2, !parts[0].isEmpty, !parts[1].isEmpty else { throw invalid("merge '\(m)'") }
            return [String(parts[0]), String(parts[1])]
        }
        guard md["tokenizer.ggml.add_bos_token"].map({ $0 == .bool(false) }) ?? true else {
            throw invalid("add_bos_token=true is not supported")
        }
        let bos = try id("tokenizer.ggml.bos_token_id")
        let eos = try id("tokenizer.ggml.eos_token_id")
        let pad = try id("tokenizer.ggml.padding_token_id")
        guard let template = md["tokenizer.chat_template"]?.string else { throw invalid("missing chat_template") }
        for i in [bos, eos, pad].compactMap({ $0 }) where i >= tokens.count {
            throw invalid("special id \(i) out of range")
        }

        // Normal, unused and byte tokens go in the BPE vocab; control and
        // user-defined tokens are added tokens, matched before BPE (llama.cpp
        // `parse_special`), with control tokens marked special.
        var vocab: [(String, Int)] = []
        var added: [[String: Any]] = []
        var seen = Set<String>()
        for (i, (tok, t)) in zip(tokens, types).enumerated() {
            guard seen.insert(tok).inserted else { throw invalid("duplicate token text at id \(i)") }
            switch t {
            case .normal, .unused, .byte:
                vocab.append((tok, i))
            case .control, .userDefined:
                added.append(["id": i, "content": tok, "single_word": false, "lstrip": false, "rstrip": false,
                              "normalized": false, "special": t == .control])
            case .unknown, .undefined:
                throw invalid("token type \(t) at id \(i) has no gpt2 BPE equivalent")
            }
        }

        func byteLevel(_ flags: [String: Bool]) -> [String: Any] {
            var stage: [String: Any] = ["type": "ByteLevel"]
            for (key, value) in flags { stage[key] = value }
            return stage
        }
        // JSONSerialization does not keep dictionary order; vocab order is by id
        // for readability only (ids are explicit).
        let tokenizer: [String: Any] = [
            "version": "1.0",
            "truncation": NSNull(),
            "padding": NSNull(),
            "added_tokens": added,
            "normalizer": NSNull(),
            "pre_tokenizer": ["type": "Sequence", "pretokenizers": [
                ["type": "Split", "pattern": ["Regex": spec.regex], "behavior": "Isolated", "invert": false],
                byteLevel(spec.preTokenizerByteLevel),
            ]],
            "post_processor": byteLevel(spec.postProcessorByteLevel),
            "decoder": byteLevel(spec.decoderByteLevel),
            "model": [
                "type": "BPE", "dropout": NSNull(), "unk_token": NSNull(),
                "continuing_subword_prefix": NSNull(), "end_of_word_suffix": NSNull(),
                "fuse_unk": false, "byte_fallback": false, "ignore_merges": spec.ignoreMerges,
                "vocab": Dictionary(uniqueKeysWithValues: vocab),
                "merges": merges,
            ] as [String: Any],
        ]
        var config: [String: Any] = [
            "add_bos_token": false,
            "add_prefix_space": false,
            "clean_up_tokenization_spaces": false,
            "errors": "replace",
            "split_special_tokens": false,
            "tokenizer_class": "PreTrainedTokenizerFast",
            "model_max_length": Int(md["\(md["general.architecture"]?.string ?? "").context_length"]?.uint64 ?? 262_144),
        ]
        config["bos_token"] = bos.map { tokens[$0] } ?? NSNull()
        config["eos_token"] = eos.map { tokens[$0] } ?? NSNull()
        config["pad_token"] = pad.map { tokens[$0] } ?? NSNull()
        config["unk_token"] = NSNull()

        let opts: JSONSerialization.WritingOptions = [.sortedKeys, .withoutEscapingSlashes]
        return Sidecars(tokenizerJSON: try JSONSerialization.data(withJSONObject: tokenizer, options: opts),
                        tokenizerConfigJSON: try JSONSerialization.data(withJSONObject: config, options: opts),
                        chatTemplate: template)
    }
}

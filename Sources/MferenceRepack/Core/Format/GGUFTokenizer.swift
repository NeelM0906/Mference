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
        /// `generation_config.json` from the GGUF's `general.sampling.*`
        /// defaults and its EOS / padding ids; nil when the file carries no
        /// sampling defaults.
        let generationConfigJSON: Data?
    }

    /// The sampling defaults a GGUF records, as `generation_config.json`
    /// writes them: `do_sample` and `temperature` always, `top_k`, `top_p`
    /// and `min_p` only when they restrict sampling (the vendor file of
    /// Inkling 2 Mini sets none of them), and the EOS and padding ids.
    static func generationConfig(metadata md: [String: GGUF.Value], eos: Int?, pad: Int?) throws -> Data? {
        func float(_ key: String) -> Double? {
            switch md[key] {
            case .f32(let f)?: return Double(f)
            case .f64(let d)?: return d
            default: return nil
            }
        }
        guard let temperature = float("general.sampling.temp") else { return nil }
        var config: [String: Any] = ["do_sample": temperature > 0, "temperature": temperature]
        if let topK = md["general.sampling.top_k"]?.uint64.flatMap({ Int(exactly: $0) }), topK > 0 {
            config["top_k"] = topK
        }
        if let topP = float("general.sampling.top_p"), topP < 1 { config["top_p"] = topP }
        if let minP = float("general.sampling.min_p"), minP > 0 { config["min_p"] = minP }
        if let eos { config["eos_token_id"] = eos }
        if let pad { config["pad_token_id"] = pad }
        return try JSONSerialization.data(withJSONObject: config,
                                          options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes])
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
            guard let u = v.uint64, let n = Int(exactly: u) else {
                throw invalid("\(key) is not an unsigned integer")
            }
            return n
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
        var added: [OrderedJSON] = []
        var seen = Set<String>()
        for (i, (tok, t)) in zip(tokens, types).enumerated() {
            guard seen.insert(tok).inserted else { throw invalid("duplicate token text at id \(i)") }
            switch t {
            case .normal, .unused, .byte:
                vocab.append((tok, i))
            case .control, .userDefined:
                added.append(.sorted(["id": .int(i), "content": .string(tok), "single_word": .bool(false),
                                      "lstrip": .bool(false), "rstrip": .bool(false), "normalized": .bool(false),
                                      "special": .bool(t == .control)]))
            case .unknown, .undefined:
                throw invalid("token type \(t) at id \(i) has no gpt2 BPE equivalent")
            }
        }

        func byteLevel(_ flags: [String: Bool]) -> OrderedJSON {
            var stage: [String: OrderedJSON] = ["type": .string("ByteLevel")]
            for (key, value) in flags { stage[key] = .bool(value) }
            return .sorted(stage)
        }
        // Written with OrderedJSON, not JSONSerialization: see its comment.
        // Members sorted by key; the vocab in id order, as Hugging Face
        // writes it.
        let tokenizer: OrderedJSON = .sorted([
            "version": .string("1.0"),
            "truncation": .null,
            "padding": .null,
            "added_tokens": .array(added),
            "normalizer": .null,
            "pre_tokenizer": .sorted(["type": .string("Sequence"), "pretokenizers": .array([
                .sorted(["type": .string("Split"), "pattern": .sorted(["Regex": .string(spec.regex)]),
                         "behavior": .string("Isolated"), "invert": .bool(false)]),
                byteLevel(spec.preTokenizerByteLevel),
            ])]),
            "post_processor": byteLevel(spec.postProcessorByteLevel),
            "decoder": byteLevel(spec.decoderByteLevel),
            "model": .sorted([
                "type": .string("BPE"), "dropout": .null, "unk_token": .null,
                "continuing_subword_prefix": .null, "end_of_word_suffix": .null,
                "fuse_unk": .bool(false), "byte_fallback": .bool(false), "ignore_merges": .bool(spec.ignoreMerges),
                "vocab": .object(vocab.map { ($0.0, .int($0.1)) }),
                "merges": .array(merges.map { .array($0.map { .string($0) }) }),
            ]),
        ])
        var config: [String: Any] = [
            "add_bos_token": false,
            "add_prefix_space": false,
            "clean_up_tokenization_spaces": false,
            "errors": "replace",
            "split_special_tokens": false,
            "tokenizer_class": "PreTrainedTokenizerFast",
            "model_max_length": md["\(md["general.architecture"]?.string ?? "").context_length"]?.uint64
                .flatMap { Int(exactly: $0) } ?? 262_144,
        ]
        config["bos_token"] = bos.map { tokens[$0] } ?? NSNull()
        config["eos_token"] = eos.map { tokens[$0] } ?? NSNull()
        config["pad_token"] = pad.map { tokens[$0] } ?? NSNull()
        config["unk_token"] = NSNull()

        let opts: JSONSerialization.WritingOptions = [.sortedKeys, .withoutEscapingSlashes]
        return Sidecars(tokenizerJSON: Data(tokenizer.bytes),
                        tokenizerConfigJSON: try JSONSerialization.data(withJSONObject: config, options: opts),
                        chatTemplate: template,
                        generationConfigJSON: try generationConfig(metadata: md, eos: eos, pad: pad))
    }
}

extension GGUFTokenizer {
    /// A JSON value written in a fixed member order, so one GGUF always gives
    /// the same `tokenizer.json` bytes. `JSONSerialization`'s `.sortedKeys`
    /// compares keys numerically and case-insensitively, which is not a total
    /// order over a 200K-entry byte-level vocabulary: its output followed the
    /// dictionary's per-process hash order, and two installs of the same file
    /// wrote the same members in different orders.
    indirect enum OrderedJSON {
        case null
        case bool(Bool)
        case int(Int)
        case string(String)
        case array([OrderedJSON])
        /// Members in the given order.
        case object([(String, OrderedJSON)])

        /// An object with its members sorted by the keys' UTF-8 bytes.
        static func sorted(_ members: [String: OrderedJSON]) -> OrderedJSON {
            .object(members.sorted { $0.key.utf8.lexicographicallyPrecedes($1.key.utf8) }.map { ($0.key, $0.value) })
        }

        /// Compact UTF-8 JSON: no whitespace, `/` unescaped, and only `"`,
        /// `\` and U+0000-U+001F escaped, as `JSONSerialization` writes it
        /// with `.withoutEscapingSlashes`.
        var bytes: [UInt8] {
            var out: [UInt8] = []
            write(to: &out)
            return out
        }

        private func write(to out: inout [UInt8]) {
            switch self {
            case .null: out.append(contentsOf: "null".utf8)
            case .bool(let b): out.append(contentsOf: (b ? "true" : "false").utf8)
            case .int(let n): out.append(contentsOf: String(n).utf8)
            case .string(let s): Self.writeString(s, to: &out)
            case .array(let items):
                out.append(UInt8(ascii: "["))
                for (i, item) in items.enumerated() {
                    if i > 0 { out.append(UInt8(ascii: ",")) }
                    item.write(to: &out)
                }
                out.append(UInt8(ascii: "]"))
            case .object(let members):
                out.append(UInt8(ascii: "{"))
                for (i, (key, value)) in members.enumerated() {
                    if i > 0 { out.append(UInt8(ascii: ",")) }
                    Self.writeString(key, to: &out)
                    out.append(UInt8(ascii: ":"))
                    value.write(to: &out)
                }
                out.append(UInt8(ascii: "}"))
            }
        }

        private static let hex = Array("0123456789abcdef".utf8)

        private static func writeString(_ s: String, to out: inout [UInt8]) {
            out.append(UInt8(ascii: "\""))
            for byte in s.utf8 {
                switch byte {
                case UInt8(ascii: "\""): out.append(contentsOf: [UInt8(ascii: "\\"), UInt8(ascii: "\"")])
                case UInt8(ascii: "\\"): out.append(contentsOf: [UInt8(ascii: "\\"), UInt8(ascii: "\\")])
                case 0x08: out.append(contentsOf: Array(#"\b"#.utf8))
                case 0x09: out.append(contentsOf: Array(#"\t"#.utf8))
                case 0x0A: out.append(contentsOf: Array(#"\n"#.utf8))
                case 0x0C: out.append(contentsOf: Array(#"\f"#.utf8))
                case 0x0D: out.append(contentsOf: Array(#"\r"#.utf8))
                case 0x00..<0x20:
                    out.append(contentsOf: Array(#"\u00"#.utf8))
                    out.append(hex[Int(byte >> 4)])
                    out.append(hex[Int(byte & 0xF)])
                default: out.append(byte)
                }
            }
            out.append(UInt8(ascii: "\""))
        }
    }
}

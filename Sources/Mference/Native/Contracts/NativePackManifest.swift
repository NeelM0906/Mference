extension Native {
    /// `manifest.json` of an `.mfpack` v1 install directory (D3).
    ///
    /// The JSON encoding of this type (default `Codable` keys, as declared) is
    /// the on-disk contract. `MferenceRepackCore` cannot import `Mference`, so
    /// the production writer (W1) keeps a mirror of these types and a test
    /// pins the two encodings to each other. Writers encode with
    /// `JSONEncoder.OutputFormatting` `[.sortedKeys, .prettyPrinted]`, so the
    /// bytes are deterministic.
    ///
    /// Directory layout:
    /// - `manifest.json` (this type)
    /// - `weights.bin`: every tensor's packed bytes; each tensor starts at a
    ///   4096-byte-aligned `offset` (zero padding between tensors)
    /// - `tokenizer.json`: `TokenizerData` (tokens, merges, types, ids)
    /// - `chat_template.jinja`: the GGUF `tokenizer.chat_template` bytes, verbatim
    /// - `verified-install.json`: written last by the writer after every hash
    ///   check passed; a pack without it is incomplete
    public struct PackManifest: Codable, Equatable, Sendable {
        /// Always `"mfpack"`.
        public var format: String
        /// Pack format version; this contract defines version 1.
        public var formatVersion: Int
        /// llama.cpp architecture name, e.g. `"qwen35"`.
        public var architecture: String
        /// Product label, e.g. `"quick"`; the directory is `<label>.mfpack`.
        public var label: String
        public var source: Source
        /// Architecture hyperparameters, decoded from GGUF metadata.
        public var config: Qwen35Config
        /// `PackLayoutTable.version` the writer used.
        public var layoutTableVersion: Int
        /// Every GGUF tensor, in GGUF tensor-info order.
        public var tensors: [Tensor]
        /// GGUF metadata needed at runtime that is not in `config`, as strings
        /// (key → value), e.g. `general.name`, `tokenizer.ggml.pre`.
        public var metadata: [String: String]

        public init(format: String = "mfpack", formatVersion: Int = 1, architecture: String,
                    label: String, source: Source, config: Qwen35Config,
                    layoutTableVersion: Int, tensors: [Tensor], metadata: [String: String]) {
            self.format = format
            self.formatVersion = formatVersion
            self.architecture = architecture
            self.label = label
            self.source = source
            self.config = config
            self.layoutTableVersion = layoutTableVersion
            self.tensors = tensors
            self.metadata = metadata
        }

        /// The source GGUF the pack was made from.
        public struct Source: Codable, Equatable, Sendable {
            public var fileName: String
            public var byteCount: UInt64
            /// SHA-256 (lowercase hex) of the whole GGUF file.
            public var sha256: String
            /// GGUF version (2 or 3).
            public var ggufVersion: Int
            public init(fileName: String, byteCount: UInt64, sha256: String, ggufVersion: Int) {
                self.fileName = fileName
                self.byteCount = byteCount
                self.sha256 = sha256
                self.ggufVersion = ggufVersion
            }
        }

        public struct Tensor: Codable, Equatable, Sendable {
            /// GGUF tensor name, e.g. `blk.3.attn_q.weight`.
            public var name: String
            public var type: GGMLType
            /// ggml `ne` order: `ne[0]` is the contiguous (input) dimension.
            public var shape: [Int]
            public var layout: TileLayout
            /// Byte offset in `weights.bin` (multiple of 4096).
            public var offset: UInt64
            /// Packed byte count; equals the GGUF byte count (no padding).
            public var byteCount: UInt64
            /// SHA-256 of the tensor's bytes in GGUF order (the source bytes).
            public var sourceSHA256: String
            /// SHA-256 of the tensor's bytes as stored in `weights.bin`.
            public var packedSHA256: String
            public init(name: String, type: GGMLType, shape: [Int], layout: TileLayout,
                        offset: UInt64, byteCount: UInt64, sourceSHA256: String, packedSHA256: String) {
                self.name = name
                self.type = type
                self.shape = shape
                self.layout = layout
                self.offset = offset
                self.byteCount = byteCount
                self.sourceSHA256 = sourceSHA256
                self.packedSHA256 = packedSHA256
            }
        }
    }

    /// `tokenizer.json` of an `.mfpack`: the GGUF `tokenizer.ggml.*` arrays
    /// and ids, verbatim.
    public struct TokenizerData: Codable, Equatable, Sendable {
        /// `tokenizer.ggml.model`, e.g. `"gpt2"`.
        public var model: String
        /// `tokenizer.ggml.pre`, e.g. `"qwen35"`.
        public var pre: String
        public var tokens: [String]
        public var tokenTypes: [Int32]
        public var merges: [String]
        public var bosTokenID: Int32?
        public var eosTokenID: Int32?
        public var paddingTokenID: Int32?
        /// `tokenizer.ggml.add_bos_token` (absent → llama.cpp's default for the model).
        public var addBOS: Bool?
        public var addEOS: Bool?
        /// Every other `tokenizer.ggml.<name>_token_id` key in the GGUF, keyed
        /// by `<name>` exactly as spelled in the key (for example `eot`, `eom`,
        /// `unknown`, `seperator` (llama.cpp's spelling), `cls`, `mask`,
        /// `fim_pre`, `fim_suf`, `fim_mid`, `fim_pad`, `fim_rep`, `fim_sep`).
        /// llama.cpp uses these for end-of-generation and special-token
        /// handling. nil in packs written before this field existed.
        public var specialTokenIDs: [String: Int32]?
        public init(model: String, pre: String, tokens: [String], tokenTypes: [Int32],
                    merges: [String], bosTokenID: Int32?, eosTokenID: Int32?,
                    paddingTokenID: Int32?, addBOS: Bool?, addEOS: Bool?,
                    specialTokenIDs: [String: Int32]? = nil) {
            self.model = model
            self.pre = pre
            self.tokens = tokens
            self.tokenTypes = tokenTypes
            self.merges = merges
            self.bosTokenID = bosTokenID
            self.eosTokenID = eosTokenID
            self.paddingTokenID = paddingTokenID
            self.addBOS = addBOS
            self.addEOS = addEOS
            self.specialTokenIDs = specialTokenIDs
        }
    }
}

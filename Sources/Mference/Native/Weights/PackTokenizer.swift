import Darwin
import Foundation

extension Native {
    /// The production `TextTokenizer`: llama.cpp's BPE tokenizer at the pin
    /// (`src/llama-vocab.cpp`) for `tokenizer.ggml.model == "gpt2"` with the
    /// `qwen35` pre-tokenizer, built for speed and byte-identical to
    /// `llama_tokenize` / `llama_token_to_piece`.
    ///
    /// - Special tokens: llama.cpp's attribute rules (token types, the
    ///   text-based EOT/FIM/EOG detection), the special-token cache sorted by
    ///   text length (ties in id order), and `tokenizer_st_partition`.
    /// - Pre-tokenizer: `unicode_regex_split_custom_qwen35` (`PackUnicode`).
    /// - BPE: llama.cpp's bigram queue (lowest rank first, ties to the
    ///   leftmost pair). Every token text and merge operand is interned once,
    ///   so a merge is an integer-pair lookup and no symbol text is rebuilt.
    /// - Pieces: `token_to_piece` (control text only with `special`, GPT-2
    ///   byte decoding of normal tokens, `<0xNN>` byte tokens).
    ///
    /// Text is handled as UTF-8 bytes throughout. Immutable after `init`, so
    /// one instance can serve concurrent callers.
    public final class PackTokenizer: TextTokenizer, @unchecked Sendable {
        public struct Attr: OptionSet, Sendable {
            public let rawValue: Int
            public init(rawValue: Int) { self.rawValue = rawValue }
            public static let unknown = Attr(rawValue: 1 << 0)
            public static let unused = Attr(rawValue: 1 << 1)
            public static let normal = Attr(rawValue: 1 << 2)
            public static let control = Attr(rawValue: 1 << 3)
            public static let userDefined = Attr(rawValue: 1 << 4)
            public static let byte = Attr(rawValue: 1 << 5)
            public static let normalized = Attr(rawValue: 1 << 6)
            public static let lstrip = Attr(rawValue: 1 << 7)
            public static let rstrip = Attr(rawValue: 1 << 8)
            public static let singleWord = Attr(rawValue: 1 << 9)
        }

        public var vocabSize: Int { tokenCount }
        public let bosTokenID: Int32?
        public let eosTokenID: Int32?
        public private(set) var endOfGenerationTokenIDs: Set<Int32> = []
        /// Special-token cache in llama.cpp order (longest text first).
        public private(set) var specialTokens: [Int32] = []
        private var attrs: [Attr]
        private let addBOS: Bool
        private let addEOS: Bool
        private let tokenCount: Int

        /// Interned byte strings: ids `0 ..< tokenCount` are the vocabulary
        /// texts in id order (a duplicate text keeps its first intern id), then
        /// merge operands and merge results.
        private let strings: PackByteStrings
        /// Intern id → token id (the last token with that text, as llama.cpp's
        /// `token_to_id` keeps it), or -1.
        private let internToken: [Int32]
        /// Token id → intern id of its text.
        private let tokenIntern: [Int32]
        /// (left intern << 32 | right intern) → merge rank (the first rank of
        /// a repeated pair, as `bpe_ranks.emplace` keeps it).
        private let ranks: [UInt64: Int32]
        /// Merge rank → intern id of `left + right`.
        private let mergeResult: [Int32]
        /// Raw byte → intern id of its GPT-2 byte-encoded text.
        private let byteIntern: [Int32]
        /// Encoded byte → token id of the one-byte text, or -1.
        private let singleByteToken: [Int32]

        public init(data: TokenizerData) throws {
            guard data.model == "gpt2" else {
                throw NativeError.unsupported("tokenizer model \(data.model)")
            }
            guard data.pre == "qwen35" else {
                throw NativeError.unsupported("pre-tokenizer \(data.pre) (the pack tokenizer implements qwen35)")
            }
            guard data.tokens.count == data.tokenTypes.count || data.tokenTypes.isEmpty else {
                throw NativeError.invalidShape("token types \(data.tokenTypes.count) vs tokens \(data.tokens.count)")
            }
            guard !data.tokens.isEmpty, data.tokens.count < Int(Int32.max) else {
                throw NativeError.invalidShape("vocabulary of \(data.tokens.count) tokens")
            }
            let n = data.tokens.count
            tokenCount = n
            var strings = PackByteStrings(expected: n + data.merges.count)
            var internToken = [Int32](repeating: -1, count: n)
            var tokenIntern = [Int32](repeating: -1, count: n)
            var attrs: [Attr] = []
            attrs.reserveCapacity(n)
            for (i, token) in data.tokens.enumerated() {
                var text = token
                // llama.cpp gives an empty token text a placeholder.
                if text.isEmpty { text = "[EMPTY_\(i)]" }
                let id = text.utf8.withContiguousStorageIfAvailable { strings.intern(UnsafeRawBufferPointer($0)) }
                    ?? Array(text.utf8).withUnsafeBytes { strings.intern($0) }
                tokenIntern[i] = id
                if Int(id) >= internToken.count { internToken.append(contentsOf: repeatElement(-1, count: Int(id) + 1 - internToken.count)) }
                internToken[Int(id)] = Int32(i)
                let type = data.tokenTypes.isEmpty ? 1 : data.tokenTypes[i]
                switch type {
                case 1: attrs.append(.normal)
                case 2: attrs.append(.unknown)
                case 3: attrs.append(.control)
                case 4: attrs.append(.userDefined)
                case 5: attrs.append(.unused)
                case 6: attrs.append(.byte)
                default: attrs.append([])
                }
            }
            var byteIntern = [Int32](repeating: -1, count: 256)
            for b in 0..<256 {
                byteIntern[b] = PackUnicode.byteToUTF8[b].withUnsafeBytes { strings.intern($0) }
            }
            var ranks: [UInt64: Int32] = [:]
            ranks.reserveCapacity(data.merges.count)
            var mergeResult = [Int32](repeating: -1, count: data.merges.count)
            var joined: [UInt8] = []
            for (rank, merge) in data.merges.enumerated() {
                let bytes = Array(merge.utf8)
                // llama.cpp splits at the first space after the first byte; a
                // merge without one is an empty pair that never matches.
                guard bytes.count > 1, let sep = bytes[1...].firstIndex(of: 0x20) else { continue }
                let l = bytes[..<sep].withUnsafeBytes { strings.intern($0) }
                let r = bytes[(sep + 1)...].withUnsafeBytes { strings.intern($0) }
                let key = UInt64(UInt32(bitPattern: l)) << 32 | UInt64(UInt32(bitPattern: r))
                guard ranks[key] == nil else { continue }
                ranks[key] = Int32(rank)
                joined.removeAll(keepingCapacity: true)
                joined.append(contentsOf: bytes[..<sep])
                joined.append(contentsOf: bytes[(sep + 1)...])
                mergeResult[rank] = joined.withUnsafeBytes { strings.intern($0) }
            }
            if strings.count > internToken.count {
                internToken.append(contentsOf: repeatElement(-1, count: strings.count - internToken.count))
            }
            var singleByteToken = [Int32](repeating: -1, count: 256)
            for b in 0..<256 {
                let id = [UInt8(b)].withUnsafeBytes { strings.find($0) }
                if id >= 0 { singleByteToken[b] = internToken[Int(id)] }
            }
            self.strings = strings
            self.internToken = internToken
            self.tokenIntern = tokenIntern
            self.ranks = ranks
            self.mergeResult = mergeResult
            self.byteIntern = byteIntern
            self.singleByteToken = singleByteToken
            self.attrs = attrs
            let ids = Self.specialIDs(data)
            bosTokenID = ids.bos
            eosTokenID = ids.eos
            addBOS = data.addBOS ?? false
            addEOS = data.addEOS ?? false
            applySpecialTokenRules(data)
        }

        /// BOS and EOS as llama.cpp resolves them for a gpt2 vocabulary: both
        /// default to id 11; a GGUF id replaces the default when it is in
        /// range; a default out of range is disabled.
        public static func specialIDs(_ data: TokenizerData) -> (bos: Int32?, eos: Int32?) {
            let n = data.tokens.count
            func resolve(_ key: Int32?) -> Int32? {
                if let key, key >= 0, Int(key) < n { return key }
                return 11 < n ? 11 : nil
            }
            return (resolve(data.bosTokenID), resolve(data.eosTokenID))
        }

        public func attr(_ token: Int32) -> Attr { attrs[Int(token)] }

        /// The token's text (vocabulary bytes).
        public func text(_ token: Int32) -> [UInt8] { Array(strings.bytes(tokenIntern[Int(token)])) }

        private func tokenID(_ text: String) -> Int32? {
            let id = Array(text.utf8).withUnsafeBytes { strings.find($0) }
            guard id >= 0, Int(id) < internToken.count, internToken[Int(id)] >= 0 else { return nil }
            return internToken[Int(id)]
        }

        // MARK: llama.cpp special-token rules (llama_vocab::impl::load)

        private func applySpecialTokenRules(_ data: TokenizerData) {
            func promote(_ t: Int32) { attrs[Int(t)].insert(.control) }
            // llama.cpp takes the first candidate in hash-map order; with one
            // present candidate the order does not matter.
            func first(_ candidates: [String]) -> Int32? { candidates.lazy.compactMap { self.tokenID($0) }.first }
            let keyed = data.specialTokenIDs ?? [:]
            // A `tokenizer.ggml.<name>_token_id` id in range is used as is (the
            // deprecated prefix/suffix/middle keys come after the fim_* ones,
            // so they win); otherwise the token is found by text and promoted
            // to control.
            func fromKeys(_ names: [String]) -> Int32? {
                var found: Int32?
                for name in names {
                    if let v = keyed[name], v >= 0, Int(v) < tokenCount { found = v }
                }
                return found
            }
            func resolve(_ names: [String], _ candidates: [String]) -> Int32? {
                if let k = fromKeys(names) { return k }
                guard let t = first(candidates) else { return nil }
                promote(t)
                return t
            }
            let eot = resolve(["eot"], ["<|eot_id|>", "<|im_end|>", "<|end|>", "<end_of_turn>", "<|endoftext|>",
                                        "<|end_of_text|>", "<EOT>", "_<EOT>", "[EOT]", "<｜end▁of▁sentence｜>",
                                        "<end_of_utterance>"])
            let eom = resolve(["eom"], ["<|eom_id|>"])
            _ = resolve(["fim_pre", "prefix"], ["<|fim_prefix|>", "<fim-prefix>", "<fim_prefix>", "<｜fim▁begin｜>",
                                                "<PRE>", "▁<PRE>", "<|code_prefix|>", "<|prefix|>",
                                                "<|plamo:fim_prefix|>"])
            _ = resolve(["fim_suf", "suffix"], ["<|fim_suffix|>", "<fim-suffix>", "<fim_suffix>", "<｜fim▁hole｜>",
                                                "<SUF>", "▁<SUF>", "<|code_suffix|>", "<|suffix|>",
                                                "<|plamo:fim_suffix|>"])
            _ = resolve(["fim_mid", "middle"], ["<|fim_middle|>", "<fim-middle>", "<fim_middle>", "<｜fim▁end｜>",
                                                "<MID>", "▁<MID>", "<|code_middle|>", "<|middle|>",
                                                "<|plamo:fim_middle|>"])
            let fimPad = resolve(["fim_pad"], ["<|fim_pad|>", "<fim-pad>", "<fim_pad>", "<PAD>", "[PAD]"])
            let fimRep = resolve(["fim_rep"], ["<|fim_repo|>", "<|repo_name|>", "<fim-repo>", "<REPO>", "<reponame>"])
            let fimSep = resolve(["fim_sep"], ["<|file_sep|>", "<|plamo:file_separator|>"])
            // Control tokens whose text contains "unused" are also unused.
            let unused = Array("unused".utf8)
            for i in 0..<tokenCount where attrs[i].contains(.control) {
                if strings.bytes(tokenIntern[i]).withUnsafeBytes({ Self.find(unused, in: $0, from: 0) }) != nil {
                    attrs[i].insert(.unused)
                }
            }
            var eog: Set<Int32> = []
            for t in [fimPad, fimRep, fimSep].compactMap({ $0 }) { eog.insert(t) }
            let eogTexts = ["<|eot_id|>", "<|eot|>", "<|im_end|>", "<|end|>", "<|return|>", "<|call|>",
                            "<|flush|>", "<|calls|>", "<end_of_turn>", "<|endoftext|>", "</s>", "<|eom_id|>",
                            "<EOT>", "_<EOT>", "[EOT]", "[EOS]", "<|end_of_text|>", "<end_of_utterance>", "<eos>",
                            "<turn|>", "<|tool_response>", "<｜end▁of▁sentence｜>", "[e~[", "<|ifm|im_end|>"]
            for text in eogTexts {
                if let t = tokenID(text) {
                    eog.insert(t)
                    promote(t)
                }
            }
            for text in ["<|channel|>", "<|message|>", "<|start|>", "<|constrain|>"] {
                if let t = tokenID(text) { attrs[Int(t)] = .userDefined }
            }
            for t in [eosTokenID, eot, eom].compactMap({ $0 }) { eog.insert(t) }
            // o200k_harmony / solar-open: "<|end|>" is not EOG next to these.
            let eogTextSet = Set(eog.map { text($0) })
            let has = { (s: String) in eogTextSet.contains(Array(s.utf8)) }
            if let end = tokenID("<|end|>"), eog.contains(end),
               (has("<|return|>") && has("<|call|>")) || ((has("<|call|>") || has("<|calls|>")) && has("<|flush|>")) {
                eog.remove(end)
                attrs[Int(end)] = .userDefined
            }
            // gemma4 / plamo / beam: a normal "</s>".
            if let s = tokenID("</s>"), eog.contains(s),
               tokenID("<rfl:parameter name=\"") != nil || has("<|tool_response>") || has("<|plamo:eos|>") {
                eog.remove(s)
                attrs[Int(s)] = .normal
            }
            endOfGenerationTokenIDs = eog

            let special = (0..<Int32(tokenCount)).filter {
                !attrs[Int($0)].intersection([.control, .userDefined, .unknown]).isEmpty
            }
            // llama.cpp sorts by text length only; equal lengths keep id order.
            specialTokens = special.enumerated().sorted { a, b in
                let la = strings.length(tokenIntern[Int(a.element)])
                let lb = strings.length(tokenIntern[Int(b.element)])
                return la != lb ? la > lb : a.offset < b.offset
            }.map { $0.element }
        }

        // MARK: Encode

        public func encode(_ text: String, addSpecial: Bool, parseSpecial: Bool) throws -> [Int32] {
            var output: [Int32] = []
            if addSpecial && addBOS {
                guard let bos = bosTokenID else { throw NativeError.missing("BOS token") }
                output.append(bos)
            }
            let utf8 = Array(text.utf8)
            if !utf8.isEmpty {
                output.reserveCapacity(utf8.count / 3 + 8)
                utf8.withUnsafeBufferPointer { bytes in
                    let session = Session()
                    for fragment in partition(UnsafeRawBufferPointer(bytes), parseSpecial: parseSpecial) {
                        if fragment.token >= 0 {
                            output.append(fragment.token)
                        } else {
                            let raw = UnsafeBufferPointer(rebasing: bytes[fragment.start ..< fragment.start + fragment.count])
                            tokenize(raw, session: session, into: &output)
                        }
                    }
                }
            }
            if addSpecial && addEOS {
                guard let eos = eosTokenID else { throw NativeError.missing("EOS token") }
                output.append(eos)
            }
            return output
        }

        struct Fragment: Equatable {
            var start: Int
            var count: Int
            /// -1 for raw text.
            var token: Int32
        }

        /// `tokenizer_st_partition`.
        func partition(_ text: UnsafeRawBufferPointer, parseSpecial: Bool) -> [Fragment] {
            var buffer = [Fragment(start: 0, count: text.count, token: -1)]
            var next: [Fragment] = []
            for special in specialTokens {
                let attr = attrs[Int(special)]
                if !parseSpecial && !attr.intersection([.control, .unknown]).isEmpty { continue }
                let needle = strings.bytes(tokenIntern[Int(special)])
                next.removeAll(keepingCapacity: true)
                var changed = false
                needle.withUnsafeBytes { needle in
                    for fragment in buffer {
                        guard fragment.token < 0 else {
                            next.append(fragment)
                            continue
                        }
                        var offset = fragment.start, length = fragment.count
                        while true {
                            guard let match = Self.find(needle, in: text, from: offset, upTo: offset + length) else {
                                next.append(Fragment(start: offset, count: length, token: -1))
                                break
                            }
                            changed = true
                            if match > offset {
                                var leftLength = match - offset
                                if attr.contains(.lstrip) {
                                    while leftLength > 0 && Self.isSpace(text[offset + leftLength - 1]) { leftLength -= 1 }
                                }
                                if leftLength > 0 { next.append(Fragment(start: offset, count: leftLength, token: -1)) }
                            }
                            next.append(Fragment(start: match, count: needle.count, token: special))
                            if match + needle.count < offset + length {
                                var rightOffset = match + needle.count
                                var rightLength = length - ((match - offset) + needle.count)
                                if attr.contains(.rstrip) {
                                    while rightLength > 0 && Self.isSpace(text[rightOffset]) {
                                        rightOffset += 1
                                        rightLength -= 1
                                    }
                                }
                                if rightLength > 0 {
                                    offset = rightOffset
                                    length = rightLength
                                    continue
                                }
                            }
                            break
                        }
                    }
                }
                if changed { swap(&buffer, &next) }
            }
            return buffer
        }

        @inline(__always)
        static func isSpace(_ b: UInt8) -> Bool { b == 0x20 || (0x09...0x0D).contains(b) }

        /// First index `i` in `from ..< upTo` where `needle` occurs, ending by `upTo`.
        static func find(_ needle: UnsafeRawBufferPointer, in haystack: UnsafeRawBufferPointer,
                         from: Int, upTo: Int) -> Int? {
            guard needle.count > 0, upTo - from >= needle.count, let base = haystack.baseAddress else { return nil }
            guard let hit = memmem(base + from, upTo - from, needle.baseAddress!, needle.count) else { return nil }
            return base.distance(to: UnsafeRawPointer(hit))
        }

        static func find(_ needle: [UInt8], in haystack: UnsafeRawBufferPointer, from: Int) -> Int? {
            needle.withUnsafeBytes { find($0, in: haystack, from: from, upTo: haystack.count) }
        }

        // MARK: BPE

        /// Per-encode scratch, reused across words (one per `encode` call, so
        /// concurrent callers never share it).
        final class Session {
            var cps: [UInt32] = []
            var offsets: [Int32] = []
            var flags: [UInt8] = []
            var ends: [Int32] = []
            var symbols: [Symbol] = []
            var heap = BigramHeap()
        }

        struct Symbol {
            /// Raw byte offset in the word and byte count (0 once merged away).
            var start: Int32
            var n: Int32
            var prev: Int32
            var next: Int32
            var intern: Int32
        }

        struct Bigram {
            var rank: Int32
            var left: Int32
            var right: Int32
            var leftIntern: Int32
            var rightIntern: Int32
        }

        /// One raw fragment: pre-tokenize, then BPE each word.
        func tokenize(_ raw: UnsafeBufferPointer<UInt8>, session s: Session, into output: inout [Int32]) {
            PackUnicode.decode(raw, cps: &s.cps, offsets: &s.offsets)
            s.flags.removeAll(keepingCapacity: true)
            for cp in s.cps { s.flags.append(PackUnicode.flags(cp)) }
            var ends = s.ends
            s.ends = []
            ends.removeAll(keepingCapacity: true)
            s.cps.withUnsafeBufferPointer { cps in
                s.flags.withUnsafeBufferPointer { flags in
                    PackUnicode.splitQwen35(cps, flags, into: &ends)
                }
            }
            var startCP = 0
            for e in ends {
                let endCP = Int(e)
                let from = Int(s.offsets[startCP]), to = Int(s.offsets[endCP])
                merge(UnsafeBufferPointer(rebasing: raw[from ..< to]), session: s, into: &output)
                startCP = endCP
            }
            s.ends = ends
        }

        @inline(__always)
        private func rank(_ l: Int32, _ r: Int32) -> Int32? {
            ranks[UInt64(UInt32(bitPattern: l)) << 32 | UInt64(UInt32(bitPattern: r))]
        }

        /// BPE of one word given as raw bytes; each raw byte is one GPT-2
        /// byte-encoded character, so the initial symbols are its bytes.
        func merge(_ word: UnsafeBufferPointer<UInt8>, session s: Session, into output: inout [Int32]) {
            let n = word.count
            guard n > 0 else { return }
            s.symbols.removeAll(keepingCapacity: true)
            for i in 0..<n {
                s.symbols.append(Symbol(start: Int32(i), n: 1, prev: Int32(i - 1), next: i + 1 == n ? -1 : Int32(i + 1),
                                        intern: byteIntern[Int(word[i])]))
            }
            if n > 1 {
                s.heap.removeAll()
                func addBigram(_ left: Int32, _ right: Int32) {
                    guard left != -1, right != -1 else { return }
                    let l = s.symbols[Int(left)].intern, r = s.symbols[Int(right)].intern
                    guard let rank = rank(l, r) else { return }
                    s.heap.push(Bigram(rank: rank, left: left, right: right, leftIntern: l, rightIntern: r))
                }
                for i in 1..<n { addBigram(Int32(i - 1), Int32(i)) }
                while let bigram = s.heap.pop() {
                    let li = Int(bigram.left), ri = Int(bigram.right)
                    let l = s.symbols[li], r = s.symbols[ri]
                    if l.n == 0 || r.n == 0 { continue }
                    // Stale entry: either side changed since it was queued.
                    if l.intern != bigram.leftIntern || r.intern != bigram.rightIntern { continue }
                    s.symbols[li].n += r.n
                    s.symbols[li].intern = mergeResult[Int(bigram.rank)]
                    s.symbols[ri].n = 0
                    s.symbols[li].next = r.next
                    if r.next >= 0 { s.symbols[Int(r.next)].prev = bigram.left }
                    addBigram(s.symbols[li].prev, bigram.left)
                    addBigram(bigram.left, s.symbols[li].next)
                }
            }
            var i: Int32 = 0
            while i != -1 {
                let sym = s.symbols[Int(i)]
                let token = internToken[Int(sym.intern)]
                if token >= 0 {
                    output.append(token)
                } else {
                    // Not a vocabulary token: one token per encoded byte that is one.
                    for k in Int(sym.start) ..< Int(sym.start + sym.n) {
                        for e in PackUnicode.byteToUTF8[Int(word[k])] where singleByteToken[Int(e)] >= 0 {
                            output.append(singleByteToken[Int(e)])
                        }
                    }
                }
                i = sym.next
            }
        }

        /// Min-heap on (rank, left): llama.cpp's `llm_bigram_bpe::comparator`.
        struct BigramHeap {
            var heap: [Bigram] = []
            mutating func removeAll() { heap.removeAll(keepingCapacity: true) }
            @inline(__always)
            static func before(_ a: Bigram, _ b: Bigram) -> Bool {
                a.rank != b.rank ? a.rank < b.rank : a.left < b.left
            }
            mutating func push(_ b: Bigram) {
                heap.append(b)
                var i = heap.count - 1
                while i > 0 {
                    let p = (i - 1) / 2
                    guard Self.before(heap[i], heap[p]) else { break }
                    heap.swapAt(i, p)
                    i = p
                }
            }
            mutating func pop() -> Bigram? {
                guard !heap.isEmpty else { return nil }
                let top = heap[0]
                let last = heap.removeLast()
                if !heap.isEmpty {
                    heap[0] = last
                    var i = 0
                    while true {
                        let l = 2 * i + 1, r = l + 1
                        var m = i
                        if l < heap.count && Self.before(heap[l], heap[m]) { m = l }
                        if r < heap.count && Self.before(heap[r], heap[m]) { m = r }
                        if m == i { break }
                        heap.swapAt(i, m)
                        i = m
                    }
                }
                return top
            }
        }

        // MARK: Pieces

        public func piece(_ token: Int32, special: Bool) -> [UInt8] {
            guard token >= 0, Int(token) < tokenCount else { return [] }
            let attr = attrs[Int(token)]
            if !special && !attr.intersection([.unknown, .control]).isEmpty { return [] }
            let text = strings.bytes(tokenIntern[Int(token)])
            if !attr.intersection([.unknown, .control, .userDefined]).isEmpty { return Array(text) }
            if attr.contains(.normal) { return Self.decodeByteText(text) }
            if attr.contains(.byte) {
                // `token_to_byte`: (char) strtol(text.substr(3, 2), NULL, 16);
                // a text under 3 bytes gives no bytes.
                guard text.count >= 3 else { return [] }
                let digits = Array(text[(text.startIndex + 3) ..< min(text.startIndex + 5, text.endIndex)])
                return [UInt8(truncatingIfNeeded: Self.strtol16(digits))]
            }
            return []
        }

        /// `llama_decode_text`: GPT-2 byte decoding; a code point outside the
        /// byte map prints `[UNK_BYTE_0x<its UTF-8 hex>` + the token text + `]`.
        static func decodeByteText(_ text: ArraySlice<UInt8>) -> [UInt8] {
            var out: [UInt8] = []
            out.reserveCapacity(text.count)
            var i = text.startIndex
            while i < text.endIndex {
                let b0 = text[i]
                let len = b0 < 0x80 ? 1 : b0 < 0xE0 ? 2 : b0 < 0xF0 ? 3 : 4
                let end = min(i + len, text.endIndex)
                var cp: UInt32
                switch len {
                case 1: cp = UInt32(b0)
                case 2: cp = UInt32(b0 & 0x1F)
                case 3: cp = UInt32(b0 & 0x0F)
                default: cp = UInt32(b0 & 0x07)
                }
                var k = i + 1
                while k < end {
                    cp = cp << 6 | UInt32(text[k] & 0x3F)
                    k += 1
                }
                if cp < 0x144, PackUnicode.codePointToByte[Int(cp)] != 0 {
                    out.append(UInt8(PackUnicode.codePointToByte[Int(cp)] - 1))
                } else {
                    out += Array("[UNK_BYTE_0x".utf8)
                    let hex = Array("0123456789abcdef".utf8)
                    for b in text[i ..< end] {
                        out.append(hex[Int(b >> 4)])
                        out.append(hex[Int(b & 0x0F)])
                    }
                    out += text
                    out.append(0x5D)
                }
                i = end
            }
            return out
        }

        /// C `strtol(s, NULL, 16)`: leading isspace, an optional sign, an
        /// optional 0x/0X prefix (only when a hex digit follows), then hex
        /// digits; 0 when none.
        static func strtol16(_ s: [UInt8]) -> Int {
            var i = 0
            while i < s.count, s[i] == 0x20 || (0x09...0x0D).contains(s[i]) { i += 1 }
            var negative = false
            if i < s.count, s[i] == 0x2B || s[i] == 0x2D {
                negative = s[i] == 0x2D
                i += 1
            }
            func digit(_ b: UInt8) -> Int? {
                switch b {
                case 0x30...0x39: return Int(b - 0x30)
                case 0x41...0x46: return Int(b - 0x41 + 10)
                case 0x61...0x66: return Int(b - 0x61 + 10)
                default: return nil
                }
            }
            if i + 2 < s.count, s[i] == 0x30, s[i + 1] == 0x78 || s[i + 1] == 0x58, digit(s[i + 2]) != nil {
                i += 2
            }
            var value = 0
            while i < s.count, let d = digit(s[i]) {
                value = value * 16 + d
                i += 1
            }
            return negative ? -value : value
        }
    }

    /// Interned byte strings in one flat buffer, with an open-addressing
    /// (linear probing, FNV-1a) index so lookups never allocate.
    struct PackByteStrings {
        private var storage: [UInt8] = []
        /// String `i` is `storage[offsets[i] ..< offsets[i + 1]]`.
        private var offsets: [Int] = [0]
        private var slots: [Int32]
        private var mask: Int

        init(expected: Int) {
            var capacity = 16
            while capacity < expected * 2 { capacity <<= 1 }
            slots = [Int32](repeating: -1, count: capacity)
            mask = capacity - 1
            storage.reserveCapacity(expected * 8)
            offsets.reserveCapacity(expected + 1)
        }

        var count: Int { offsets.count - 1 }

        func bytes(_ id: Int32) -> ArraySlice<UInt8> { storage[offsets[Int(id)] ..< offsets[Int(id) + 1]] }

        func length(_ id: Int32) -> Int { offsets[Int(id) + 1] - offsets[Int(id)] }

        @inline(__always)
        static func hash(_ bytes: UnsafeRawBufferPointer) -> Int {
            var h: UInt64 = 0xcbf2_9ce4_8422_2325
            for b in bytes {
                h ^= UInt64(b)
                h &*= 0x0000_0100_0000_01b3
            }
            return Int(truncatingIfNeeded: h ^ (h >> 29))
        }

        private func equal(_ id: Int32, _ bytes: UnsafeRawBufferPointer) -> Bool {
            let start = offsets[Int(id)], end = offsets[Int(id) + 1]
            guard end - start == bytes.count else { return false }
            if bytes.isEmpty { return true }
            return storage.withUnsafeBytes { memcmp($0.baseAddress! + start, bytes.baseAddress!, bytes.count) == 0 }
        }

        /// The intern id of `bytes`, or -1.
        func find(_ bytes: UnsafeRawBufferPointer) -> Int32 {
            var slot = Self.hash(bytes) & mask
            while true {
                let id = slots[slot]
                if id < 0 { return -1 }
                if equal(id, bytes) { return id }
                slot = (slot + 1) & mask
            }
        }

        /// The intern id of `bytes`, adding it if new.
        mutating func intern(_ bytes: UnsafeRawBufferPointer) -> Int32 {
            if (count + 1) * 2 > slots.count { grow() }
            var slot = Self.hash(bytes) & mask
            while true {
                let id = slots[slot]
                if id < 0 { break }
                if equal(id, bytes) { return id }
                slot = (slot + 1) & mask
            }
            let id = Int32(count)
            storage.append(contentsOf: bytes)
            offsets.append(storage.count)
            slots[slot] = id
            return id
        }

        private mutating func grow() {
            let capacity = slots.count * 2
            var newSlots = [Int32](repeating: -1, count: capacity)
            let newMask = capacity - 1
            storage.withUnsafeBytes { all in
                for id in 0..<count {
                    let s = offsets[id], e = offsets[id + 1]
                    var slot = Self.hash(UnsafeRawBufferPointer(rebasing: all[s ..< e])) & newMask
                    while newSlots[slot] >= 0 { slot = (slot + 1) & newMask }
                    newSlots[slot] = Int32(id)
                }
            }
            slots = newSlots
            mask = newMask
        }
    }

    /// `TokenizerSource` over a pack's `tokenizer.json`; builds the tokenizer
    /// once and hands out that instance.
    public final class PackTokenizerSource: TokenizerSource, @unchecked Sendable {
        public let data: TokenizerData
        private let lock = NSLock()
        private var built: PackTokenizer?

        public init(data: TokenizerData) { self.data = data }

        public func makeTokenizer() throws -> any TextTokenizer { try packTokenizer() }

        public func packTokenizer() throws -> PackTokenizer {
            lock.lock()
            defer { lock.unlock() }
            if let built { return built }
            let t = try PackTokenizer(data: data)
            built = t
            return t
        }
    }

    /// `ChatTemplateSource` over a pack's `chat_template.jinja`: the GGUF
    /// `tokenizer.chat_template` text, byte for byte, and the BOS / EOS token
    /// texts as llama.cpp resolves the ids (`common_chat_templates_init`).
    public final class PackChatTemplateSource: ChatTemplateSource {
        public let template: String
        public let bosToken: String?
        public let eosToken: String?

        public init(template: String, data: TokenizerData) {
            self.template = template
            let ids = PackTokenizer.specialIDs(data)
            func text(_ id: Int32?) -> String? { id.map { data.tokens[Int($0)] } }
            bosToken = text(ids.bos)
            eosToken = text(ids.eos)
        }
    }
}

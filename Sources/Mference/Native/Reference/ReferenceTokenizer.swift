import Foundation

extension Native {
    /// The reference `TextTokenizer`: a direct port of llama.cpp's BPE
    /// tokenizer at the pin (`src/llama-vocab.cpp`, `src/unicode.cpp`) for
    /// `tokenizer.ggml.model == "gpt2"` with the `qwen35` pre-tokenizer.
    ///
    /// - Special tokens: llama.cpp's attribute rules (token types, the
    ///   text-based EOT/FIM/EOG detection), the special-token cache sorted by
    ///   text length, and `tokenizer_st_partition`.
    /// - Pre-tokenizer: `unicode_regex_split_custom_qwen35`, then GPT-2 byte
    ///   encoding.
    /// - BPE: llama.cpp's bigram queue (lowest rank first, ties to the
    ///   leftmost pair).
    /// - Pieces: `token_to_piece` (control text only with `special`, byte
    ///   decoding of normal tokens).
    ///
    /// All text is handled as UTF-8 bytes (never Swift `String` equality,
    /// which would apply canonical equivalence). Unicode categories come from
    /// the OS's Unicode tables (`Unicode.Scalar.Properties`) instead of
    /// llama.cpp's generated table; they agree for assigned code points of
    /// the Unicode version both cover.
    public final class ReferenceTokenizer: TextTokenizer {
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

        public let data: TokenizerData
        public var vocabSize: Int { texts.count }
        public let eosTokenID: Int32?
        public let bosTokenID: Int32?
        public private(set) var endOfGenerationTokenIDs: Set<Int32> = []
        public private(set) var attrs: [Attr]
        /// Special-token cache in llama.cpp order (longest text first).
        public private(set) var specialTokens: [Int32] = []
        let texts: [[UInt8]]
        let textToID: [[UInt8]: Int32]
        let ranks: [BytePair: Int]
        let addBOS: Bool
        let addEOS: Bool

        struct BytePair: Hashable {
            var left: [UInt8]
            var right: [UInt8]
        }

        public init(data: TokenizerData) throws {
            guard data.model == "gpt2" else {
                throw NativeError.unsupported("tokenizer model \(data.model)")
            }
            guard data.pre == "qwen35" else {
                throw NativeError.unsupported("pre-tokenizer \(data.pre) (the reference implements qwen35)")
            }
            guard data.tokens.count == data.tokenTypes.count || data.tokenTypes.isEmpty else {
                throw NativeError.invalidShape("token types \(data.tokenTypes.count) vs tokens \(data.tokens.count)")
            }
            self.data = data
            var texts: [[UInt8]] = []
            texts.reserveCapacity(data.tokens.count)
            var textToID: [[UInt8]: Int32] = [:]
            textToID.reserveCapacity(data.tokens.count)
            var attrs: [Attr] = []
            attrs.reserveCapacity(data.tokens.count)
            for (i, token) in data.tokens.enumerated() {
                var bytes = Array(token.utf8)
                if bytes.isEmpty { bytes = Array("[EMPTY_\(i)]".utf8) }
                textToID[bytes] = Int32(i)
                texts.append(bytes)
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
            var ranks: [BytePair: Int] = [:]
            ranks.reserveCapacity(data.merges.count)
            for (i, merge) in data.merges.enumerated() {
                let bytes = Array(merge.utf8)
                // llama.cpp splits at the first space after the first byte
                // (no space: an empty pair) and keeps the first rank of a
                // repeated pair (`bpe_ranks.emplace`).
                var pair = BytePair(left: [], right: [])
                if bytes.count > 1, let sep = bytes[1...].firstIndex(of: 0x20) {
                    pair = BytePair(left: Array(bytes[..<sep]), right: Array(bytes[(sep + 1)...]))
                }
                if ranks[pair] == nil { ranks[pair] = i }
            }
            self.texts = texts
            self.textToID = textToID
            self.ranks = ranks
            self.attrs = attrs
            let ids = Self.specialIDs(data)
            bosTokenID = ids.bos
            eosTokenID = ids.eos
            addBOS = data.addBOS ?? false
            addEOS = data.addEOS ?? false
            applySpecialTokenRules()
        }

        // MARK: llama.cpp special-token rules (llama_vocab::impl::load)

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

        private func applySpecialTokenRules() {
            func id(_ text: String) -> Int32? { textToID[Array(text.utf8)] }
            func promote(_ t: Int32) { attrs[Int(t)].insert(.control) }
            func first(_ candidates: [String]) -> Int32? {
                // llama.cpp takes the first match in hash-map order; with a
                // single present candidate the order does not matter. (Qwen
                // has two EOT candidates, <|im_end|> and <|endoftext|>, both
                // EOG either way.)
                candidates.lazy.compactMap { id($0) }.first
            }
            // An id from a GGUF `tokenizer.ggml.<name>_token_id` key (in range;
            // the deprecated prefix/suffix/middle keys are read after the
            // fim_* ones, so they win) is used as is; otherwise the token is
            // detected by text and promoted to control, as llama.cpp does.
            let keyed = data.specialTokenIDs ?? [:]
            func fromKeys(_ names: [String]) -> Int32? {
                var found: Int32?
                for name in names {
                    if let v = keyed[name], v >= 0, Int(v) < texts.count { found = v }
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
            // Control tokens whose text contains "unused" are marked unused.
            for (i, text) in texts.enumerated() where attrs[i].contains(.control) {
                if text.windowsContain(Array("unused".utf8)) { attrs[i].insert(.unused) }
            }
            var eog: Set<Int32> = []
            for t in [fimPad, fimRep, fimSep].compactMap({ $0 }) { eog.insert(t) }
            let eogTexts = ["<|eot_id|>", "<|eot|>", "<|im_end|>", "<|end|>", "<|return|>", "<|call|>",
                            "<|flush|>", "<|calls|>", "<end_of_turn>", "<|endoftext|>", "</s>", "<|eom_id|>",
                            "<EOT>", "_<EOT>", "[EOT]", "[EOS]", "<|end_of_text|>", "<end_of_utterance>", "<eos>",
                            "<turn|>", "<|tool_response>", "<｜end▁of▁sentence｜>", "[e~[", "<|ifm|im_end|>"]
            for text in eogTexts {
                if let t = id(text) {
                    eog.insert(t)
                    promote(t)
                }
            }
            for text in ["<|channel|>", "<|message|>", "<|start|>", "<|constrain|>"] {
                if let t = id(text) { attrs[Int(t)] = .userDefined }
            }
            for t in [eosTokenID, eot, eom].compactMap({ $0 }) { eog.insert(t) }
            // o200k_harmony / solar-open: "<|end|>" is not EOG next to these.
            let eogTextSet = Set(eog.map { texts[Int($0)] })
            let has = { (s: String) in eogTextSet.contains(Array(s.utf8)) }
            if let end = id("<|end|>"), eog.contains(end),
               (has("<|return|>") && has("<|call|>")) || ((has("<|call|>") || has("<|calls|>")) && has("<|flush|>")) {
                eog.remove(end)
                attrs[Int(end)] = .userDefined
            }
            // gemma4 / plamo / beam: a normal "</s>".
            if let s = id("</s>"), eog.contains(s),
               id("<rfl:parameter name=\"") != nil || has("<|tool_response>") || has("<|plamo:eos|>") {
                eog.remove(s)
                attrs[Int(s)] = .normal
            }
            endOfGenerationTokenIDs = eog

            let special = (0..<Int32(texts.count)).filter {
                !attrs[Int($0)].intersection([.control, .userDefined, .unknown]).isEmpty
            }
            // llama.cpp sorts by text length only (std::sort); equal lengths
            // keep id order here.
            specialTokens = special.enumerated().sorted { a, b in
                let la = texts[Int(a.element)].count, lb = texts[Int(b.element)].count
                return la != lb ? la > lb : a.offset < b.offset
            }.map { $0.element }
        }

        // MARK: Encode

        enum Fragment {
            case raw(start: Int, count: Int)
            case token(Int32)
        }

        public func encode(_ text: String, addSpecial: Bool, parseSpecial: Bool) throws -> [Int32] {
            let bytes = Array(text.utf8)
            var output: [Int32] = []
            if addSpecial && addBOS {
                guard let bos = bosTokenID else { throw NativeError.missing("BOS token") }
                output.append(bos)
            }
            if !bytes.isEmpty {
                for fragment in partition(bytes, parseSpecial: parseSpecial) {
                    switch fragment {
                    case .token(let t): output.append(t)
                    case .raw(let start, let count):
                        bpe(Array(bytes[start ..< start + count]), into: &output)
                    }
                }
            }
            if addSpecial && addEOS {
                guard let eos = eosTokenID else { throw NativeError.missing("EOS token") }
                output.append(eos)
            }
            return output
        }

        /// `tokenizer_st_partition`.
        func partition(_ text: [UInt8], parseSpecial: Bool) -> [Fragment] {
            var buffer: [Fragment] = [.raw(start: 0, count: text.count)]
            for special in specialTokens {
                let attr = attrs[Int(special)]
                if !parseSpecial && !attr.intersection([.control, .unknown]).isEmpty { continue }
                let needle = texts[Int(special)]
                var next: [Fragment] = []
                for fragment in buffer {
                    guard case .raw(var offset, var length) = fragment else {
                        next.append(fragment)
                        continue
                    }
                    while true {
                        guard let match = text.find(needle, from: offset, upTo: offset + length) else {
                            next.append(.raw(start: offset, count: length))
                            break
                        }
                        if match > offset {
                            var leftLength = match - offset
                            if attr.contains(.lstrip) {
                                while leftLength > 0 && isSpace(text[offset + leftLength - 1]) { leftLength -= 1 }
                            }
                            if leftLength > 0 { next.append(.raw(start: offset, count: leftLength)) }
                        }
                        next.append(.token(special))
                        if match + needle.count < offset + length {
                            var rightOffset = match + needle.count
                            var rightLength = length - ((match - offset) + needle.count)
                            if attr.contains(.rstrip) {
                                while rightLength > 0 && isSpace(text[rightOffset]) {
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
                buffer = next
            }
            return buffer
        }

        private func isSpace(_ b: UInt8) -> Bool { b == 0x20 || (0x09...0x0D).contains(b) }

        /// `llm_tokenizer_bpe_session::tokenize` for one raw fragment.
        func bpe(_ text: [UInt8], into output: inout [Int32]) {
            for word in Self.preTokenize(text) {
                for piece in merge(word) {
                    if let t = textToID[piece] {
                        output.append(t)
                    } else {
                        for b in piece {
                            if let t = textToID[[b]] { output.append(t) }
                        }
                    }
                }
            }
        }

        /// The qwen35 split, then GPT-2 byte encoding of each word.
        static func preTokenize(_ text: [UInt8]) -> [[UInt8]] {
            let cpts = Array(String(decoding: text, as: UTF8.self).unicodeScalars.map { $0.value })
            var words: [[UInt8]] = []
            var start = 0
            for length in ReferenceUnicode.splitQwen35(cpts) {
                var word: [UInt8] = []
                for cp in cpts[start ..< start + length] {
                    for b in ReferenceUnicode.utf8(cp) { word += ReferenceUnicode.byteToUTF8[Int(b)] }
                }
                words.append(word)
                start += length
            }
            return words
        }

        struct Symbol {
            var start: Int
            var n: Int
            var prev: Int
            var next: Int
        }

        struct Bigram {
            var left: Int
            var right: Int
            var text: [UInt8]
            var rank: Int
        }

        /// BPE merges of one byte-encoded word.
        func merge(_ word: [UInt8]) -> [[UInt8]] {
            var symbols: [Symbol] = []
            var offset = 0
            var index = 0
            while offset < word.count {
                let n = min(word.count - offset, ReferenceUnicode.utf8Length(word[offset]))
                symbols.append(Symbol(start: offset, n: n, prev: index - 1,
                                      next: offset + n == word.count ? -1 : index + 1))
                offset += n
                index += 1
            }
            var queue = BigramQueue()
            func text(_ i: Int) -> [UInt8] { Array(word[symbols[i].start ..< symbols[i].start + symbols[i].n]) }
            func addBigram(_ left: Int, _ right: Int) {
                guard left != -1, right != -1 else { return }
                let l = text(left), r = text(right)
                guard let rank = ranks[BytePair(left: l, right: r)] else { return }
                queue.push(Bigram(left: left, right: right, text: l + r, rank: rank))
            }
            if symbols.count > 1 {
                for i in 1..<symbols.count { addBigram(i - 1, i) }
            }
            while let bigram = queue.pop() {
                let l = symbols[bigram.left], r = symbols[bigram.right]
                if l.n == 0 || r.n == 0 { continue }
                if text(bigram.left) + text(bigram.right) != bigram.text { continue }
                symbols[bigram.left].n += r.n
                symbols[bigram.right].n = 0
                symbols[bigram.left].next = r.next
                if r.next >= 0 { symbols[r.next].prev = bigram.left }
                addBigram(symbols[bigram.left].prev, bigram.left)
                addBigram(bigram.left, symbols[bigram.left].next)
            }
            return symbols.filter { $0.n > 0 }.map { Array(word[$0.start ..< $0.start + $0.n]) }
        }

        /// Min-heap on (rank, left), llama.cpp's `llm_bigram_bpe::comparator`.
        struct BigramQueue {
            var heap: [Bigram] = []
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
            guard token >= 0, Int(token) < texts.count else { return [] }
            let attr = attrs[Int(token)]
            if !special && !attr.intersection([.unknown, .control]).isEmpty { return [] }
            let text = texts[Int(token)]
            if !attr.intersection([.unknown, .control, .userDefined]).isEmpty { return text }
            if attr.contains(.normal) { return ReferenceUnicode.decodeByteText(text) }
            if attr.contains(.byte) {
                // `token_to_byte` for BPE: (char) strtol(text.substr(3, 2), NULL, 16).
                // A text under 3 bytes makes substr throw at the pin; the
                // reference returns no bytes for it.
                guard text.count >= 3 else { return [] }
                return [UInt8(truncatingIfNeeded: Self.strtol16(Array(text[3 ..< min(5, text.count)])))]
            }
            return []
        }

        public func attr(_ token: Int32) -> Attr { attrs[Int(token)] }

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
            func digit(_ b: UInt8) -> Int? { Int(String(UnicodeScalar(b)), radix: 16) }
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

    /// The pieces of llama.cpp's `unicode.cpp` the reference tokenizer uses.
    public enum ReferenceUnicode {
        struct Flags {
            var letter = false
            var number = false
            var mark = false
            var whitespace = false
            var inRange = false
        }

        static let whitespace: Set<UInt32> = [
            0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x20, 0x85, 0xA0, 0x1680, 0x2000, 0x2001, 0x2002, 0x2003, 0x2004,
            0x2005, 0x2006, 0x2007, 0x2008, 0x2009, 0x200A, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000,
        ]

        static func flags(_ cp: UInt32) -> Flags {
            var f = Flags()
            f.inRange = true
            f.whitespace = whitespace.contains(cp)
            guard let scalar = Unicode.Scalar(cp) else { return f }
            switch scalar.properties.generalCategory {
            case .uppercaseLetter, .lowercaseLetter, .titlecaseLetter, .modifierLetter, .otherLetter:
                f.letter = true
            case .decimalNumber, .letterNumber, .otherNumber:
                f.number = true
            case .nonspacingMark, .spacingMark, .enclosingMark:
                f.mark = true
            default:
                break
            }
            return f
        }

        /// `unicode_regex_split_custom_qwen35` over one span of code points;
        /// returns the word lengths in code points.
        static func splitQwen35(_ cpts: [UInt32]) -> [Int] {
            let end = cpts.count
            let outOfRange: UInt32 = 0xFFFF_FFFF
            func cpt(_ pos: Int) -> UInt32 { pos >= 0 && pos < end ? cpts[pos] : outOfRange }
            func fl(_ pos: Int) -> Flags { pos >= 0 && pos < end ? flags(cpts[pos]) : Flags() }
            func lower(_ c: UInt32) -> UInt32 { (0x41...0x5A).contains(c) ? c + 0x20 : c }
            var out: [Int] = []
            var prevEnd = 0
            @discardableResult func addToken(_ e: Int) -> Int {
                let len = e - prevEnd
                if len > 0 { out.append(len) }
                prevEnd = e
                return len
            }
            let cr: UInt32 = 0x0D, lf: UInt32 = 0x0A
            var pos = 0
            while pos < end {
                let c = cpt(pos)
                let f = fl(pos)
                // (?i:'s|'t|'re|'ve|'m|'ll|'d)
                if c == 0x27 && pos + 1 < end {
                    let n1 = lower(cpt(pos + 1))
                    if n1 == 0x73 || n1 == 0x74 || n1 == 0x6D || n1 == 0x64 {
                        pos += addToken(pos + 2)
                        continue
                    }
                    if pos + 2 < end {
                        let n2 = lower(cpt(pos + 2))
                        if (n1 == 0x72 && n2 == 0x65) || (n1 == 0x76 && n2 == 0x65) || (n1 == 0x6C && n2 == 0x6C) {
                            pos += addToken(pos + 3)
                            continue
                        }
                    }
                }
                // [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
                if !(c == cr || c == lf || f.number) {
                    if f.letter || f.mark || fl(pos + 1).mark || fl(pos + 1).letter {
                        pos += 1
                        while fl(pos).letter || fl(pos).mark { pos += 1 }
                        addToken(pos)
                        continue
                    }
                }
                // \p{N}
                if f.number {
                    pos += 1
                    addToken(pos)
                    continue
                }
                // <space>?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
                var f2 = c == 0x20 ? fl(pos + 1) : f
                if !(f2.whitespace || f2.letter || f2.mark || f2.number) && f.inRange {
                    if c == 0x20 { pos += 1 }
                    while !(f2.whitespace || f2.letter || f2.mark || f2.number) && f2.inRange {
                        pos += 1
                        f2 = fl(pos)
                    }
                    var c2 = cpt(pos)
                    while c2 == cr || c2 == lf {
                        pos += 1
                        c2 = cpt(pos)
                    }
                    addToken(pos)
                    continue
                }
                var spaces = 0
                var lastEndRN = 0
                while fl(pos + spaces).whitespace {
                    let c2 = cpt(pos + spaces)
                    if c2 == cr || c2 == lf { lastEndRN = pos + spaces + 1 }
                    spaces += 1
                }
                // \s*[\r\n]+
                if lastEndRN > 0 {
                    pos = lastEndRN
                    addToken(pos)
                    continue
                }
                // \s+(?!\S)
                if spaces > 1 && cpt(pos + spaces) != outOfRange {
                    pos += spaces - 1
                    addToken(pos)
                    continue
                }
                // \s+
                if spaces > 0 {
                    pos += spaces
                    addToken(pos)
                    continue
                }
                pos += 1
                addToken(pos)
            }
            return out
        }

        /// GPT-2 `bytes_to_unicode`, as UTF-8 bytes per input byte.
        static let byteToUTF8: [[UInt8]] = {
            var map = [UInt32](repeating: 0, count: 256)
            var assigned = [Bool](repeating: false, count: 256)
            for ch in Array(0x21...0x7E) + Array(0xA1...0xAC) + Array(0xAE...0xFF) {
                map[ch] = UInt32(ch)
                assigned[ch] = true
            }
            var n: UInt32 = 0
            for ch in 0..<256 where !assigned[ch] {
                map[ch] = 256 + n
                n += 1
            }
            return map.map { utf8($0) }
        }()

        /// The inverse map, code point to byte.
        static let utf8ToByte: [UInt32: UInt8] = {
            var out: [UInt32: UInt8] = [:]
            for b in 0..<256 {
                let cp = String(decoding: byteToUTF8[b], as: UTF8.self).unicodeScalars.first!.value
                out[cp] = UInt8(b)
            }
            return out
        }()

        static func utf8(_ cp: UInt32) -> [UInt8] {
            guard let s = Unicode.Scalar(cp) else { return [] }
            return Array(String(Character(s)).utf8)
        }

        /// `unicode_len_utf8`.
        static func utf8Length(_ first: UInt8) -> Int {
            let lookup = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4]
            return lookup[Int(first >> 4)]
        }

        /// `llama_decode_text`.
        static func decodeByteText(_ text: [UInt8]) -> [UInt8] {
            var out: [UInt8] = []
            for scalar in String(decoding: text, as: UTF8.self).unicodeScalars {
                if let b = utf8ToByte[scalar.value] {
                    out.append(b)
                } else {
                    out += Array("[UNK_BYTE_0x".utf8)
                    for b in utf8(scalar.value) { out += Array(String(format: "%02x", b).utf8) }
                    // llama.cpp appends the whole token text, then "]".
                    out += text + Array("]".utf8)
                }
            }
            return out
        }
    }
}

extension Array where Element == UInt8 {
    /// First index of `needle` starting in `from ..< upTo` and ending by `upTo`.
    func find(_ needle: [UInt8], from: Int, upTo: Int) -> Int? {
        guard !needle.isEmpty, upTo - from >= needle.count else { return nil }
        var i = from
        while i + needle.count <= upTo {
            if self[i] == needle[0] {
                var j = 1
                while j < needle.count && self[i + j] == needle[j] { j += 1 }
                if j == needle.count { return i }
            }
            i += 1
        }
        return nil
    }

    func windowsContain(_ needle: [UInt8]) -> Bool { find(needle, from: 0, upTo: count) != nil }
}

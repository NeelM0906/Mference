import Foundation

extension Native {
    /// The pieces of llama.cpp's `unicode.cpp` (at the pin) the pack tokenizer
    /// needs, over flat code-point buffers.
    ///
    /// Categories come from the OS's Unicode tables
    /// (`Unicode.Scalar.Properties.generalCategory`); the Basic Multilingual
    /// Plane is tabulated once, other planes are looked up per code point.
    public enum PackUnicode {
        static let letter: UInt8 = 1
        static let number: UInt8 = 2
        static let mark: UInt8 = 4
        static let whitespace: UInt8 = 8
        /// Set for every in-range position (out of range is all zero).
        static let inRange: UInt8 = 16

        /// llama.cpp's whitespace code points (`unicode_ranges_whitespace`).
        static let whitespaceCodePoints: [UInt32] = [
            0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x20, 0x85, 0xA0, 0x1680, 0x2000, 0x2001, 0x2002, 0x2003, 0x2004,
            0x2005, 0x2006, 0x2007, 0x2008, 0x2009, 0x200A, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000,
        ]

        static func computeFlags(_ cp: UInt32) -> UInt8 {
            var f = inRange
            if whitespaceCodePoints.contains(cp) { f |= whitespace }
            guard let scalar = Unicode.Scalar(cp) else { return f }
            switch scalar.properties.generalCategory {
            case .uppercaseLetter, .lowercaseLetter, .titlecaseLetter, .modifierLetter, .otherLetter:
                f |= letter
            case .decimalNumber, .letterNumber, .otherNumber:
                f |= number
            case .nonspacingMark, .spacingMark, .enclosingMark:
                f |= mark
            default:
                break
            }
            return f
        }

        static let bmpFlags: [UInt8] = (0 ..< 0x10000).map { computeFlags(UInt32($0)) }

        @inline(__always)
        static func flags(_ cp: UInt32) -> UInt8 {
            cp < 0x10000 ? bmpFlags[Int(cp)] : computeFlags(cp)
        }

        /// GPT-2 `bytes_to_unicode`: the code point each byte maps to.
        static let byteToCodePoint: [UInt32] = {
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
            return map
        }()

        /// The UTF-8 bytes of `byteToCodePoint[b]`.
        static let byteToUTF8: [[UInt8]] = byteToCodePoint.map { utf8($0) }

        /// Inverse of `byteToCodePoint`: code point → byte + 1 (0 = unmapped);
        /// every mapped code point is below 0x144.
        static let codePointToByte: [UInt16] = {
            var out = [UInt16](repeating: 0, count: 0x144)
            for (b, cp) in byteToCodePoint.enumerated() { out[Int(cp)] = UInt16(b) + 1 }
            return out
        }()

        static func utf8(_ cp: UInt32) -> [UInt8] {
            switch cp {
            case 0 ..< 0x80: return [UInt8(cp)]
            case 0x80 ..< 0x800: return [UInt8(0xC0 | (cp >> 6)), UInt8(0x80 | (cp & 0x3F))]
            case 0x800 ..< 0x10000:
                return [UInt8(0xE0 | (cp >> 12)), UInt8(0x80 | ((cp >> 6) & 0x3F)), UInt8(0x80 | (cp & 0x3F))]
            default:
                return [UInt8(0xF0 | (cp >> 18)), UInt8(0x80 | ((cp >> 12) & 0x3F)),
                        UInt8(0x80 | ((cp >> 6) & 0x3F)), UInt8(0x80 | (cp & 0x3F))]
            }
        }

        /// Decodes well-formed UTF-8 (a Swift string's bytes) into code points
        /// and the byte offset where each starts; `offsets` gets one extra
        /// entry, the end.
        static func decode(_ bytes: UnsafeBufferPointer<UInt8>, cps: inout [UInt32], offsets: inout [Int32]) {
            cps.removeAll(keepingCapacity: true)
            offsets.removeAll(keepingCapacity: true)
            var i = 0
            let n = bytes.count
            while i < n {
                let b0 = UInt32(bytes[i])
                offsets.append(Int32(i))
                if b0 < 0x80 {
                    cps.append(b0)
                    i += 1
                } else if b0 < 0xE0, i + 1 < n {
                    cps.append((b0 & 0x1F) << 6 | (UInt32(bytes[i + 1]) & 0x3F))
                    i += 2
                } else if b0 < 0xF0, i + 2 < n {
                    cps.append((b0 & 0x0F) << 12 | (UInt32(bytes[i + 1]) & 0x3F) << 6 | (UInt32(bytes[i + 2]) & 0x3F))
                    i += 3
                } else if i + 3 < n {
                    cps.append((b0 & 0x07) << 18 | (UInt32(bytes[i + 1]) & 0x3F) << 12
                               | (UInt32(bytes[i + 2]) & 0x3F) << 6 | (UInt32(bytes[i + 3]) & 0x3F))
                    i += 4
                } else {
                    // Truncated sequence: cannot occur in a Swift string.
                    cps.append(0xFFFD)
                    i = n
                }
            }
            offsets.append(Int32(n))
        }

        /// `unicode_regex_split_custom_qwen35` over `cps` (with `flags[i] =
        /// flags(cps[i])`): appends each word's end index (exclusive, in code
        /// points) to `ends`.
        static func splitQwen35(_ cps: UnsafeBufferPointer<UInt32>, _ flags: UnsafeBufferPointer<UInt8>,
                                into ends: inout [Int32]) {
            let end = cps.count
            let outOfRange: UInt32 = 0xFFFF_FFFF
            @inline(__always) func cpt(_ pos: Int) -> UInt32 { pos >= 0 && pos < end ? cps[pos] : outOfRange }
            @inline(__always) func fl(_ pos: Int) -> UInt8 { pos >= 0 && pos < end ? flags[pos] : 0 }
            @inline(__always) func lower(_ c: UInt32) -> UInt32 { (0x41...0x5A).contains(c) ? c + 0x20 : c }
            let lm = letter | mark, wlmn = whitespace | letter | mark | number
            var prevEnd = 0
            @inline(__always) func addToken(_ e: Int) -> Int {
                let len = e - prevEnd
                if len > 0 { ends.append(Int32(e)) }
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
                if !(c == cr || c == lf || f & number != 0) {
                    if f & lm != 0 || fl(pos + 1) & lm != 0 {
                        pos += 1
                        while fl(pos) & lm != 0 { pos += 1 }
                        _ = addToken(pos)
                        continue
                    }
                }
                // \p{N}
                if f & number != 0 {
                    pos += 1
                    _ = addToken(pos)
                    continue
                }
                // <space>?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
                var f2 = c == 0x20 ? fl(pos + 1) : f
                if f2 & wlmn == 0 && f & inRange != 0 {
                    if c == 0x20 { pos += 1 }
                    while f2 & wlmn == 0 && f2 & inRange != 0 {
                        pos += 1
                        f2 = fl(pos)
                    }
                    var c2 = cpt(pos)
                    while c2 == cr || c2 == lf {
                        pos += 1
                        c2 = cpt(pos)
                    }
                    _ = addToken(pos)
                    continue
                }
                var spaces = 0
                var lastEndRN = 0
                while fl(pos + spaces) & whitespace != 0 {
                    let c2 = cpt(pos + spaces)
                    if c2 == cr || c2 == lf { lastEndRN = pos + spaces + 1 }
                    spaces += 1
                }
                // \s*[\r\n]+
                if lastEndRN > 0 {
                    pos = lastEndRN
                    _ = addToken(pos)
                    continue
                }
                // \s+(?!\S)
                if spaces > 1 && cpt(pos + spaces) != outOfRange {
                    pos += spaces - 1
                    _ = addToken(pos)
                    continue
                }
                // \s+
                if spaces > 0 {
                    pos += spaces
                    _ = addToken(pos)
                    continue
                }
                pos += 1
                _ = addToken(pos)
            }
        }
    }
}

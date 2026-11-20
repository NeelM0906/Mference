import Foundation

/// Turns text into vectors for Find. The live one runs EmbeddingGemma 2
/// through the engine (`Engine/FindEmbedderLive.swift`); tests use
/// `HashingTextEmbedder`, which never loads a model.
public protocol FindTextEmbedding: Sendable {
    /// Vectors for document passages, in order (raw: not truncated or normalised).
    func embed(documents: [FindEmbeddingInput]) async throws -> [[Float]]
    /// The vector for one search query.
    func embed(query: String) async throws -> [Float]
    /// Frees the model's memory; the next call loads it again.
    func unload() async
    /// Identifies the model and its prompts; a change re-embeds the index.
    var fingerprint: String { get }
    /// Whether `embed(media:)` works (the media encoders are there).
    var supportsMedia: Bool { get }
    /// Vectors for photos, frames and sound, in the same space as text (raw,
    /// like `embed(documents:)`); nil for an input that could not be
    /// embedded.
    func embed(media: [FindMediaData]) async throws -> [[Float]?]
    /// Identifies the media encoders and how media is sampled; a change
    /// reads photos, video and audio again.
    var mediaFingerprint: String { get }
    /// Work at user-initiated priority (a large backlog) or in the background.
    func setHurry(_ hurry: Bool) async
}

extension FindTextEmbedding {
    public func setHurry(_ hurry: Bool) async {}
    public var supportsMedia: Bool { false }
    public var mediaFingerprint: String { "none" }
    public func embed(media: [FindMediaData]) async throws -> [[Float]?] {
        media.map { _ in nil }
    }
}

/// One passage to embed.
public struct FindEmbeddingInput: Sendable, Hashable {
    public var title: String?
    public var text: String

    public init(title: String?, text: String) {
        self.title = title
        self.text = text
    }
}

/// EmbeddingGemma's retrieval prompts (model card): queries and documents
/// are prefixed differently.
public enum FindEmbeddingPrompts {
    public static func query(_ text: String) -> String {
        "task: search result | query: \(text)"
    }

    public static func document(_ input: FindEmbeddingInput) -> String {
        let title = input.title.map { $0.isEmpty ? "none" : $0 } ?? "none"
        return "title: \(title) | text: \(input.text)"
    }
}

/// Vector handling for the index: Matryoshka truncation, normalisation,
/// int8 and 1-bit codes, and the two scans.
public enum FindVectors {
    /// Dimensions kept (EmbeddingGemma 2 is trained for 768 → 512/256/128;
    /// 256 costs about a point of MMTEB for a third of the storage).
    public static let dimensions = 256
    /// 64-bit words in the 1-bit code.
    public static let bitWords = dimensions / 64
    /// Bytes stored per passage: scale (4) + bits (32) + int8 (256).
    public static let recordBytes = 4 + bitWords * 8 + dimensions

    /// The first `dimensions` values, L2-normalised. Shorter vectors are
    /// zero-padded.
    public static func prepare(_ raw: [Float]) -> [Float] {
        var vector = Array(raw.prefix(dimensions))
        if vector.count < dimensions { vector += [Float](repeating: 0, count: dimensions - vector.count) }
        let norm = vector.reduce(0) { $0 + $1 * $1 }.squareRoot()
        guard norm > 0 else { return vector }
        return vector.map { $0 / norm }
    }

    /// Packs a prepared vector into its stored record.
    public static func encode(_ vector: [Float]) -> Data {
        precondition(vector.count == dimensions)
        let maxAbs = vector.reduce(Float(0)) { max($0, abs($1)) }
        let scale = maxAbs > 0 ? maxAbs / 127 : 1
        var data = Data(capacity: recordBytes)
        withUnsafeBytes(of: scale.bitPattern.littleEndian) { data.append(contentsOf: $0) }
        for word in 0..<bitWords {
            var bits: UInt64 = 0
            for bit in 0..<64 where vector[word * 64 + bit] > 0 {
                bits |= 1 << UInt64(bit)
            }
            withUnsafeBytes(of: bits.littleEndian) { data.append(contentsOf: $0) }
        }
        for value in vector {
            let quantized = (value / scale).rounded()
            data.append(UInt8(bitPattern: Int8(max(-127, min(127, quantized)))))
        }
        return data
    }

    /// The query side of the scans.
    public struct Query: Sendable {
        public let vector: [Float]
        public let bits: [UInt64]

        public init(_ prepared: [Float]) {
            vector = prepared
            var bits = [UInt64](repeating: 0, count: bitWords)
            for word in 0..<bitWords {
                for bit in 0..<64 where prepared[word * 64 + bit] > 0 {
                    bits[word] |= 1 << UInt64(bit)
                }
            }
            self.bits = bits
        }
    }

    /// Hamming distance between the query's bits and a record's.
    @inline(__always)
    static func hamming(_ query: Query, record: UnsafeRawBufferPointer) -> Int {
        var distance = 0
        for word in 0..<bitWords {
            let stored = UInt64(littleEndian: record.loadUnaligned(fromByteOffset: 4 + word * 8, as: UInt64.self))
            distance += (stored ^ query.bits[word]).nonzeroBitCount
        }
        return distance
    }

    /// Cosine between the (normalised) query and a record's int8 vector.
    @inline(__always)
    static func cosine(_ query: Query, record: UnsafeRawBufferPointer) -> Float {
        let scale = Float(bitPattern: UInt32(littleEndian: record.loadUnaligned(as: UInt32.self)))
        let base = 4 + bitWords * 8
        var sum: Float = 0
        query.vector.withUnsafeBufferPointer { q in
            for index in 0..<dimensions {
                sum += q[index] * Float(Int8(bitPattern: record[base + index]))
            }
        }
        return sum * scale
    }
}

/// A model-free embedder for tests and previews: hashed bag of words (with a
/// few hand-made synonyms so "meaning" tests can tell it from word search).
/// With `media` on it also "sees" pictures by their main colour ("red",
/// "green", "blue", "dark", "light") and "hears" sound as "loud" or "quiet".
/// Deterministic; never loads anything.
public struct HashingTextEmbedder: FindTextEmbedding {
    public var fingerprint: String { "hashing-v1" }
    public var mediaFingerprint: String { media ? "hashing-media-v1" : "none" }
    /// Words treated as the same meaning.
    public var synonyms: [String: String]
    public var media: Bool
    public var supportsMedia: Bool { media }

    public init(synonyms: [String: String] = HashingTextEmbedder.defaultSynonyms, media: Bool = false) {
        self.synonyms = synonyms
        self.media = media
    }

    public func embed(media inputs: [FindMediaData]) async throws -> [[Float]?] {
        guard media else { return inputs.map { _ in nil } }
        return inputs.map { input in
            switch input {
            case .image(_, _, let rgb):
                var sums = [Double](repeating: 0, count: 3)
                for (index, value) in rgb.enumerated() { sums[index % 3] += Double(value) }
                let pixels = Double(max(1, rgb.count / 3))
                let (red, green, blue) = (sums[0] / pixels, sums[1] / pixels, sums[2] / pixels)
                let brightest = max(red, green, blue), darkest = min(red, green, blue)
                let word: String
                if brightest - darkest < 40 {
                    word = brightest < 100 ? "dark" : "light"
                } else {
                    word = brightest == red ? "red" : brightest == green ? "green" : "blue"
                }
                return vector(for: word)
            case .audio(let samples):
                let rms = (samples.reduce(Float(0)) { $0 + $1 * $1 } / Float(max(1, samples.count))).squareRoot()
                return vector(for: rms > 0.1 ? "loud" : "quiet")
            }
        }
    }

    public static let defaultSynonyms: [String: String] = [
        "pets": "animal", "pet": "animal", "cat": "animal", "dog": "animal", "dogs": "animal", "cats": "animal",
        "revenue": "money", "sales": "money", "income": "money",
        "wifi": "network", "wi-fi": "network", "internet": "network",
    ]

    public func embed(documents: [FindEmbeddingInput]) async throws -> [[Float]] {
        documents.map { vector(for: ($0.title ?? "") + " " + $0.text) }
    }

    public func embed(query: String) async throws -> [Float] {
        // The model-facing prompt prefix carries no meaning of its own here.
        let prefix = FindEmbeddingPrompts.query("")
        return vector(for: query.hasPrefix(prefix) ? String(query.dropFirst(prefix.count)) : query)
    }

    public func unload() async {}

    func vector(for text: String) -> [Float] {
        var vector = [Float](repeating: 0, count: 768)
        for raw in FindText.words(text) {
            let word = synonyms[raw] ?? raw
            guard !FindText.stopWords.contains(word) else { continue }
            var hasher = StableHash()
            hasher.combine(word)
            // Inside the kept Matryoshka prefix, so truncation keeps every word.
            let index = Int(hasher.value % UInt64(FindVectors.dimensions))
            vector[index] += 1
        }
        return vector
    }

    /// FNV-1a, stable across runs (Swift's Hasher is seeded per process).
    struct StableHash {
        var value: UInt64 = 0xcbf2_9ce4_8422_2325
        mutating func combine(_ text: String) {
            for byte in text.utf8 {
                value ^= UInt64(byte)
                value = value &* 0x100_0000_01b3
            }
        }
    }
}

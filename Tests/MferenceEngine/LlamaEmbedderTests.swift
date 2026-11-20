import CoreGraphics
import Foundation
import ImageIO
import Testing
@testable import MferenceEngine

@Suite struct LlamaEmbedderTests {
    @Test func packKeepsOrderAndRespectsTokenBudget() {
        let groups = LlamaEmbedder.pack(tokenCounts: [900, 900, 300, 2_048, 10], batchTokens: 2_048, maxSequences: 16)
        #expect(groups == [[0, 1], [2], [3], [4]])
    }

    @Test func packRespectsSequenceLimit() {
        let groups = LlamaEmbedder.pack(tokenCounts: Array(repeating: 5, count: 7), batchTokens: 2_048, maxSequences: 3)
        #expect(groups == [[0, 1, 2], [3, 4, 5], [6]])
    }

    @Test func packOfNothingIsEmpty() {
        #expect(LlamaEmbedder.pack(tokenCounts: [], batchTokens: 512, maxSequences: 4).isEmpty)
    }

    @Test func missingFileFailsToLoad() {
        let url = FileManager.default.temporaryDirectory.appending(path: "no-such-embedder-\(UUID()).gguf")
        #expect(throws: LlamaEmbedderError.modelLoadFailed(url.path)) {
            _ = try LlamaEmbedder(modelPath: url)
        }
    }

    @Test func imageSizeFillsTheBudgetInWholeTiles() {
        let size = LlamaEmbedder.imageSize(width: 640, height: 488, tokens: 140)
        #expect(size?.width == 624 && size?.height == 480)
        // 13 × 10 tiles of 48 px: within the 140-token budget.
        #expect((624 / 48) * (480 / 48) <= 140)
    }

    @Test func imageSizeKeepsThinImagesOneTileHigh() {
        let size = LlamaEmbedder.imageSize(width: 10_000, height: 10, tokens: 140)
        #expect(size?.height == 48 && size?.width == 140 * 48)
        #expect(LlamaEmbedder.imageSize(width: 0, height: 10, tokens: 140) == nil)
    }

    @Test func mediaWithoutAMediaModelIsUnavailable() {
        let url = FileManager.default.temporaryDirectory.appending(path: "no-such-embedder-\(UUID()).gguf")
        // Loading fails first; the type still says it has no media.
        #expect((try? LlamaEmbedder(modelPath: url)) == nil)
    }

    // A model run: only with `MFERENCE_EMBEDDING_GGUF` pointing at the
    // EmbeddingGemma 2 text GGUF, under the machine lock (AGENTS.md).
    static let modelPath = ProcessInfo.processInfo.environment["MFERENCE_EMBEDDING_GGUF"]

    @Test(.enabled(if: Self.modelPath != nil, "set MFERENCE_EMBEDDING_GGUF"))
    func realModelRanksTheRelatedPassageFirst() throws {
        let embedder = try LlamaEmbedder(modelPath: URL(filePath: Self.modelPath!))
        defer { embedder.unload() }
        let texts = [
            "task: search result | query: lease clause about pets",
            "title: lease | text: Tenant may keep one cat or one dog under 30 lb, with a refundable deposit.",
            "title: recipe | text: Whisk the eggs with sugar until pale, then fold in the flour.",
        ]
        let vectors = try embedder.embed(texts)
        #expect(vectors.count == 3)
        #expect(vectors.allSatisfy { $0.count == embedder.dimensions })
        func cosine(_ a: [Float], _ b: [Float]) -> Float {
            let dot = zip(a, b).reduce(0) { $0 + $1.0 * $1.1 }
            let norms = (a.reduce(0) { $0 + $1 * $1 } * b.reduce(0) { $0 + $1 * $1 }).squareRoot()
            return dot / norms
        }
        #expect(cosine(vectors[0], vectors[1]) > cosine(vectors[0], vectors[2]))
        // The same text alone and in a batch gives the same vector.
        let alone = try embedder.embed([texts[1]])[0]
        #expect(cosine(alone, vectors[1]) > 0.999)
    }

    // Media: with `MFERENCE_EMBEDDING_MMPROJ` too (the mmproj GGUF), under
    // the machine lock. Uses the account pictures every Mac has.
    static let mediaPath = ProcessInfo.processInfo.environment["MFERENCE_EMBEDDING_MMPROJ"]

    @Test(.enabled(if: Self.modelPath != nil && Self.mediaPath != nil, "set MFERENCE_EMBEDDING_GGUF and _MMPROJ"))
    func realModelMatchesPicturesAndSoundToText() throws {
        let embedder = try LlamaEmbedder(modelPath: URL(filePath: Self.modelPath!),
                                         mediaModelPath: URL(filePath: Self.mediaPath!))
        defer { embedder.unload() }
        let pictures = ["Animals/Zebra", "Flowers/Sunflower", "Instruments/Guitar", "Animals/Penguin"]
        var images: [[Float]] = []
        var inputs: [LlamaMediaInput] = []
        for name in pictures {
            let url = URL(filePath: "/Library/User Pictures/\(name).heic").resolvingSymlinksInPath()
            let rgb = try #require(Self.rgb(url, tokens: 140))
            let input = LlamaMediaInput.image(width: rgb.width, height: rgb.height, rgb: rgb.bytes)
            inputs.append(input)
            images.append(try embedder.embed(media: input))
        }
        #expect(images.allSatisfy { $0.count == embedder.dimensions })
        // Batched (same-size pictures share the encoder pass, all share the
        // text pass) gives the same vectors, and a broken input is nil
        // without sinking the rest.
        let broken = LlamaMediaInput.image(width: 2, height: 2, rgb: [0])
        let batched = try embedder.embed(media: [inputs[0], broken] + inputs[1...])
        #expect(batched.count == inputs.count + 1)
        #expect(batched[1] == nil)
        for (single, together) in zip(images, [batched[0]] + batched[2...]) {
            #expect(Self.cosine(single, try #require(together)) > 0.999)
        }
        let queries = try embedder.embed(["a zebra", "a yellow sunflower", "a guitar", "a penguin"]
            .map { "task: search result | query: \($0)" })
        for (index, query) in queries.enumerated() {
            let best = images.indices.max { Self.cosine(query, images[$0]) < Self.cosine(query, images[$1]) }
            #expect(best == index, "query \(index) matched \(pictures[best ?? 0])")
        }
        // Audio: one second of a tone gives a vector of the same width.
        let tone = (0..<LlamaEmbedder.audioSampleRate).map { Float(sin(Double($0) * 2 * .pi * 440 / 16_000)) * 0.3 }
        #expect(try embedder.embed(media: .audio(samples: tone)).count == embedder.dimensions)
    }

    static func cosine(_ a: [Float], _ b: [Float]) -> Float {
        let dot = zip(a, b).reduce(0) { $0 + $1.0 * $1.1 }
        let norms = (a.reduce(0) { $0 + $1 * $1 } * b.reduce(0) { $0 + $1 * $1 }).squareRoot()
        return dot / norms
    }

    /// Packed RGB of an image file at the size for `tokens`.
    static func rgb(_ url: URL, tokens: Int) -> (width: Int, height: Int, bytes: [UInt8])? {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil),
              let size = LlamaEmbedder.imageSize(width: image.width, height: image.height, tokens: tokens),
              let context = CGContext(data: nil, width: size.width, height: size.height, bitsPerComponent: 8,
                                      bytesPerRow: size.width * 4, space: CGColorSpaceCreateDeviceRGB(),
                                      bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else { return nil }
        context.interpolationQuality = .high
        context.draw(image, in: CGRect(x: 0, y: 0, width: size.width, height: size.height))
        guard let data = context.data else { return nil }
        let pixels = data.bindMemory(to: UInt8.self, capacity: size.width * size.height * 4)
        var bytes = [UInt8](repeating: 0, count: size.width * size.height * 3)
        for index in 0..<(size.width * size.height) {
            bytes[index * 3] = pixels[index * 4]
            bytes[index * 3 + 1] = pixels[index * 4 + 1]
            bytes[index * 3 + 2] = pixels[index * 4 + 2]
        }
        return (size.width, size.height, bytes)
    }
}

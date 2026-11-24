import CoreGraphics
import Foundation
import ImageIO
import Mference
import Testing
import UniformTypeIdentifiers
@testable import MferenceEngine

/// Model-free tests of image input: the media value, request validation, the
/// prompt rendering llama.cpp's server would produce, the position and
/// prefix-reuse layout of a prompt with images, projector headers, and
/// discovery's pairing of a projector with its model. The real-model check
/// (Quick describing an image through mtmd) is a gated checkpoint run.
@Suite("Llama vision")
struct LlamaVisionTests {
    static func media(_ byte: UInt8, width: Int = 2, height: Int = 2) -> EngineMedia {
        EngineMedia(width: width, height: height, rgb: Data(repeating: byte, count: width * height * 3))!
    }

    // MARK: EngineMedia

    @Test func mediaNeedsMatchingPixelsAndHashesThem() {
        #expect(EngineMedia(width: 2, height: 2, rgb: Data(count: 11)) == nil)
        #expect(EngineMedia(width: 0, height: 2, rgb: Data()) == nil)
        let a = Self.media(1), same = Self.media(1), other = Self.media(2)
        #expect(a.hash == same.hash)
        #expect(a.hash != other.hash)
        #expect(a.hash.count == 64)
        // The size is part of the hash: same bytes, other shape.
        let wide = EngineMedia(width: 4, height: 1, rgb: Data(repeating: 1, count: 12))!
        #expect(wide.hash != a.hash)
    }

    static func png(width: Int, height: Int, red: CGFloat) throws -> Data {
        let space = CGColorSpace(name: CGColorSpace.sRGB)!
        let context = try #require(CGContext(data: nil, width: width, height: height, bitsPerComponent: 8,
                                             bytesPerRow: 0, space: space,
                                             bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue))
        context.setFillColor(CGColor(srgbRed: red, green: 0, blue: 0, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: width, height: height))
        let image = try #require(context.makeImage())
        let data = NSMutableData()
        let destination = try #require(CGImageDestinationCreateWithData(data, UTType.png.identifier as CFString,
                                                                        1, nil))
        CGImageDestinationAddImage(destination, image, nil)
        #expect(CGImageDestinationFinalize(destination))
        return data as Data
    }

    @Test func decodesAnEncodedImageToRGBAndBoundsItsSize() throws {
        let decoded = try #require(EngineMedia.image(encoded: try Self.png(width: 6, height: 4, red: 1)))
        #expect(decoded.width == 6 && decoded.height == 4)
        #expect(decoded.rgb.count == 6 * 4 * 3)
        #expect(Array(decoded.rgb.prefix(3)) == [255, 0, 0])

        let scaled = try #require(EngineMedia.image(encoded: try Self.png(width: 40, height: 20, red: 1),
                                                    maximumSide: 10))
        #expect(scaled.width == 10 && scaled.height == 5)
        #expect(EngineMedia.image(encoded: Data("not an image".utf8)) == nil)
    }

    // MARK: Validation

    static func request(media: [Int: [EngineMedia]]) -> EngineRequest {
        EngineRequest(messages: [MFTokenizer.Message(role: .system, content: "Be brief."),
                                 MFTokenizer.Message(role: .user, content: "What is this?")],
                      maximumCompletionTokens: 16, media: media)
    }

    @Test func mediaIsRefusedByAModelWithoutVisionAndOffUserMessages() throws {
        let plain = Self.request(media: [:])
        #expect(!plain.hasMedia)
        try plain.validateMedia(supportsVision: false)
        // An empty list is no media.
        try Self.request(media: [1: []]).validateMedia(supportsVision: false)

        let image = Self.request(media: [1: [Self.media(1)]])
        #expect(image.hasMedia)
        try image.validateMedia(supportsVision: true)
        #expect(throws: EngineError.invalidRequest(message: "this model does not accept images",
                                                   param: "messages", code: "unsupported_content")) {
            try image.validateMedia(supportsVision: false)
        }
        for index in [0, 2] {
            #expect(throws: EngineError.self) {
                try Self.request(media: [index: [Self.media(1)]]).validateMedia(supportsVision: true)
            }
        }
    }

    @Test func orderedMediaFollowsTheMessages() {
        var request = Self.request(media: [:])
        request.messages.append(MFTokenizer.Message(role: .assistant, content: "A cat."))
        request.messages.append(MFTokenizer.Message(role: .user, content: "And this?"))
        request.media = [3: [Self.media(3)], 1: [Self.media(1), Self.media(2)]]
        #expect(request.orderedMedia.map(\.hash) == [1, 2, 3].map { Self.media(UInt8($0)).hash })
    }

    @Test func theFakeModelSeesImagesOnlyWithAProjector() async throws {
        let backend = FakeEngineBackend(engine: .llamaCpp)
        let blind = try await backend.load(FakeEngineBackend.descriptor(id: "blind", engine: .llamaCpp),
                                           options: EngineLoadOptions())
        #expect(!blind.traits.supportsVision)
        await #expect(throws: EngineError.self) { try await blind.prepare(Self.request(media: [1: [Self.media(1)]])) }

        let projector = EngineVisionProjector(location: URL(fileURLWithPath: "/nonexistent/mmproj.gguf"),
                                              sizeBytes: 1)
        let seeing = try await backend.load(
            FakeEngineBackend.descriptor(id: "seeing", engine: .llamaCpp, visionProjector: projector),
            options: EngineLoadOptions())
        #expect(seeing.traits.supportsVision)
        let textOnly = try await seeing.prepare(Self.request(media: [:]))
        let prepared = try await seeing.prepare(Self.request(media: [1: [Self.media(1), Self.media(2)]]))
        #expect(prepared.promptTokens.count == textOnly.promptTokens.count + 2 * FakeEngineModel.imageTokens)
        #expect(prepared.promptTokens.suffix(2 * FakeEngineModel.imageTokens)
            .allSatisfy { $0 == EngineMedia.placeholderToken })
    }

    // MARK: Rendering

    static let marker = "<__media_test__>"

    @Test func aMessageWithImagesIsMarkersThenItsText() throws {
        let message = MFTokenizer.Message(role: .user, content: "What is this?")
        let plain = try LlamaChatTemplate.templateMessage(message)
        #expect(plain["content"] as? String == "What is this?")
        let one = try LlamaChatTemplate.templateMessage(message, mediaMarkers: 1, marker: Self.marker)
        #expect(one["content"] as? String == Self.marker + "What is this?")
        let two = try LlamaChatTemplate.templateMessage(message, mediaMarkers: 2, marker: Self.marker)
        #expect(two["content"] as? String == Self.marker + Self.marker + "What is this?")
        let imageOnly = try LlamaChatTemplate.templateMessage(MFTokenizer.Message(role: .user, content: nil),
                                                              mediaMarkers: 1, marker: Self.marker)
        #expect(imageOnly["content"] as? String == Self.marker)
    }

    /// llama.cpp's server hands the Quick template a string with the marker in
    /// it (never an image part), so the template prints no vision tokens of
    /// its own; `mtmd_tokenize` adds `<|vision_start|>` and `<|vision_end|>`.
    /// A text-only render is unchanged by the media parameters.
    @Test func theQuickTemplatePlacesTheMarkerWhereLlamaCppDoes() throws {
        let template = try LlamaChatTemplate(source: LlamaChatTemplateFixtures.quickTemplate,
                                             bosToken: nil, eosToken: "<|im_end|>")
        let messages = [MFTokenizer.Message(role: .user, content: "What is this?")]
        let withImage = try template.render(messages: messages, tools: [], mediaCounts: [0: 1],
                                            mediaMarker: Self.marker)
        #expect(withImage.contains("<|im_start|>user\n\(Self.marker)What is this?<|im_end|>"))
        #expect(!withImage.contains("<|vision_start|>"))
        let flattened = try template.render(
            messages: [MFTokenizer.Message(role: .user, content: Self.marker + "What is this?")], tools: [])
        #expect(withImage == flattened)
        #expect(try template.render(messages: messages, tools: [])
            == template.render(messages: messages, tools: [], mediaCounts: [:], mediaMarker: Self.marker))
    }

    // MARK: Layout and prefix reuse

    /// `<|vision_start|>` image `<|vision_end|>` as mtmd splits it: an image
    /// of 6 tokens on a 3 × 2 grid takes 3 positions under M-RoPE.
    static func layout(hash: String = "a1b2c3d4" + String(repeating: "0", count: 56)) -> LlamaMediaLayout {
        LlamaMediaLayout(segments: [
            .text([10, 11, 12]),
            .media(tokens: 6, positions: 3, hash: hash),
            .text([13, 14]),
        ])
    }

    @Test func aLayoutCountsCellsInTokensAndKeysInPositions() {
        let layout = Self.layout()
        #expect(layout.promptTokens == [10, 11, 12, -1, -1, -1, -1, -1, -1, 13, 14])
        #expect(layout.totalPositions == 8)
        let keys = layout.cacheKeys
        #expect(keys.count == 8)
        #expect(Array(keys.prefix(3)) == [10, 11, 12])
        #expect(keys[3..<6].allSatisfy { $0 < 0 })
        #expect(Set(keys[3..<6]).count == 3)
        #expect(Array(keys.suffix(2)) == [13, 14])
        // Another image differs from its first position on.
        let other = Self.layout(hash: "ffff0000" + String(repeating: "0", count: 56)).cacheKeys
        #expect(other[3] != keys[3])
    }

    @Test func remainingWorkSkipsWhatIsReusedButNeverHalfAnImage() throws {
        let layout = Self.layout()
        let cold = try #require(layout.remaining(afterReusing: 0))
        #expect(cold.reusedTokens == 0)
        #expect(cold.pieces == [.text(segment: 0, skipping: 0, position: 0),
                                .media(segment: 1, position: 3),
                                .text(segment: 2, skipping: 0, position: 6)])

        let partText = try #require(layout.remaining(afterReusing: 2))
        #expect(partText.reusedTokens == 2)
        #expect(partText.pieces.first == .text(segment: 0, skipping: 2, position: 2))

        let pastImage = try #require(layout.remaining(afterReusing: 7))
        #expect(pastImage.reusedTokens == 3 + 6 + 1)
        #expect(pastImage.pieces == [.text(segment: 2, skipping: 1, position: 7)])

        #expect(layout.remaining(afterReusing: 4) == nil)
        #expect(layout.remaining(afterReusing: 5) == nil)
    }

    /// Records a layout's prefill in the cache the way `LlamaModel` does:
    /// text tokens, and each image's keys with its full hash.
    static func record(_ layout: LlamaMediaLayout, in cache: inout LlamaPromptCache) {
        for segment in layout.segments {
            switch segment {
            case .text(let tokens): cache.append(tokens)
            case .media(_, let positions, let hash):
                cache.appendMedia(LlamaMediaLayout.mediaKeys(hash: hash, positions: positions), hash: hash)
            }
        }
    }

    /// The prompt cache, keyed by position, reuses a conversation that keeps
    /// its image and stops reusing at an image that changed.
    @Test func thePromptCacheReusesTheSameImageAndNotAnother() {
        let first = Self.layout()
        var cache = LlamaPromptCache()
        #expect(cache.begin(prompt: first.cacheKeys, media: first.mediaStarts, trim: { _ in true }, clear: {}) == 0)
        Self.record(first, in: &cache)
        cache.append([20, 21])  // the answer
        cache.commit()
        #expect(cache.mediaHashes == first.mediaStarts)

        let followUp = LlamaMediaLayout(segments: first.segments + [.text([20, 21, 30, 31])])
        let reused = cache.begin(prompt: followUp.cacheKeys, media: followUp.mediaStarts,
                                 trim: { _ in Issue.record("an extension needs no trim"); return false },
                                 clear: { Issue.record("an extension needs no clear") })
        #expect(reused == first.totalPositions + 2)
        let plan = followUp.remaining(afterReusing: reused)
        #expect(plan?.reusedTokens == first.promptTokens.count + 2)
        #expect(plan?.pieces == [.text(segment: 3, skipping: 2, position: 10)])
        cache.append(Array(followUp.cacheKeys[reused...]))
        cache.commit()

        let swapped = Self.layout(hash: "0badf00d" + String(repeating: "0", count: 56))
        var trimmedAt: Int?
        let again = cache.begin(prompt: swapped.cacheKeys, media: swapped.mediaStarts,
                                trim: { trimmedAt = $0; return true }, clear: {})
        #expect(again == 3)
        #expect(trimmedAt == 3)
        #expect(cache.mediaHashes.isEmpty)
        #expect(swapped.remaining(afterReusing: again)?.pieces.first == .media(segment: 1, position: 3))
    }

    /// Two images whose hashes differ only in the top bit of their first 32
    /// get the same keys at every position; the full hash still tells them
    /// apart, so the second never reuses the first's KV.
    @Test func imagesWithCollidingKeysAreToldApartByTheirFullHash() {
        let rest = String(repeating: "7", count: 56)
        let one = Self.layout(hash: "95abfcd1" + rest)
        let other = Self.layout(hash: "15abfcd1" + rest)
        #expect(one.cacheKeys == other.cacheKeys)

        var cache = LlamaPromptCache()
        _ = cache.begin(prompt: one.cacheKeys, media: one.mediaStarts, trim: { _ in true }, clear: {})
        Self.record(one, in: &cache)
        cache.commit()
        var trimmedAt: Int?
        let reused = cache.begin(prompt: other.cacheKeys, media: other.mediaStarts,
                                 trim: { trimmedAt = $0; return true }, clear: {})
        #expect(reused == 3)
        #expect(trimmedAt == 3)
        #expect(other.remaining(afterReusing: reused)?.pieces.first == .media(segment: 1, position: 3))

        // A memory that cannot roll back starts over instead.
        var recurrent = LlamaPromptCache()
        _ = recurrent.begin(prompt: one.cacheKeys, media: one.mediaStarts, trim: { _ in true }, clear: {})
        Self.record(one, in: &recurrent)
        recurrent.commit()
        var cleared = false
        #expect(recurrent.begin(prompt: other.cacheKeys, media: other.mediaStarts,
                                trim: { _ in false }, clear: { cleared = true }) == 0)
        #expect(cleared)
    }

    @Test func aLayoutListsWhereEachImageStarts() {
        let hash = "a1b2c3d4" + String(repeating: "0", count: 56)
        #expect(Self.layout(hash: hash).mediaStarts == [3: hash])
        let two = LlamaMediaLayout(segments: [.media(tokens: 4, positions: 2, hash: "a"), .text([1]),
                                              .media(tokens: 9, positions: 3, hash: "b")])
        #expect(two.mediaStarts == [0: "a", 3: "b"])
    }

    // MARK: Projector headers and discovery

    static func projectorHeader(dimension: UInt32, vision: Bool = true) -> Data {
        var builder = GGUFHeaderBuilder()
        builder.keys = [
            ("general.architecture", .string("clip")),
            ("general.type", .string("mmproj")),
            ("clip.has_vision_encoder", .bool(vision)),
            ("clip.vision.projection_dim", .u32(dimension)),
            ("clip.vision.image_size", .u32(768)),
            ("clip.projector_type", .string("qwen3vl_merger")),
            ("clip.vision.image_mean", .u32s([1, 2, 3])),
        ]
        builder.tensors = ["mm.0.weight", "mm.2.weight", "v.patch_embd.weight"]
        return builder.data()
    }

    @Test func theReaderRecognisesAProjectorAndAModelsWidth() throws {
        var reader = EngineGGUFReader.Reader(data: Self.projectorHeader(dimension: 2_560))
        let projector = try EngineGGUFReader.parse(&reader)
        #expect(projector.architecture == "clip")
        #expect(projector.contextLength == nil)
        #expect(projector.projector == EngineGGUFProjector(projectorType: "qwen3vl_merger",
                                                           hasVisionEncoder: true, projectionDimension: 2_560))

        var builder = GGUFHeaderBuilder.qwen35(mtp: false)
        builder.keys.append(("qwen35.embedding_length", .u32(2_560)))
        var modelReader = EngineGGUFReader.Reader(data: builder.data())
        let model = try EngineGGUFReader.parse(&modelReader)
        #expect(model.embeddingLength == 2_560)
        #expect(model.projector == nil)
    }

    static let directory = "/lib/gguf"
    static let quickFile = "/lib/gguf/Qwen3.5-4B-UD-Q4_K_XL.gguf"
    static let deepFile = "/lib/gguf/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf"

    static func modelMeta(width: Int, mtp: Bool = false) -> EngineGGUFMetadata {
        EngineGGUFMetadata(architecture: "qwen35", name: "x", contextLength: 262_144, tensorCount: 10,
                           nextNTensorCount: mtp ? 4 : 0, embeddingLength: width)
    }

    static func projectorMeta(width: Int, vision: Bool = true) -> EngineGGUFMetadata {
        EngineGGUFMetadata(architecture: "clip", contextLength: nil, tensorCount: 3, nextNTensorCount: 0,
                           projector: EngineGGUFProjector(projectorType: "qwen3vl_merger",
                                                          hasVisionEncoder: vision, projectionDimension: width))
    }

    static func discover(files: [String], headers: [String: EngineGGUFMetadata],
                         scope: EngineCatalogScope = .product) -> EngineCatalog {
        let fileSystem = EngineCatalogDiscovery.FileSystem(
            childDirectories: { _ in [] },
            ggufFiles: { $0.path == directory ? files.map { URL(fileURLWithPath: $0) } : [] },
            probe: { _ in .notAModelDirectory },
            checkpointID: { _ in nil },
            readGGUF: { url in
                guard let header = headers[url.path] else { throw EngineGGUFReader.ReadError.notGGUF }
                return header
            },
            size: { $0.lastPathComponent.contains("mmproj") ? 672_423_616 : 42 })
        return EngineCatalogDiscovery.discover(roots: [], ggufDirectories: [URL(fileURLWithPath: directory)],
                                               scope: scope, fileSystem: fileSystem)
    }

    @Test func eachModelPairsTheProjectorOfItsWidth() throws {
        let quickProjector = "/lib/gguf/Qwen3.5-4B-mmproj-F16.gguf"
        let deepProjector = "/lib/gguf/mmproj-Qwen3.8-27B-BF16.gguf"
        let catalog = Self.discover(
            files: [Self.quickFile, Self.deepFile, quickProjector, deepProjector],
            headers: [Self.quickFile: Self.modelMeta(width: 2_560),
                      Self.deepFile: Self.modelMeta(width: 5_120, mtp: true),
                      quickProjector: Self.projectorMeta(width: 2_560),
                      deepProjector: Self.projectorMeta(width: 5_120)])
        #expect(catalog.models.map(\.id) == ["qwen3.5-4b", "qwen3.8-27b"])
        let quick = try #require(catalog.descriptor(for: "qwen3.5-4b"))
        #expect(quick.visionProjector == EngineVisionProjector(location: URL(fileURLWithPath: quickProjector),
                                                               sizeBytes: 672_423_616))
        #expect(catalog.descriptor(for: "qwen3.8-27b")?.visionProjector?.location.path == deepProjector)
        // A projector is neither a model nor a skip.
        #expect(!catalog.skipped.contains { $0.location.lastPathComponent.contains("mmproj") })
    }

    @Test func aProjectorOfAnotherWidthOrWithoutVisionIsNotPaired() {
        let generic = "/lib/gguf/mmproj-F16.gguf"
        let catalog = Self.discover(files: [Self.quickFile, generic],
                                    headers: [Self.quickFile: Self.modelMeta(width: 2_560),
                                              generic: Self.projectorMeta(width: 5_120)])
        #expect(catalog.descriptor(for: "qwen3.5-4b")?.visionProjector == nil)

        let audioOnly = Self.discover(files: [Self.quickFile, generic],
                                      headers: [Self.quickFile: Self.modelMeta(width: 2_560),
                                                generic: Self.projectorMeta(width: 2_560, vision: false)])
        #expect(audioOnly.descriptor(for: "qwen3.5-4b")?.visionProjector == nil)

        let unknownWidth = Self.discover(
            files: [Self.quickFile, generic],
            headers: [Self.quickFile: EngineGGUFMetadata(architecture: "qwen35", contextLength: 262_144,
                                                         tensorCount: 10, nextNTensorCount: 0),
                      generic: Self.projectorMeta(width: 2_560)])
        #expect(unknownWidth.descriptor(for: "qwen3.5-4b")?.visionProjector == nil)
    }

    @Test func theProductNameWinsAmongFittingProjectorsAndTheRepositoryNameAlsoPairs() {
        let downloaded = "/lib/gguf/Qwen3.5-4B-mmproj-F16.gguf"
        let generic = "/lib/gguf/mmproj-F16.gguf"
        let other = "/lib/gguf/a-mmproj.gguf"
        let headers = [Self.quickFile: Self.modelMeta(width: 2_560),
                       downloaded: Self.projectorMeta(width: 2_560),
                       generic: Self.projectorMeta(width: 2_560),
                       other: Self.projectorMeta(width: 2_560)]
        let all = Self.discover(files: [Self.quickFile, other, generic, downloaded], headers: headers)
        #expect(all.descriptor(for: "qwen3.5-4b")?.visionProjector?.location.path == downloaded)
        let repositoryName = Self.discover(files: [Self.quickFile, other, generic], headers: headers)
        #expect(repositoryName.descriptor(for: "qwen3.5-4b")?.visionProjector?.location.path == generic)
    }

    @Test func theDeveloperScopeNeverListsAProjectorAsAModel() {
        let projector = "/lib/gguf/mmproj-Some-Model-F16.gguf"
        let model = "/lib/gguf/Some-Model.Q8_0.gguf"
        let catalog = Self.discover(files: [model, projector],
                                    headers: [model: Self.modelMeta(width: 1_024),
                                              projector: Self.projectorMeta(width: 1_024)],
                                    scope: .all)
        #expect(catalog.models.map(\.id) == ["some-model.q8_0"])
        #expect(catalog.models.first?.visionProjector?.location.path == projector)
    }
}

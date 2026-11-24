import CoreGraphics
import CoreText
import Foundation
import Mference
import Testing
@testable import MferenceEngine

/// The real-model check of image input, on a GGUF model and its vision
/// projector. Skipped unless both are set:
///
/// - `MFERENCE_VISION_GGUF`: the model (Quick: `Qwen3.5-4B-UD-Q4_K_XL.gguf`).
/// - `MFERENCE_VISION_MMPROJ`: its projector (Quick: `mmproj-F16.gguf`).
///
/// It is a model run: follow AGENTS.md's model-run rules and run it under the
/// machine lock, alone. It checks, at greedy decoding with thinking off:
/// that a text-only request gives the same tokens with and without the
/// projector; that the model reads a number and names colours from rendered
/// pictures; and that a follow-up turn with the same picture reuses the
/// prompt through it. `MFERENCE_VISION_OUT`, if set, gets a JSON summary.
@Suite("Llama vision on an installed model", .serialized)
struct LlamaVisionInstalledTests {
    static let environment = ProcessInfo.processInfo.environment
    static let enabled = ["MFERENCE_VISION_GGUF", "MFERENCE_VISION_MMPROJ"]
        .allSatisfy { environment[$0] != nil }

    static func descriptor(gguf: URL, projector: URL?) throws -> EngineModelDescriptor {
        let metadata = try EngineGGUFReader.read(url: gguf)
        return EngineModelDescriptor(
            id: gguf.deletingPathExtension().lastPathComponent, displayName: gguf.lastPathComponent,
            engine: .llamaCpp, format: .gguf(metadata), location: gguf, sizeBytes: 0,
            contextLength: metadata.contextLength ?? 32_768, product: nil,
            visionProjector: projector.map { EngineVisionProjector(location: $0, sizeBytes: 0) })
    }

    /// A white picture with `text` in large black type.
    static func picture(text: String, width: Int = 640, height: Int = 360) throws -> EngineMedia {
        let space = CGColorSpace(name: CGColorSpace.sRGB)!
        let context = try #require(CGContext(data: nil, width: width, height: height, bitsPerComponent: 8,
                                             bytesPerRow: 0, space: space,
                                             bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue))
        context.setFillColor(CGColor(srgbRed: 1, green: 1, blue: 1, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: width, height: height))
        let font = CTFontCreateWithName("Helvetica-Bold" as CFString, 160, nil)
        let attributes: [CFString: Any] = [kCTFontAttributeName: font,
                                           kCTForegroundColorAttributeName: CGColor(gray: 0, alpha: 1)]
        let line = CTLineCreateWithAttributedString(
            CFAttributedStringCreate(nil, text as CFString, attributes as CFDictionary))
        let bounds = CTLineGetImageBounds(line, context)
        context.textPosition = CGPoint(x: (CGFloat(width) - bounds.width) / 2 - bounds.minX,
                                       y: (CGFloat(height) - bounds.height) / 2 - bounds.minY)
        CTLineDraw(line, context)
        let image = try #require(context.makeImage())
        return try #require(EngineMedia.image(image))
    }

    /// Left half red, right half blue.
    static func halves(width: Int = 512, height: Int = 512) throws -> EngineMedia {
        var rgb = Data(count: width * height * 3)
        rgb.withUnsafeMutableBytes { raw in
            let bytes = raw.bindMemory(to: UInt8.self)
            for y in 0..<height {
                for x in 0..<width {
                    let offset = (y * width + x) * 3
                    if x < width / 2 { bytes[offset] = 230 } else { bytes[offset + 2] = 230 }
                }
            }
        }
        return try #require(EngineMedia(width: width, height: height, rgb: rgb))
    }

    static func ask(_ model: any EngineModel, _ messages: [MFTokenizer.Message],
                    media: [Int: [EngineMedia]] = [:]) async throws -> (EngineCompletion, EnginePreparedPrompt) {
        let request = EngineRequest(messages: messages, generationConfig: GenerationConfig(temperature: 0),
                                    maximumCompletionTokens: 48, reasoningEffort: .off, media: media)
        let prepared = try await model.prepare(request)
        return (try await model.generate(prepared) { _ in }, prepared)
    }

    @Test(.enabled(if: enabled, "set MFERENCE_VISION_GGUF and MFERENCE_VISION_MMPROJ"))
    func seesPicturesAndLeavesTextAlone() async throws {
        let gguf = URL(fileURLWithPath: Self.environment["MFERENCE_VISION_GGUF"]!)
        let projector = URL(fileURLWithPath: Self.environment["MFERENCE_VISION_MMPROJ"]!)
        let question = [MFTokenizer.Message(role: .user, content: "Name three primary colours.")]
        var summary: [String: Any] = [:]

        // Text only, without the projector.
        let plain = try await LlamaBackend().load(try Self.descriptor(gguf: gguf, projector: nil),
                                                  options: EngineLoadOptions())
        #expect(!plain.traits.supportsVision)
        let (plainAnswer, plainPrompt) = try await Self.ask(plain, question)
        let plainTokens = await (plain as? LlamaModel)?.lastSampledTokens
        await plain.unload()

        let model = try await LlamaBackend().load(try Self.descriptor(gguf: gguf, projector: projector),
                                                  options: EngineLoadOptions())
        let outcome: Result<[String: Any], any Error>
        do {
            outcome = .success(try await Self.check(model, question: question, plainAnswer: plainAnswer,
                                                    plainPrompt: plainPrompt, plainTokens: plainTokens))
        } catch {
            outcome = .failure(error)
        }
        await model.unload()
        summary.merge(try outcome.get()) { $1 }

        if let out = Self.environment["MFERENCE_VISION_OUT"] {
            let data = try JSONSerialization.data(withJSONObject: summary, options: [.prettyPrinted, .sortedKeys])
            try data.write(to: URL(fileURLWithPath: out))
        }
    }

    static func check(_ model: any EngineModel, question: [MFTokenizer.Message], plainAnswer: EngineCompletion,
                      plainPrompt: EnginePreparedPrompt, plainTokens: [Int32]?) async throws -> [String: Any] {
        var summary: [String: Any] = [:]
        #expect(model.traits.supportsVision)

        // The same text request, with the projector: the same tokens.
        let (textAnswer, textPrompt) = try await ask(model, question)
        #expect(textPrompt.promptTokens == plainPrompt.promptTokens)
        #expect(await (model as? LlamaModel)?.lastSampledTokens == plainTokens)
        #expect(textAnswer.content == plainAnswer.content)
        summary["text_only_identical"] = textAnswer.content == plainAnswer.content
        summary["text_answer"] = textAnswer.content

        // A number in a picture.
        let number = try Self.picture(text: "742")
        let readQuestion = MFTokenizer.Message(role: .user,
                                               content: "What number is written in this image? Reply with the number only.")
        let clock = ContinuousClock()
        let start = clock.now
        let (read, readPrompt) = try await Self.ask(model, [readQuestion], media: [0: [number]])
        summary["number_answer"] = read.content
        summary["number_prompt_tokens"] = read.usage.promptTokens
        summary["number_image_tokens"] = readPrompt.promptTokens.filter { $0 == EngineMedia.placeholderToken }.count
        summary["number_seconds"] = Double((clock.now - start).components.seconds)
        #expect(read.content.contains("742"), "answer: \(read.content)")
        #expect(readPrompt.promptTokens.contains(EngineMedia.placeholderToken))
        #expect(read.usage.promptTokens == readPrompt.promptTokens.count)

        // A follow-up with the same picture reuses the prompt through it.
        let followUp = [readQuestion,
                        MFTokenizer.Message(role: .assistant, content: read.content),
                        MFTokenizer.Message(role: .user, content: "Is that number even or odd? One word.")]
        let (parity, _) = try await Self.ask(model, followUp, media: [0: [number]])
        summary["follow_up_answer"] = parity.content
        summary["follow_up_cached_tokens"] = parity.usage.cachedPromptTokens ?? -1
        #expect((parity.usage.cachedPromptTokens ?? 0) > readPrompt.promptTokens.count / 2,
                "cached \(parity.usage.cachedPromptTokens ?? -1) of \(parity.usage.promptTokens)")
        #expect(parity.content.lowercased().contains("even"), "answer: \(parity.content)")

        // Colours.
        let colours = MFTokenizer.Message(role: .user,
                                          content: "Which two colours does this image show? Name them only.")
        let (named, _) = try await Self.ask(model, [colours], media: [0: [try Self.halves()]])
        summary["colour_answer"] = named.content
        #expect(named.content.lowercased().contains("red"), "answer: \(named.content)")
        #expect(named.content.lowercased().contains("blue"), "answer: \(named.content)")

        // Text only again, after images: still the same tokens.
        let (after, _) = try await Self.ask(model, question)
        #expect(after.content == plainAnswer.content)
        summary["text_only_identical_after_images"] = after.content == plainAnswer.content
        return summary
    }
}

import AppKit
import Foundation
@testable import MferenceAppCore
import MferenceEngine
import SwiftData
import Testing
import UniformTypeIdentifiers

/// Image understanding (Phase 2), model-free: pictures reach a model that
/// sees them as pixels through the engine contract, every other model reads
/// their text as before; the add-on downloads, pairs and removes; the chat
/// offers it once.
@Suite(.serialized) struct ImageUnderstandingTests {
    static func png(width: Int = 64, height: Int = 48) throws -> Data {
        let context = try #require(CGContext(data: nil, width: width, height: height, bitsPerComponent: 8,
                                             bytesPerRow: 0, space: CGColorSpaceCreateDeviceRGB(),
                                             bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue))
        context.setFillColor(CGColor(red: 1, green: 0, blue: 0, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: width, height: height))
        let image = try #require(context.makeImage())
        let data = NSMutableData()
        let destination = try #require(CGImageDestinationCreateWithData(data, UTType.png.identifier as CFString, 1, nil))
        CGImageDestinationAddImage(destination, image, nil)
        #expect(CGImageDestinationFinalize(destination))
        return data as Data
    }

    /// The prompt size the engine reports for one turn, which the fake
    /// engine grows by `FakeEngineModel.imageTokens` per picture it gets.
    static func promptTokens(_ engine: EngineService, images: [ChatImage]) async throws -> Int {
        let turn = ChatTurn(role: .user, content: "What is this?", images: images)
        for try await update in engine.generate(model: .quick, turns: [turn], options: GenerationOptions()) {
            if case .prepared(let tokens) = update { return tokens }
        }
        Issue.record("no prepared update")
        return -1
    }

    // MARK: Engine contract

    @Test func picturesReachOnlyAModelThatSeesThem() async throws {
        let picture = ChatImage(data: try Self.png())
        let seeing = AppServices.fakeEngine(installed: [.quick], seeing: [.quick])
        let option = try #require(await seeing.models().first { $0.model == .quick })
        #expect(option.seesImages)
        #expect(option.imageUnderstandingBytes == DownloadSource.quickImageUnderstanding.sizeBytes)
        let textOnly = try await Self.promptTokens(seeing, images: [])
        #expect(try await Self.promptTokens(seeing, images: [picture]) == textOnly + FakeEngineModel.imageTokens)
        #expect(try await Self.promptTokens(seeing, images: [picture, picture])
            == textOnly + 2 * FakeEngineModel.imageTokens)
        // Bytes ImageIO cannot read are left out rather than failing the turn.
        #expect(try await Self.promptTokens(seeing, images: [ChatImage(data: Data("not a picture".utf8))]) == textOnly)
        await seeing.shutdown()

        let blind = AppServices.fakeEngine(installed: [.quick])
        #expect(await blind.models().first { $0.model == .quick }?.seesImages == false)
        #expect(try await Self.promptTokens(blind, images: [picture]) == textOnly)
        await blind.shutdown()
    }

    @Test func onlyQuickAndDeepHaveImageUnderstanding() {
        #expect(AppModel.allCases.filter(\.supportsImageUnderstanding) == [.quick, .deep])
        // A native model is never reported as seeing, whatever its catalog says.
        let options = EngineService.options(catalog: AppServices.fakeCatalog([.maple], seeing: [.maple]),
                                            status: EngineHostStatus(), physicalMemory: 64 << 30)
        #expect(options.first { $0.model == .maple }?.seesImages == false)
    }

    // MARK: What the model reads

    static let picture = ChatAttachment(filename: "sign.png", kind: .image, byteCount: 10, text: "NO PARKING",
                                        imageData: Data([1, 2, 3]))
    static let blank = ChatAttachment(filename: "cat.png", kind: .image, byteCount: 10, text: "",
                                      imageData: Data([4, 5]))

    @Test func aModelThatSeesGetsThePixelsAndTheText() {
        let seen = ChatAttachment.modelContent(message: "What does it say?", attachments: [Self.picture],
                                               budget: 1_000, seesImages: true)
        #expect(seen.contains("(An image, shown to you. The text read from it:)\nNO PARKING"))
        let blank = ChatAttachment.modelContent(message: "What is it?", attachments: [Self.blank],
                                                budget: 1_000, seesImages: true)
        #expect(blank.contains("(An image, shown to you. No text was found in it.)"))
        #expect(ChatAttachment.images(of: [Self.picture, Self.blank]).map(\.data) == [Data([1, 2, 3]), Data([4, 5])])

        // Without vision, Phase 1's wording, unchanged.
        let text = ChatAttachment.modelContent(message: "What is it?", attachments: [Self.blank], budget: 1_000)
        #expect(text.contains("(An image. No text was found in it, and this model cannot see images.)"))
        // A picture kept before Mference stored pixels is text only, even on a seeing model.
        var old = Self.blank
        old.imageData = nil
        #expect(ChatAttachment.modelContent(message: "?", attachments: [old], budget: 1_000, seesImages: true)
            .contains("this model cannot see images"))
    }

    @Test func attachmentsStoredBeforePhaseTwoStillDecode() throws {
        let json = #"[{"id":"6F9619FF-8B86-D011-B42D-00C04FC964FF","filename":"a.png","kind":"image","byteCount":3,"text":"","truncated":false}]"#
        let decoded = try JSONDecoder().decode([ChatAttachment].self, from: Data(json.utf8))
        #expect(decoded.first?.imageData == nil)
    }

    @Test @MainActor func historyTurnsCarryTheirPicturesOnlyForASeeingModel() throws {
        let container = try AppModelContainer.inMemory()
        let store = ConversationStore(container: container)
        let chat = try store.createConversation(model: .quick, webEnabled: false, isPrivate: false, now: .now)
        _ = try store.append(to: chat, role: .user, content: "What is this?", now: .now) {
            $0.attachments = [Self.picture]
        }
        _ = try store.append(to: chat, role: .assistant, content: "A sign.", now: .now) { _ in }
        _ = try store.append(to: chat, role: .user, content: "And the colour?", now: .now) { _ in }

        let seen = store.turns(for: chat, seesImages: true)
        #expect(seen.first?.images == [ChatImage(data: Data([1, 2, 3]))])
        #expect(seen.dropFirst().allSatisfy { $0.images.isEmpty })
        #expect(store.turns(for: chat).allSatisfy { $0.images.isEmpty })
    }

    @Test func aReadPictureKeepsAScaledJPEG() async throws {
        let folder = try FindTempFolder()
        let big = try FindSpeedTests.textImage(folder.url.appending(path: "big.png"), lines: ["EXIT"])
        let reader = ChatAttachmentReader(extractor: InProcessFindExtractor(timeout: .seconds(60)))
        let attachment = try await reader.read(big).get()
        let data = try #require(attachment.imageData)
        let source = try #require(CGImageSourceCreateWithData(data as CFData, nil))
        #expect(CGImageSourceGetType(source) as String? == UTType.jpeg.identifier)
        let properties = try #require(CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any])
        let width = try #require(properties[kCGImagePropertyPixelWidth] as? Int)
        #expect(width == 1_200)  // under imageSide: not enlarged

        let notes = try folder.write("notes.txt", "plain text")
        #expect(try await reader.read(notes).get().imageData == nil)
    }

    // MARK: The chat's offer

    @MainActor final class FakeOffering: ImageUnderstandingOffering {
        var hasOffered = false
        var downloads: [AppModel] = []
        func downloadBytes(for model: AppModel) -> UInt64? {
            DownloadSource.imageUnderstanding[model]?.sizeBytes
        }
        func download(_ model: AppModel) { downloads.append(model) }
    }

    @MainActor static func chat(options: [ModelOption], offering: FakeOffering) throws -> ChatViewModel {
        let service = ScriptedEngineService(options: options)
        let chat = ChatViewModel(service: service, store: ConversationStore(container: try AppModelContainer.inMemory()),
                                 attachmentReader: ChatAttachmentReader(extractor: InProcessFindExtractor()),
                                 imageUnderstanding: offering)
        chat.start()
        return chat
    }

    @Test @MainActor func theFirstPictureOnQuickOffersImageUnderstandingOnce() async throws {
        let folder = try FindTempFolder()
        let sign = try FindSpeedTests.textImage(folder.url.appending(path: "sign.png"), lines: ["STOP"])
        let notes = try folder.write("notes.txt", "text")
        let offering = FakeOffering()
        let chat = try Self.chat(options: [ModelOption(model: .quick, availability: .ready)], offering: offering)
        await waitUntil { chat.option(for: .quick) != nil && chat.modelOptions.count == 1 }
        #expect(!chat.seesImages)
        #expect(chat.imageCaption == "Images are read as text on this Mac")

        chat.attach([notes])
        #expect(chat.imageUnderstandingOffer == nil)
        chat.attach([sign])
        let offer = try #require(chat.imageUnderstandingOffer)
        #expect(offer.model == .quick)
        #expect(offer.sizeBytes == DownloadSource.quickImageUnderstanding.sizeBytes)
        #expect(offer.message.contains("672"))
        #expect(offering.hasOffered)
        chat.acceptImageUnderstandingOffer()
        #expect(offering.downloads == [.quick])
        #expect(chat.imageUnderstandingOffer == nil)

        // Never again, accepted or not.
        chat.attach([sign])
        #expect(chat.imageUnderstandingOffer == nil)
        await chat.waitForAttachments()
    }

    @Test @MainActor func noOfferWhenTheModelSeesOrNeverCan() async throws {
        let folder = try FindTempFolder()
        let sign = try FindSpeedTests.textImage(folder.url.appending(path: "sign.png"), lines: ["STOP"])

        let seeingOffering = FakeOffering()
        let seeing = try Self.chat(options: [ModelOption(model: .quick, availability: .ready, seesImages: true)],
                                   offering: seeingOffering)
        await waitUntil { seeing.seesImages }
        #expect(seeing.imageCaption == "Quick 4B can see images")
        seeing.attach([sign])
        #expect(seeing.imageUnderstandingOffer == nil)
        #expect(!seeingOffering.hasOffered)

        let mapleOffering = FakeOffering()
        let maple = try Self.chat(options: [ModelOption(model: .quick, availability: .ready),
                                            ModelOption(model: .maple, availability: .ready)],
                                  offering: mapleOffering)
        await waitUntil { maple.option(for: .maple) != nil }
        maple.selectModel(.maple)
        maple.attach([sign])
        #expect(maple.imageUnderstandingOffer == nil)

        // Quick not installed yet: nothing to add image understanding to.
        let missing = try Self.chat(options: [ModelOption(model: .quick, availability: .notInstalled(reason: "x"))],
                                    offering: FakeOffering())
        await waitUntil { missing.modelOptions.count == 1 }
        missing.attach([sign])
        #expect(missing.imageUnderstandingOffer == nil)
        await seeing.waitForAttachments()
        await maple.waitForAttachments()
        await missing.waitForAttachments()
    }

    @Test @MainActor func aSeeingModelIsSentThePicturesInEveryTurn() async throws {
        let folder = try FindTempFolder()
        let sign = try FindSpeedTests.textImage(folder.url.appending(path: "sign.png"), lines: ["STOP"])
        let service = ScriptedEngineService(
            options: [ModelOption(model: .quick, availability: .ready, seesImages: true)],
            replies: [.init(content: "A stop sign."), .init(content: "Red.")])
        let chat = ChatViewModel(service: service,
                                 store: ConversationStore(container: try AppModelContainer.inMemory()),
                                 attachmentReader: ChatAttachmentReader(extractor: InProcessFindExtractor()))
        chat.start()
        await waitUntil { chat.seesImages }
        chat.draft = "What is this?"
        chat.attach([sign])
        await chat.waitForAttachments()
        chat.send()
        await chat.waitUntilIdle()
        chat.draft = "What colour is it?"
        chat.send()
        await chat.waitUntilIdle()

        let first = try #require(service.requests.first?.turns.last { $0.role == .user })
        #expect(first.images.count == 1)
        #expect(first.content?.contains("shown to you") == true)
        let second = try #require(service.requests.last?.turns)
        #expect(second.first { $0.role == .user }?.images.count == 1)
        #expect(second.last?.images.isEmpty == true)
    }

    // MARK: Downloads and the library

    @Test func theAddOnsArePinnedBesideTheirModels() throws {
        for (model, addOn) in DownloadSource.imageUnderstanding {
            let product = try #require(DownloadSource.product[model])
            #expect(addOn.model == model)
            #expect(addOn.repository == product.repository)
            #expect(addOn.revision == product.revision)
            #expect(addOn.sha256.count == 64)
            #expect(addOn.url.path().contains("/resolve/\(addOn.revision)/\(addOn.fileName)"))
            #expect(EngineCatalogDiscovery.isProjectorFileName(addOn.localFileName))
            #expect(EngineCatalogDiscovery.productProjectorFiles[EngineProductSlot(modelID: model.modelID)!]?
                .first == addOn.localFileName)
        }
        #expect(DownloadSource.quickImageUnderstanding.localFileName == "Qwen3.5-4B-mmproj-F16.gguf")
        #expect(DownloadSource.quickImageUnderstanding.sizeBytes == 672_423_616)
        #expect(DownloadSource.deepImageUnderstanding.sizeBytes == 931_146_528)
    }

    @Test func anAddOnDownloadsUnderItsOwnNameAndCanBeRemoved() async throws {
        let body = fakeModelBytes(50_000)
        let server = try await LoopbackHTTPServer(routes: ["/p": .init(body: body)])
        let destination = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: destination) }
        let addOn = DownloadSource(model: .quick, repository: "t/t", revision: "abc", fileName: "mmproj-F16.gguf",
                                   sizeBytes: UInt64(body.count), sha256: sha256Hex(body), licence: "Apache 2.0",
                                   url: server.url("/p"), localFileName: "Qwen3.5-4B-mmproj-F16.gguf")
        let installed = InstalledLog()
        let downloads = manager([addOn], destination: destination, installed: installed)
        let url = try await downloads.download(DownloadApproval(confirming: addOn))
        #expect(url.lastPathComponent == "Qwen3.5-4B-mmproj-F16.gguf")
        #expect(await downloads.state(of: .quick) == .installed(url))

        try await downloads.remove(.quick)
        #expect(!FileManager.default.fileExists(atPath: url.path))
        #expect(await downloads.state(of: .quick) == .notStarted)
        // The engine is told twice: after the install and after the removal.
        #expect(installed.models == [.quick, .quick])
    }

    @Test @MainActor func theLibraryShowsImageUnderstandingUnderQuickAndDeep() async throws {
        let destination = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: destination) }
        let images = DownloadManager(destination: destination, sources: DownloadSource.imageUnderstanding,
                                     allowedHosts: [])
        let library = ModelLibrary(engine: AppServices.fakeEngine(installed: [.quick, .deep, .maple], seeing: [.deep]),
                                   downloads: DownloadManager(destination: destination, allowedHosts: []),
                                   imageDownloads: images, settings: AppSettings(store: InMemorySettingsStore()),
                                   memoryFit: MemoryFit(physicalMemory: 32 << 30), chipName: nil)
        await library.refresh()
        #expect(library.row(for: .quick)?.imageUnderstanding
            == ImageUnderstandingRow(state: .downloadable, sizeBytes: DownloadSource.quickImageUnderstanding.sizeBytes))
        #expect(library.row(for: .deep)?.imageUnderstanding?.isInstalled == true)
        #expect(library.row(for: .maple)?.imageUnderstanding == nil)
        #expect(library.imageUnderstandingSource(for: .quick) == DownloadSource.quickImageUnderstanding)
        #expect(library.imageUnderstandingSource(for: .maple) == nil)

        // Deep is not offered on 16 GB, and neither is its add-on.
        let small = ModelLibrary(engine: AppServices.fakeEngine(installed: [.quick]),
                                 downloads: DownloadManager(destination: destination, allowedHosts: []),
                                 imageDownloads: images, settings: AppSettings(store: InMemorySettingsStore()),
                                 memoryFit: MemoryFit(physicalMemory: 16 << 30), chipName: nil)
        await small.refresh()
        #expect(small.row(for: .deep)?.imageUnderstanding == nil)
        #expect(small.imageUnderstandingSource(for: .deep) == nil)
    }

    @Test @MainActor func theOfflineSwitchStopsAddOnDownloadsAndTheLiveOfferingRemembers() async throws {
        let settings = AppSettings(store: InMemorySettingsStore())
        let downloads = DownloadManager(destination: temporaryDirectory(), sources: DownloadSource.imageUnderstanding,
                                        allowedHosts: [])
        let offering = ImageUnderstandingDownloads(settings: settings, downloads: downloads,
                                                   memoryFit: MemoryFit(physicalMemory: 16 << 30))
        #expect(!offering.hasOffered)
        offering.hasOffered = true
        #expect(settings.hasOfferedImageUnderstanding)
        #expect(offering.downloadBytes(for: .quick) == DownloadSource.quickImageUnderstanding.sizeBytes)
        #expect(offering.downloadBytes(for: .deep) == nil)
    }
}

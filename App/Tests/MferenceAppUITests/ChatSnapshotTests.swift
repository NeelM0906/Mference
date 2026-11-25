import AppKit
import Foundation
import MferenceAppCore
import MferenceAppUI
import MferenceDesign
import SwiftData
import SwiftUI
import Testing

/// G-A3: renders the chat screen in the mockup's three states with fixture
/// data. Set `MFERENCE_APP_VISUAL_DIR` to write the PNGs (2x, named like the
/// mockup screenshots) for the visual review.
@Suite struct ChatSnapshotTests {
    static let answer = """
    Three desks under $600 have warranties worth comparing. Prices are as of this morning.

    | Desk | Price today | Warranty |
    | --- | --- | --- |
    | Stride E2 | $589 | 15 yr frame, 5 yr motor [1] |
    | Oakline Duo | $499 ~~$579~~ | 7 yr frame and motor [2] |
    | Fernway Lift | $549 | 10 yr frame, 3 yr motor [3] |

    For the longest cover, the Stride E2. For the best price, the Oakline Duo, which is on sale through Sunday [2].
    """

    static func fixture() async throws -> (ChatViewModel, ModelContainer) {
        let container = try AppModelContainer.inMemory()
        let store = ConversationStore(container: container)
        let now = Date.now
        for (offset, title) in ["Summarize Q3 board deck", "Lisbon trip, 4 days"].enumerated() {
            let other = try store.createConversation(model: .quick, now: now.addingTimeInterval(-Double(offset + 2) * 600))
            try store.update(other) { $0.title = title }
        }
        for title in ["Tax documents 2026", "Lab results questions"] {
            let other = try store.createConversation(model: .quick, isPrivate: true, now: now.addingTimeInterval(-86_400 * 3))
            try store.update(other) { $0.title = title }
        }
        let conversation = try store.createConversation(model: .quick, webEnabled: true, now: now)
        try store.append(to: conversation, role: .user,
                         content: "Which standing desk under $600 has the best warranty right now? Check current prices.",
                         now: now)
        try store.update(conversation) { $0.title = "Standing desk warranties" }
        try store.append(to: conversation, role: .assistant, content: answer, model: .quick, now: now) {
            $0.steps = [.search("standing desk under 600 warranty 2026", results: 5),
                        .search("oakline duo sale", results: 3),
                        .search("stride e2 motor warranty", results: 4)]
                + (1...7).map { .readPage(URL(string: "https://page\($0).example")!, title: nil) }
            $0.citations = [
                ChatCitation(number: 1, url: URL(string: "https://stride-desks.example/e2")!, title: "Stride E2"),
                ChatCitation(number: 2, url: URL(string: "https://oakline.example/sale")!, title: "Oakline Duo sale"),
                ChatCitation(number: 3, url: URL(string: "https://deskreviews.example/")!, title: "Desk reviews"),
            ]
            $0.finishRawValue = ChatFinish.stop.rawValue
        }
        for index in 0..<12 { container.mainContext.insert(MemoryItem(text: "Fact \(index)")) }
        try container.mainContext.save()

        let service = ScriptedEngineService(status: EngineServiceStatus(phase: .ready, residentModel: .quick))
        let model = ChatViewModel(service: service, store: store)
        model.start()
        model.select(conversation.id)
        model.setWebEnabled(true)
        for _ in 0..<200 where model.option(for: .quick)?.availability != .loaded {
            try await Task.sleep(for: .milliseconds(5))
        }
        return (model, container)
    }

    /// The mockup draws a 1200x748 window 40/36 pt inside the 1280x820
    /// canvas, under a toolbar about 56 pt tall. Until U1's shell merges, the
    /// chat is framed the same way (blank toolbar) so the PNGs line up with
    /// the mockup's.
    static func mockupFrame(_ view: some View) -> some View {
        VStack(spacing: 0) {
            Rectangle().fill(MFColor.surface).frame(height: 56)
            Rectangle().fill(MFColor.hairlineFaint).frame(height: 1)
            view
        }
        .frame(width: 1200, height: 748)
        .clipShape(RoundedRectangle(cornerRadius: MFRadius.l + 2))
        .shadow(color: .black.opacity(0.15), radius: 30, y: 20)
        .frame(width: MFLayout.mockupWindow.width, height: MFLayout.mockupWindow.height)
        .background(MFColor.wallpaper)
    }

    static func render(_ view: some View, name: String) throws -> NSImage {
        let size = MFLayout.mockupWindow
        let content = mockupFrame(view).environment(\.colorScheme, .light)
        let hosting = NSHostingView(rootView: content)
        hosting.frame = CGRect(origin: .zero, size: size)
        let window = NSWindow(contentRect: hosting.frame, styleMask: [.borderless], backing: .buffered, defer: false)
        window.contentView = hosting
        window.appearance = NSAppearance(named: .aqua)
        hosting.layoutSubtreeIfNeeded()
        let rep = try #require(hosting.bitmapImageRepForCachingDisplay(in: hosting.bounds))
        rep.size = size
        hosting.cacheDisplay(in: hosting.bounds, to: rep)
        let image = NSImage(size: size)
        image.addRepresentation(rep)

        if let dir = ProcessInfo.processInfo.environment["MFERENCE_APP_VISUAL_DIR"] {
            // 2x like the mockup PNGs.
            let hiDPI = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: Int(size.width * 2),
                                         pixelsHigh: Int(size.height * 2), bitsPerSample: 8, samplesPerPixel: 4,
                                         hasAlpha: true, isPlanar: false, colorSpaceName: .deviceRGB,
                                         bytesPerRow: 0, bitsPerPixel: 0)
            if let hiDPI {
                hiDPI.size = size
                hosting.cacheDisplay(in: hosting.bounds, to: hiDPI)
                let url = URL(fileURLWithPath: dir).appending(path: "\(name).png")
                try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
                try hiDPI.representation(using: .png, properties: [:])?.write(to: url)
            }
        }
        return image
    }

    @Test func chatWithWebAnswer() async throws {
        let (model, _) = try await Self.fixture()
        let image = try Self.render(ChatView(model: model), name: "02-chat-web")
        #expect(image.size == MFLayout.mockupWindow)
        #expect(model.messages.count == 2)
        #expect(model.messages.last?.citations.count == 3)
    }

    @Test func chatModelPicker() async throws {
        let (model, _) = try await Self.fixture()
        _ = try Self.render(ChatView(model: model, presentation: ChatPresentation(pickerOpen: true)),
                                  name: "02b-chat-model-picker")
    }

    @Test func chatStepsExpanded() async throws {
        let (model, _) = try await Self.fixture()
        _ = try Self.render(ChatView(model: model, presentation: ChatPresentation(stepsExpanded: true)),
                                  name: "02c-chat-steps")
    }

    /// A message sent with two files, and the composer holding a file being
    /// read, a ready one and one that can't be attached.
    @Test @MainActor func chatWithAttachments() async throws {
        let container = try AppModelContainer.inMemory()
        let store = ConversationStore(container: container)
        let conversation = try store.createConversation(model: .quick, isPrivate: true)
        try store.append(to: conversation, role: .user, content: "What does the lease say about pets?") {
            $0.attachments = [
                ChatAttachment(filename: "Lease agreement 2026.pdf", kind: .document, byteCount: 482_000,
                               text: "14. ANIMALS. Tenant may keep one cat.", pageCount: 12),
                ChatAttachment(filename: "IMG_2041.HEIC", kind: .image, byteCount: 2_100_000, text: ""),
            ]
        }
        try store.append(to: conversation, role: .assistant, content: "One cat, with a deposit.", model: .quick) {
            $0.finishRawValue = ChatFinish.stop.rawValue
        }
        let folder = FileManager.default.temporaryDirectory.appending(path: "chat-snapshot-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }
        let notes = folder.appending(path: "Move-in checklist.md")
        try Data("Keys, meter readings, photos of the walls.".utf8).write(to: notes)
        let deck = folder.appending(path: "Budget.numbers")
        try Data("zip".utf8).write(to: deck)

        let model = ChatViewModel(service: ScriptedEngineService(), store: store)
        model.select(conversation.id)
        model.attach([notes, deck])
        await model.waitForAttachments()
        let pending = model.pendingAttachments
        #expect(pending.count == 2)
        #expect(pending.last?.state == .failed("Export it as PDF to attach it"))
        let messages = model.messages
        #expect(messages.first?.attachments.count == 2)
        _ = try Self.render(ChatView(model: model), name: "02d-chat-attachments")
    }

    @Test func emptyChatRendersWithImageRenderer() throws {
        let container = try AppModelContainer.inMemory()
        let model = ChatViewModel(service: ScriptedEngineService(), store: ConversationStore(container: container))
        let renderer = ImageRenderer(content: ChatView(model: model).frame(width: MFLayout.mockupWindow.width, height: MFLayout.mockupWindow.height))
        #expect(renderer.nsImage != nil)
    }
}

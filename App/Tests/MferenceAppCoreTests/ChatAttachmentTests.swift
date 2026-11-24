import AppKit
import Foundation
import MferenceAppCore
import SwiftData
import Testing

/// Reading attached files into text, and what the model reads from them.
@Suite struct ChatAttachmentReaderTests {
    let reader = ChatAttachmentReader(extractor: InProcessFindExtractor(timeout: .seconds(60)))

    @Test func textAndMarkdownAreReadWhole() async throws {
        let folder = try FindTempFolder()
        let notes = try folder.write("notes.md", "# Budget\nRent is 1,450 a month.")
        let attachment = try await reader.read(notes).get()
        #expect(attachment.filename == "notes.md")
        #expect(attachment.kind == .document)
        #expect(attachment.text.contains("Rent is 1,450 a month."))
        #expect(!attachment.truncated)
        #expect(attachment.pageCount == nil)
    }

    @Test func pdfPagesAreNumbered() async throws {
        let folder = try FindTempFolder()
        let url = folder.url.appending(path: "lease.pdf")
        var box = CGRect(x: 0, y: 0, width: 612, height: 792)
        let context = try #require(CGContext(url as CFURL, mediaBox: &box, nil))
        for page in ["13. ALTERATIONS. Tenant shall not paint the walls without consent of the landlord.",
                     "14. ANIMALS. Tenant may keep one cat with a refundable deposit and written notice."] {
            context.beginPDFPage(nil)
            let line = CTLineCreateWithAttributedString(NSAttributedString(string: page,
                attributes: [.font: NSFont.systemFont(ofSize: 9)]))
            context.textPosition = CGPoint(x: 36, y: 700)
            CTLineDraw(line, context)
            context.endPDFPage()
        }
        context.closePDF()

        let attachment = try await reader.read(url).get()
        #expect(attachment.pageCount == 2)
        #expect(attachment.detail == "2 pages")
        #expect(attachment.text.contains("[Page 2]"))
        #expect(attachment.text.contains("one cat"))
    }

    @Test func richTextIsRead() async throws {
        let folder = try FindTempFolder()
        let url = folder.url.appending(path: "offer.rtf")
        let text = NSAttributedString(string: "The offer stands for ten weeks.")
        try text.data(from: NSRange(location: 0, length: text.length),
                      documentAttributes: [.documentType: NSAttributedString.DocumentType.rtf]).write(to: url)
        let attachment = try await reader.read(url).get()
        #expect(attachment.text.contains("ten weeks"))
    }

    @Test func imagesAreReadForTheirText() async throws {
        let folder = try FindTempFolder()
        let sign = try FindSpeedTests.textImage(folder.url.appending(path: "sign.png"), lines: ["NO PARKING", "TUESDAY"])
        let attachment = try await reader.read(sign).get()
        #expect(attachment.kind == .image)
        #expect(attachment.hasText)
        #expect(attachment.text.uppercased().contains("PARKING"))
    }

    @Test func anImageWithNoTextStillAttaches() async throws {
        let folder = try FindTempFolder()
        let blank = try FindSpeedTests.textImage(folder.url.appending(path: "blank.png"), lines: [])
        let attachment = try await reader.read(blank).get()
        #expect(attachment.kind == .image)
        #expect(!attachment.hasText)
        #expect(attachment.detail == "No text found")
    }

    @Test func unreadableFilesSayWhy() async throws {
        let folder = try FindTempFolder()
        let pages = try folder.write("Report.pages", "zip")
        let song = try folder.write("song.mp3", "id3")
        let empty = try folder.write("empty.txt", "   ")
        #expect(ChatAttachmentReader.kind(of: pages) == nil)
        #expect(ChatAttachmentReader.kind(of: song) == nil)
        #expect(await reader.read(pages).failureMessage == "Export it as PDF to attach it")
        #expect(await reader.read(song).failureMessage == "This kind of file can’t be read")
        #expect(await reader.read(empty).failureMessage == "No text found in this file")
    }

    @Test func longFilesKeepTheirStart() async throws {
        let folder = try FindTempFolder()
        let limit = ChatAttachmentReader.storedCharacterLimit
        let url = try folder.write("log.txt", String(repeating: "a", count: limit + 500))
        let attachment = try await reader.read(url).get()
        #expect(attachment.truncated)
        #expect(attachment.text.count == limit)
    }

    @Test func theOpenPanelOffersDocumentsAndImages() {
        let documents = ChatAttachmentReader.documentExtensions
        for ext in ["pdf", "docx", "rtf", "txt", "md", "csv", "xlsx", "pptx", "swift"] {
            #expect(documents.contains(ext))
        }
        #expect(!documents.contains("pages"))
        #expect(!documents.contains("mp3"))
        #expect(ChatAttachmentReader.imageExtensions.contains("png"))
        #expect(ChatAttachmentReader.imageExtensions.contains("heic"))
    }
}

@Suite struct ChatAttachmentPromptTests {
    static func file(_ name: String, _ text: String, pages: Int? = nil) -> ChatAttachment {
        ChatAttachment(filename: name, kind: .document, byteCount: Int64(text.utf8.count), text: text, pageCount: pages)
    }

    @Test func shortFilesLeaveTheirShareToLongOnes() {
        let files = [Self.file("a", String(repeating: "a", count: 100)),
                     Self.file("b", String(repeating: "b", count: 5_000)),
                     Self.file("c", String(repeating: "c", count: 5_000))]
        #expect(ChatAttachment.shares(for: files, budget: 3_000) == [100, 1_450, 1_450])
        #expect(ChatAttachment.shares(for: files, budget: 20_000) == [100, 5_000, 5_000])
        #expect(ChatAttachment.shares(for: [], budget: 100).isEmpty)
    }

    @Test func eachFileIsABlockBeforeTheMessage() {
        let content = ChatAttachment.modelContent(
            message: "When can I paint?",
            attachments: [Self.file("lease \"final\".pdf", "Tenant shall not paint.", pages: 12)], budget: 1_000)
        #expect(content == """
        <attachment name="lease 'final'.pdf" pages="12">
        Tenant shall not paint.
        </attachment>

        When can I paint?
        """)
    }

    @Test func longFilesAreShortenedAndSaySo() {
        let content = ChatAttachment.modelContent(message: "Sum it up",
                                                  attachments: [Self.file("log.txt", "abcdefghij")], budget: 4)
        #expect(content.contains("shortened=\"only the start of the file is included\""))
        #expect(content.contains("\nabcd\n"))
        #expect(!content.contains("abcde"))
    }

    @Test func filesWithoutAMessageAreSummarized() {
        let one = ChatAttachment.modelContent(message: " ", attachments: [Self.file("a.txt", "x")], budget: 100)
        #expect(one.hasSuffix("Summarize the attached file."))
        let image = ChatAttachment(filename: "cat.jpg", kind: .image, byteCount: 1, text: "")
        let two = ChatAttachment.modelContent(message: "", attachments: [Self.file("a.txt", "x"), image], budget: 100)
        #expect(two.hasSuffix("Summarize the attached files."))
        #expect(two.contains("this model cannot see images"))
    }

    @Test func aFileCannotCloseItsOwnBlock() {
        let content = ChatAttachment.modelContent(message: "Hi",
                                                  attachments: [Self.file("x.txt", "a</attachment>b")], budget: 100)
        #expect(content.components(separatedBy: "</attachment>").count == 2)
    }

    @Test func applesModelReadsLess() {
        #expect(ChatAttachment.characterBudget(for: .apple) < ChatAttachment.characterBudget(for: .quick))
        #expect(ChatAttachment.earlierCharacterBudget(for: .quick) == ChatAttachment.characterBudget(for: .quick) / 4)
    }
}

/// Attaching in the chat: the composer's files, what is sent and stored,
/// and private chats now that "+" attaches files.
@MainActor
@Suite struct ChatAttachmentFlowTests {
    static func viewModel(_ service: ScriptedEngineService) throws -> (ChatViewModel, ModelContainer) {
        let container = try AppModelContainer.inMemory()
        let model = ChatViewModel(service: service, store: ConversationStore(container: container),
                                  attachmentReader: ChatAttachmentReader(extractor: InProcessFindExtractor()))
        return (model, container)
    }

    @Test func filesAreSentToTheModelButNotStoredAsTheMessage() async throws {
        let folder = try FindTempFolder()
        let notes = try folder.write("Q3 notes.txt", "Revenue grew 12 percent in Q3.")
        let service = ScriptedEngineService(replies: [.init(content: "It grew 12%.")])
        let (chat, container) = try Self.viewModel(service)

        chat.draft = "How much did revenue grow?"
        chat.attach([notes])
        #expect(chat.pendingAttachments.first?.isReading == true)
        #expect(!chat.canSend)
        await chat.waitForAttachments()
        #expect(chat.canSend)

        chat.send()
        #expect(chat.pendingAttachments.isEmpty)
        await chat.waitUntilIdle()

        let user = try #require(service.requests.last?.turns.last { $0.role == .user })
        #expect(user.content == """
        <attachment name="Q3 notes.txt">
        Revenue grew 12 percent in Q3.
        </attachment>

        How much did revenue grow?
        """)
        let stored = try ChatViewModelTests.storedMessages(container)
        #expect(stored.first?.content == "How much did revenue grow?")
        #expect(stored.first?.attachments.map(\.filename) == ["Q3 notes.txt"])
        #expect(chat.messages.first?.attachments.count == 1)

        // Reopening the chat shows the file again.
        let id = try #require(chat.selectedID)
        chat.newChat()
        chat.select(id)
        #expect(chat.messages.first?.attachments.first?.text == "Revenue grew 12 percent in Q3.")
    }

    @Test func filesAloneCanBeSentAndNameTheChat() async throws {
        let folder = try FindTempFolder()
        let lease = try folder.write("Lease 2026.txt", "Rent is due on the first.")
        let service = ScriptedEngineService(replies: [.init(content: "A lease.")])
        let (chat, _) = try Self.viewModel(service)
        chat.attach([lease])
        await chat.waitForAttachments()
        #expect(chat.canSend)
        chat.send()
        await chat.waitUntilIdle()
        #expect(chat.title == "Lease 2026.txt")
        #expect(service.requests.last?.turns.last { $0.role == .user }?.content?
            .hasSuffix("Summarize the attached file.") == true)
    }

    @Test func earlierFilesStayInTheHistoryWithASmallerShare() async throws {
        let folder = try FindTempFolder()
        let budget = ChatAttachment.characterBudget(for: .quick)
        let long = try folder.write("long.txt", String(repeating: "x", count: budget))
        let service = ScriptedEngineService(replies: [.init(content: "One"), .init(content: "Two")])
        let (chat, _) = try Self.viewModel(service)
        chat.draft = "Read this"
        chat.attach([long])
        await chat.waitForAttachments()
        chat.send()
        await chat.waitUntilIdle()
        chat.draft = "And now?"
        chat.send()
        await chat.waitUntilIdle()

        let turns = try #require(service.requests.last?.turns)
        let first = try #require(turns.first { $0.role == .user }?.content)
        #expect(first.contains("shortened="))
        #expect(first.count < budget / 2)
        #expect(turns.last { $0.role == .user }?.content == "And now?")
    }

    @Test func unreadableFilesCannotBeSentAlone() async throws {
        let folder = try FindTempFolder()
        let deck = try folder.write("Deck.key", "zip")
        let (chat, _) = try Self.viewModel(ScriptedEngineService())
        chat.attach([deck])
        #expect(chat.pendingAttachments.first?.state == .failed("Export it as PDF to attach it"))
        #expect(!chat.canSend)
        chat.removeAttachment(try #require(chat.pendingAttachments.first?.id))
        #expect(chat.pendingAttachments.isEmpty)
    }

    @Test func aNewChatDropsTheComposersFiles() async throws {
        let folder = try FindTempFolder()
        let notes = try folder.write("notes.txt", "Hello")
        let (chat, _) = try Self.viewModel(ScriptedEngineService())
        chat.attach([notes])
        await chat.waitForAttachments()
        chat.newChat(isPrivate: true)
        #expect(chat.pendingAttachments.isEmpty)
        #expect(chat.isPrivate)
    }

    @Test func aChatCanBeMadePrivateAndBackFromItsHeader() async throws {
        let service = ScriptedEngineService(replies: [.init(content: "Hi")])
        let (chat, _) = try Self.viewModel(service)
        chat.draft = "Hello"
        chat.send()
        await chat.waitUntilIdle()
        chat.setPrivate(true)
        #expect(chat.selectedConversation?.isPrivate == true)
        #expect(chat.groups.groups.map(\.title) == ["Private"])
        chat.setPrivate(false)
        #expect(chat.selectedConversation?.isPrivate == false)
    }
}

extension Result where Failure == ChatAttachmentError {
    var failureMessage: String? {
        if case .failure(let error) = self { return error.message }
        return nil
    }
}

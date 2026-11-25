import AppKit
import Foundation
import MferenceAppCore
import Testing

/// ⌘V in the chat composer: what the clipboard attaches, and what stays text.
/// Each test uses its own named pasteboard, never the user's clipboard.
@MainActor
@Suite struct ChatPasteTests {
    private func withPasteboard(_ body: (NSPasteboard) throws -> Void) rethrows {
        let pasteboard = NSPasteboard(name: NSPasteboard.Name("ai.mference.tests.paste.\(UUID().uuidString)"))
        defer { pasteboard.releaseGlobally() }
        pasteboard.clearContents()
        try body(pasteboard)
    }

    @Test func aFileCopiedInFinderAttachesNotItsName() throws {
        let folder = try FindTempFolder()
        let file = try folder.write("6-math.txt", "23 x 4")
        withPasteboard { pasteboard in
            // Finder puts the file's URL and its name as text.
            pasteboard.writeObjects([file as NSURL, file.lastPathComponent as NSString])
            #expect(ChatPasteContent.read(pasteboard) == .files([file]))
        }
    }

    @Test func pictureDataAttaches() throws {
        let png = Data([0x89, 0x50, 0x4E, 0x47])
        withPasteboard { pasteboard in
            pasteboard.setData(png, forType: .png)
            #expect(ChatPasteContent.read(pasteboard) == .image(png, fileExtension: "png"))
        }
        let tiff = Data([0x4D, 0x4D, 0x00, 0x2A])
        withPasteboard { pasteboard in
            pasteboard.setData(tiff, forType: .tiff)
            #expect(ChatPasteContent.read(pasteboard) == .image(tiff, fileExtension: "tiff"))
        }
    }

    @Test func textStaysWithTheField() {
        withPasteboard { pasteboard in
            pasteboard.setString("Plan my week", forType: .string)
            #expect(ChatPasteContent.read(pasteboard) == nil)
        }
        // Text copied with a picture beside it (a slide, a rich page) is text.
        withPasteboard { pasteboard in
            pasteboard.declareTypes([.string, .png], owner: nil)
            pasteboard.setString("Quarterly results", forType: .string)
            pasteboard.setData(Data([0x89, 0x50]), forType: .png)
            #expect(ChatPasteContent.read(pasteboard) == nil)
        }
        withPasteboard { pasteboard in
            #expect(ChatPasteContent.read(pasteboard) == nil)
        }
    }

    @Test func aPastedPictureIsKeptAndReadLikeAnAttachedOne() async throws {
        let folder = try FindTempFolder()
        let sign = try FindSpeedTests.textImage(folder.url.appending(path: "sign.png"), lines: ["PLATFORM 9"])
        let (chat, _) = try ChatAttachmentFlowTests.viewModel(ScriptedEngineService())

        chat.paste(.image(try Data(contentsOf: sign), fileExtension: "png"))
        let pending = try #require(chat.pendingAttachments.first)
        #expect(pending.kind == .image)
        #expect(pending.filename.hasPrefix("Pasted image "))
        #expect(pending.filename.hasSuffix(".png"))
        await chat.waitForAttachments()
        let attachment = try #require(chat.pendingAttachments.first?.attachment)
        #expect(attachment.kind == .image)
        #expect(attachment.imageData != nil)
        #expect(attachment.text.uppercased().contains("PLATFORM"))
    }

    @Test func pastedFilesAttachLikeDroppedOnes() async throws {
        let folder = try FindTempFolder()
        let notes = try folder.write("notes.txt", "Rent is due on the first.")
        let (chat, _) = try ChatAttachmentFlowTests.viewModel(ScriptedEngineService())
        chat.paste(.files([notes]))
        await chat.waitForAttachments()
        #expect(chat.pendingAttachments.map(\.filename) == ["notes.txt"])
        #expect(chat.pendingAttachments.first?.attachment?.text == "Rent is due on the first.")
    }

    /// A pasted picture's bytes live only until it is read; the attachment
    /// keeps its own scaled copy. A file attached from disk is left alone.
    @Test func aPastedPictureIsDeletedOnceRead() async throws {
        let folder = try FindTempFolder()
        let sign = try FindSpeedTests.textImage(folder.url.appending(path: "sign.png"), lines: ["EXIT"])
        let (chat, _) = try ChatAttachmentFlowTests.viewModel(ScriptedEngineService())
        chat.paste(.image(try Data(contentsOf: sign), fileExtension: "png"))
        chat.attach([sign])
        await chat.waitForAttachments()

        let pasted = try #require(chat.pendingAttachments.first?.attachment)
        #expect(pasted.filename.hasPrefix("Pasted image "))
        #expect(pasted.imageData != nil)
        let leftovers = try FileManager.default.contentsOfDirectory(
            at: URL.temporaryDirectory, includingPropertiesForKeys: nil)
            .filter { $0.lastPathComponent.hasPrefix("mference-pasted-") }
            .flatMap { (try? FileManager.default.contentsOfDirectory(atPath: $0.path)) ?? [] }
        #expect(!leftovers.contains(pasted.filename))
        #expect(FileManager.default.fileExists(atPath: sign.path))
    }
}

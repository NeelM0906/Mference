import Foundation
@testable import MferenceAppCore
import Testing

/// Pictures to Apple's on-device model: the row says it sees when macOS says
/// so, the chat then sends a message's pictures, and the context split keeps
/// them with their turns. The live case runs only when asked.
@MainActor
@Suite struct AppleChatImageTests {
    @Test func theAppleRowSeesWhenTheModelDoes() async {
        let (seeing, _, _, _) = AppleChatTests.routing(info: AppleModelInfo(state: .ready, seesImages: true))
        #expect(await seeing.models().first { $0.model == .apple }?.seesImages == true)
        let (blind, _, _, _) = AppleChatTests.routing(info: AppleModelInfo(state: .ready))
        #expect(await blind.models().first { $0.model == .apple }?.seesImages == false)
    }

    @Test func theSplitKeepsPicturesWithTheirTurns() throws {
        let old = ChatImage(data: Data([1])), new = ChatImage(data: Data([2]))
        let plan = try #require(AppleChatContext.split([
            ChatTurn(role: .system, content: "Be brief."),
            ChatTurn(role: .user, content: "What is this?", images: [old]),
            ChatTurn(role: .assistant, content: "A sign."),
            ChatTurn(role: .user, content: "And this one?", images: [new]),
        ]))
        #expect(plan.history.first?.images == [old])
        #expect(plan.promptImages == [new])
        #expect(plan.prompt == "And this one?")
        #expect(try #require(AppleChatContext.split([ChatTurn(role: .user, content: "Hi")])).promptImages.isEmpty)
    }

    @Test func aChatOnAppleSendsPicturesOnlyWhenItSees() async throws {
        let folder = try FindTempFolder()
        let sign = try FindSpeedTests.textImage(folder.url.appending(path: "sign.png"), lines: ["STOP"])
        for sees in [true, false] {
            let (routing, engine, apple, _) = AppleChatTests.routing(
                info: AppleModelInfo(state: .ready, seesImages: sees),
                answer: .success(AppleChatTests.summary("A stop sign.")))
            let container = try AppModelContainer.inMemory()
            let chat = ChatViewModel(service: routing, store: ConversationStore(container: container),
                                     attachmentReader: ChatAttachmentReader(extractor: InProcessFindExtractor()))
            chat.start()
            await waitUntil { chat.option(for: .apple) != nil }
            chat.selectModel(.apple)
            #expect(chat.seesImages == sees)
            #expect(chat.imageCaption.hasSuffix(sees ? "can see images" : "read as text on this Mac"))
            chat.draft = "What is this?"
            chat.attach([sign])
            // Apple's model needs no download: never an offer.
            #expect(chat.imageUnderstandingOffer == nil)
            await chat.waitForAttachments()
            chat.send()
            await chat.waitUntilIdle()
            #expect(engine.requests.isEmpty)
            let user = try #require(apple.requests.first?.last { $0.role == .user })
            #expect(user.images.count == (sees ? 1 : 0))
            // Only a model that sees is told the picture is shown to it.
            #expect(user.content?.contains("shown to you") == sees, "sees \(sees): \(user.content ?? "nil")")
            chat.stopObserving()
        }
    }

    // MARK: Live

    /// `MFERENCE_APPLE_VISION_DIR=<dir>` with `cases.tsv` (picture, question,
    /// expected patterns joined by ` && `) and `images/`: each question
    /// through the app's `FoundationChatModel`, then a follow-up with the
    /// picture in the history. Loads no Mference model.
    @Test(.enabled(if: ProcessInfo.processInfo.environment["MFERENCE_APPLE_VISION_DIR"] != nil))
    func liveAppleAnswersAboutPictures() async throws {
        let dir = URL(fileURLWithPath: ProcessInfo.processInfo.environment["MFERENCE_APPLE_VISION_DIR"]!)
        let model = FoundationChatModel()
        let info = await model.info()
        try #require(info.state == .ready, "Apple Intelligence is not ready on this Mac")
        try #require(info.seesImages, "Apple's model on this Mac does not see images")

        func answer(_ turns: [ChatTurn]) async throws -> String {
            var text = ""
            for try await update in model.generate(turns: turns, options: GenerationOptions(maximumCompletionTokens: 64,
                                                                                             temperature: 0)) {
                if case .completed(let summary) = update { text = summary.content }
            }
            return text.replacingOccurrences(of: "\n", with: " ")
        }

        var report: [String] = []
        var passed = 0, total = 0
        let table = try String(contentsOf: dir.appending(path: "cases.tsv"), encoding: .utf8)
        for line in table.split(separator: "\n") {
            let fields = line.split(separator: "\t").map(String.init)
            let picture = ChatImage(data: try Data(contentsOf: dir.appending(path: "images/\(fields[0])")))
            let reply = try await answer([ChatTurn(role: .user, content: fields[1], images: [picture])])
            let ok = fields[2].components(separatedBy: " && ").allSatisfy {
                reply.range(of: $0, options: [.regularExpression, .caseInsensitive]) != nil
            }
            total += 1
            if ok { passed += 1 }
            report.append("\(ok ? "PASS" : "FAIL")\t\(fields[0])\t\(reply)")
        }

        // A follow-up turn about the first picture, which stays in the history.
        let first = try #require(table.split(separator: "\n").first?.split(separator: "\t").map(String.init))
        let sign = ChatImage(data: try Data(contentsOf: dir.appending(path: "images/\(first[0])")))
        let followUp = try await answer([
            ChatTurn(role: .user, content: first[1], images: [sign]),
            ChatTurn(role: .assistant, content: "14:35"),
            ChatTurn(role: .user, content: "Which platform does it leave from? Reply with the number only."),
        ])
        report.append("\(followUp.contains("9") ? "PASS" : "FAIL")\tfollow-up platform\t\(followUp)")
        print("live apple vision (\(info.labels.detail)): \(passed)/\(total)\n" + report.joined(separator: "\n"))
        #expect(passed >= total - 1, "\(report)")
        #expect(followUp.contains("9"))
    }
}

import Foundation
@testable import MferenceAppCore
import MferenceEngine
import Testing

/// Answers each generation with the next scripted answer (the last one
/// repeats), once `gate` opens, and records the person's prompt.
final class ScriptedComposeEngine: EngineServicing, @unchecked Sendable {
    let gate: ScanTestLatch?
    private let lock = NSLock()
    private var answers: [String]
    private var recorded: [String] = []

    init(answers: [String], gate: ScanTestLatch? = nil) {
        self.answers = answers
        self.gate = gate
    }

    var prompts: [String] { lock.withLock { recorded } }

    func models() async -> [ModelOption] { [ModelOption(model: .quick, availability: .ready)] }
    func refreshCatalog() async {}
    var status: EngineServiceStatus { .empty }
    func statusUpdates() -> AsyncStream<EngineServiceStatus> { AsyncStream { $0.finish() } }
    func load(_ model: AppModel) async throws {}
    func unload() async throws {}
    func generate(model: AppModel, turns: [ChatTurn], options: MferenceAppCore.GenerationOptions) -> GenerationStream {
        let answer: String = lock.withLock {
            recorded.append(turns.last?.content ?? "")
            return answers.count > 1 ? answers.removeFirst() : (answers.first ?? "")
        }
        let gate = gate
        return GenerationStream { continuation in
            Task {
                await gate?.wait()
                continuation.yield(.content(answer))
                continuation.yield(.completed(GenerationSummary(content: answer, reasoning: nil, toolCalls: [],
                                                                finish: .stop, promptTokens: 1, completionTokens: 1)))
                continuation.finish()
            }
        }
    }
    func cancelActiveGeneration() async {}
    func shutdown() async {}
}

@Suite struct MailComposerTests {
    @MainActor static func inbox(engine: (any EngineServicing)?, sender: any MailSending = RecordingMailSender())
        -> MailInboxModel {
        let empty = URL.temporaryDirectory.appending(path: "no-mail-\(UUID().uuidString)")
        return MailInboxModel(source: AppleMailSource(mailRoot: empty),
                              assistant: AppleModelService(primary: ScriptedAssistantTextModel(kind: .appleOnDevice,
                                                                                               answer: "{}"),
                                                           fallback: nil),
                              drafter: nil, composeWriter: engine.map { MailComposeWriter(engine: $0) },
                              sender: sender, broker: RecordingPermissionBroker())
    }

    @Test @MainActor func oneDraftOpensMinimizesAndKeepsItsText() {
        let model = Self.inbox(engine: nil)
        let composer = model.composer
        #expect(!composer.isOpen)
        #expect(!model.isEditingText)
        composer.open()
        #expect(composer.isOpen && !composer.isMinimized)
        // Expanded: bare letters type text, so C is off.
        #expect(model.isEditingText)
        composer.to = "a@example.com"
        composer.body = "Notes"
        composer.minimize()
        // Minimized: C is free again and brings the same draft back.
        #expect(!model.isEditingText)
        composer.open()
        #expect(!composer.isMinimized)
        #expect(composer.to == "a@example.com" && composer.body == "Notes")
        #expect(composer.hasContent)
        composer.discard()
        #expect(!composer.isOpen && composer.to.isEmpty && composer.body.isEmpty && !composer.hasContent)
    }

    @Test @MainActor func sendsOnlyOnSendAndClosesWhenSent() async {
        let sender = RecordingMailSender()
        let model = Self.inbox(engine: nil, sender: sender)
        let composer = model.composer
        composer.open()
        composer.to = "nobody"
        composer.body = "Hello"
        #expect(await composer.send() != nil)
        #expect(composer.isOpen)
        #expect(composer.sendState == .failed(MailSendError.invalidRecipient.userMessage))
        #expect(await sender.sent.isEmpty)
        composer.to = " a@example.com "
        composer.subject = "Hi"
        #expect(await composer.send() == nil)
        #expect(await sender.sent == [MailDraft(to: "a@example.com", subject: "Hi", body: "Hello")])
        #expect(!composer.isOpen && composer.body.isEmpty)
    }

    @Test @MainActor func draftsFromThePromptAndUndoKeepsThePrompt() async {
        let engine = ScriptedComposeEngine(answers: ["Hi Sam, I can't make Friday. Next week?"])
        let composer = Self.inbox(engine: engine).composer
        composer.open()
        #expect(!composer.canDraft)  // no prompt yet
        composer.prompt = "decline friday politely, suggest next week"
        composer.body = "cant do fri"
        #expect(composer.canDraft)
        composer.run(.draft)
        #expect(composer.isLocked)
        await composer.waitForAssist()
        #expect(composer.body == "Hi Sam, I can't make Friday. Next week?")
        #expect(composer.assistState == .idle)
        let prompt = engine.prompts.first ?? ""
        #expect(prompt.contains("decline friday politely, suggest next week"))
        // The person's notes go in as data to work on.
        #expect(prompt.contains("<untrusted>\ncant do fri\n</untrusted>"))
        composer.undoAssist()
        #expect(composer.body == "cant do fri")
        #expect(composer.prompt == "decline friday politely, suggest next week")
        #expect(!composer.canUndo)
    }

    @Test @MainActor func rewritesNeedABodyAndUseTheirTone() async {
        let engine = ScriptedComposeEngine(answers: ["Dear Sam, I am unable to attend on Friday."])
        let composer = Self.inbox(engine: engine).composer
        composer.open()
        composer.run(.rewrite(.formal))
        composer.run(.fixGrammar)
        composer.run(.suggestSubject)
        #expect(composer.assistState == .idle)
        #expect(engine.prompts.isEmpty)
        composer.body = "cant make friday sorry"
        composer.run(.rewrite(.formal))
        await composer.waitForAssist()
        #expect(composer.body == "Dear Sam, I am unable to attend on Friday.")
        #expect(engine.prompts.first?.contains(DraftTone.formal.instruction) == true)
    }

    @Test @MainActor func aSuggestedSubjectReplacesTheTypedOneAndUndoRestoresIt() async {
        let engine = ScriptedComposeEngine(answers: ["Subject: \"Friday plans\"\nextra line"])
        let composer = Self.inbox(engine: engine).composer
        composer.open()
        composer.subject = "hey"
        composer.body = "Can we move Friday's lunch?"
        composer.run(.suggestSubject)
        await composer.waitForAssist()
        #expect(composer.subject == "Friday plans")
        #expect(composer.body == "Can we move Friday's lunch?")
        composer.undoAssist()
        #expect(composer.subject == "hey")
    }

    @Test @MainActor func aMissingQuickSaysSo() async {
        let composer = Self.inbox(engine: AppleModelServiceTests.quickEngine(answer: "unused", installed: false)).composer
        composer.open()
        composer.body = "Notes"
        composer.run(.fixGrammar)
        await composer.waitForAssist()
        guard case .failed = composer.assistState else {
            Issue.record("expected a failure, got \(composer.assistState)")
            return
        }
        #expect(composer.body == "Notes")
        composer.dismissAssistError()
        #expect(composer.assistState == .idle)

        let none = Self.inbox(engine: nil).composer
        none.open()
        none.body = "Notes"
        none.run(.fixGrammar)
        #expect(none.assistState == .failed("Quick is needed to write drafts. Install it from the model library."))
    }

    @Test @MainActor func aStoppedRewriteFinishingLateChangesNothing() async {
        let gate = ScanTestLatch()
        let engine = ScriptedComposeEngine(answers: ["Too late"], gate: gate)
        let composer = Self.inbox(engine: engine).composer
        composer.open()
        composer.body = "Original"
        composer.run(.rewrite(.shorter))
        composer.cancelAssist()
        #expect(composer.assistState == .idle)
        #expect(!composer.isLocked)
        composer.body = "Edited by hand"
        await gate.open()
        await composer.waitForAssist()
        #expect(composer.body == "Edited by hand")
        #expect(composer.assistState == .idle)
        #expect(!composer.canUndo)
    }

    @Test func sendErrorsTalkAboutTheMessage() {
        #expect(MailSendError.scriptFailed("x").userMessage == "Mail could not send the message: x")
    }
}

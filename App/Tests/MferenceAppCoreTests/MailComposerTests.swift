import Foundation
@testable import MferenceAppCore
import MferenceEngine
import Testing

/// Answers each generation with the next scripted answer (the last one
/// repeats), once `gate` opens, and records the person's prompt.
final class ScriptedComposeEngine: EngineServicing, @unchecked Sendable {
    let gate: ScanTestLatch?
    let finish: GenerationFinish
    private let lock = NSLock()
    private var answers: [String]
    private var recorded: [String] = []

    init(answers: [String], gate: ScanTestLatch? = nil, finish: GenerationFinish = .stop) {
        self.answers = answers
        self.gate = gate
        self.finish = finish
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
        let gate = gate, finish = finish
        return GenerationStream { continuation in
            Task {
                await gate?.wait()
                continuation.yield(.content(answer))
                continuation.yield(.completed(GenerationSummary(content: answer, reasoning: nil, toolCalls: [],
                                                                finish: finish, promptTokens: 1, completionTokens: 1)))
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

    /// Quick never sees a clipped copy that would then replace the whole
    /// body: past the limit, nothing is sent to it and nothing changes.
    @Test @MainActor func aMessageTooLongForQuickIsLeftAlone() async {
        let engine = ScriptedComposeEngine(answers: ["Short."])
        let composer = Self.inbox(engine: engine).composer
        composer.open()
        let long = String(repeating: "word ", count: MailComposeWriter.maximumInput / 5 + 1)
        composer.body = long
        for action in [ComposeAction.fixGrammar, .rewrite(.shorter)] {
            composer.run(action)
            await composer.waitForAssist()
            #expect(composer.assistState == .failed(MailComposeError.tooLong(limit: MailComposeWriter.maximumInput)
                .userMessage))
            #expect(composer.body == long)
        }
        composer.prompt = "turn this into an email"
        composer.run(.draft)
        await composer.waitForAssist()
        #expect(composer.body == long)
        #expect(engine.prompts.isEmpty)
    }

    /// An answer cut off at Quick's length limit never replaces the body.
    @Test @MainActor func anIncompleteAnswerIsNotUsed() async {
        let composer = Self.inbox(engine: ScriptedComposeEngine(answers: ["Hi Sam, I"], finish: .length)).composer
        composer.open()
        composer.body = "hi sam i cant make it friday, maybe next week"
        composer.run(.fixGrammar)
        await composer.waitForAssist()
        #expect(composer.assistState == .failed(MailComposeError.incomplete.userMessage))
        #expect(composer.body == "hi sam i cant make it friday, maybe next week")
        #expect(!composer.canUndo)
    }

    /// While the message sends, the draft cannot be closed, so the send's
    /// outcome can only ever close the draft it sent.
    @Test @MainActor func aDraftCannotBeDiscardedWhileItSends() async {
        let gate = ScanTestLatch()
        let sender = GatedSender(gate: gate)
        let composer = Self.inbox(engine: nil, sender: sender).composer
        composer.open()
        composer.to = "a@example.com"
        composer.body = "Hello"
        let sending = Task { await composer.send() }
        while composer.sendState != .sending { await Task.yield() }
        #expect(!composer.canDiscard && !composer.canSend)
        composer.discard()
        #expect(composer.isOpen && composer.body == "Hello")
        await gate.open()
        #expect(await sending.value == nil)
        #expect(await sender.sent == [MailDraft(to: "a@example.com", subject: "", body: "Hello")])
        #expect(!composer.isOpen && composer.body.isEmpty && composer.sendState == .idle)
    }

    /// Stop ends the engine's request too, not only the card's wait: the
    /// live `EngineService` cancels it and the engine is free again.
    @Test @MainActor func stopCancelsTheEngineRequest() async throws {
        let reached = FakeEngineLatch()
        let held = FakeEngineScript(events: [.content("Hi"), .content(" there")], holdAfterEvents: 1,
                                    holdReached: reached, holdGate: FakeEngineLatch())
        let journal = FakeEngineJournal()
        let llama = FakeEngineBackend(engine: .llamaCpp, journal: journal, scripts: [AppModel.quick.modelID: held])
        let catalog = EngineCatalog(models: [
            FakeEngineBackend.descriptor(id: AppModel.quick.modelID, engine: .llamaCpp,
                                         product: EngineProductSlot(modelID: AppModel.quick.modelID)),
        ])
        let engine = EngineService(discover: { catalog }) { catalog, observer in
            EngineHost(catalog: catalog, backends: [llama], idleUnloadAfter: nil, observer: observer)
        }
        let composer = Self.inbox(engine: engine).composer
        composer.open()
        composer.body = "hi sam"
        composer.run(.fixGrammar)
        try await reached.wait()
        composer.cancelAssist()
        await composer.waitForAssist()
        #expect(composer.body == "hi sam" && composer.assistState == .idle)
        let deadline = ContinuousClock.now + .seconds(10)
        while !journal.entries.contains(.generationCancelled(AppModel.quick.modelID)), ContinuousClock.now < deadline {
            try await Task.sleep(for: .milliseconds(5))
        }
        #expect(journal.entries.contains(.generationCancelled(AppModel.quick.modelID)))
    }

    @Test func sendErrorsTalkAboutTheMessage() {
        #expect(MailSendError.scriptFailed("x").userMessage == "Mail could not send the message: x")
    }
}

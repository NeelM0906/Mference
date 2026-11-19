import Foundation
@testable import MferenceAppCore
import MferenceEngine
import SwiftData
import Testing

/// A broker that records requests and never shows a system prompt.
final class RecordingPermissionBroker: PermissionBroker, @unchecked Sendable {
    private let lock = NSLock()
    private var states: [AppPermission: PermissionState]
    private var recorded: [AppPermission] = []
    private var opened: [AppPermission] = []

    init(_ states: [AppPermission: PermissionState] = [:]) {
        self.states = states
    }

    var requests: [AppPermission] { lock.withLock { recorded } }
    var openedSettings: [AppPermission] { lock.withLock { opened } }

    func state(of permission: AppPermission) async -> PermissionState {
        lock.withLock { states[permission] ?? .notDetermined }
    }

    func request(_ permission: AppPermission) async -> PermissionState {
        lock.withLock {
            recorded.append(permission)
            return states[permission] ?? .denied
        }
    }

    func openSystemSettings(for permission: AppPermission) async {
        lock.withLock { opened.append(permission) }
    }
}

/// Answers every generation with `answer` once `gate` opens.
final class GatedEngine: EngineServicing, @unchecked Sendable {
    let gate: ScanTestLatch
    let answer: String

    init(gate: ScanTestLatch, answer: String) {
        self.gate = gate
        self.answer = answer
    }

    func models() async -> [ModelOption] { [ModelOption(model: .quick, availability: .ready)] }
    func refreshCatalog() async {}
    var status: EngineServiceStatus { .empty }
    func statusUpdates() -> AsyncStream<EngineServiceStatus> { AsyncStream { $0.finish() } }
    func load(_ model: AppModel) async throws {}
    func unload() async throws {}
    func generate(model: AppModel, turns: [ChatTurn], options: MferenceAppCore.GenerationOptions) -> GenerationStream {
        let gate = gate, answer = answer
        return GenerationStream { continuation in
            Task {
                await gate.wait()
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

/// Sorts every message into "important" once `gate` opens.
struct GatedSorter: AppleModelServicing {
    var gate: ScanTestLatch

    func activeModel() async -> Result<AssistantModelKind, AssistantModelError> { .success(.appleOnDevice) }
    func title(for text: String) async throws -> AssistantResult<String> { .init(value: "", model: .appleOnDevice) }
    func summary(of text: String) async throws -> AssistantResult<String> { .init(value: "", model: .appleOnDevice) }
    func sortMail(_ messages: [MailSortItem], into buckets: [MailSortBucket], allowFallback: Bool) async throws
        -> AssistantResult<[String: String]> {
        await gate.wait()
        return AssistantResult(value: Dictionary(uniqueKeysWithValues: messages.map { ($0.messageID, "important") }),
                               model: .appleOnDevice)
    }
    func fillReceipt(from recognizedText: String) async throws -> AssistantResult<ReceiptDraft> {
        throw AssistantModelError.unreadableAnswer
    }
    func suggestEvent(subject: String, body: String, received: Date, timeZone: TimeZone, allowFallback: Bool)
        async throws -> AssistantResult<MailEventSuggestion?> { .init(value: nil, model: .appleOnDevice) }
}

@Suite struct AppleMailTests {
    @Test func readsTheSyntheticStoreNewestFirstWithoutDeletedMail() async throws {
        let fixture = try MailStoreFixture.make()
        defer { fixture.remove() }
        let source = AppleMailSource(mailRoot: fixture.root)
        #expect(await source.status() == .available(version: "V10"))
        let inbox = try await source.messages(in: [.inbox], limit: 10)
        #expect(inbox.map(\.messageID) == ["<review@example.com>", "<redlines@example.com>", "<dental@example.com>"])
        #expect(inbox[0].senderName == "Priya Raman")
        #expect(inbox[0].initials == "PR")
        #expect(inbox[0].mailbox.kind == .inbox)
        #expect(inbox[0].date == MailStoreFixture.received)
        #expect(!inbox[0].isRead && inbox[2].isRead)
        let sent = try await source.messages(in: [.sent], limit: 10)
        #expect(sent.map(\.subject) == ["Re: Lisbon flights"])
        #expect(try await source.messages(in: [.inbox], limit: 1).count == 1)
        #expect(try await source.accounts().map(\.id) == [MailStoreFixture.account])
    }

    @Test func aQuietMailboxIsNotCrowdedOutByNewerMail() async throws {
        let fixture = try MailStoreFixture.make(extraInbox: 300)
        defer { fixture.remove() }
        let source = AppleMailSource(mailRoot: fixture.root)
        #expect(try await source.messages(in: [.sent], limit: 50).map(\.subject) == ["Re: Lisbon flights"])
        #expect(try await source.messages(in: [.inbox], limit: 10).count == 10)
    }

    @Test @MainActor func aMoveDuringSortingSticks() async throws {
        let fixture = try MailStoreFixture.make()
        defer { fixture.remove() }
        let container = try AppModelContainer.inMemory()
        let gate = ScanTestLatch()
        let sorter = GatedSorter(gate: gate)
        let model = MailInboxModel(source: AppleMailSource(mailRoot: fixture.root), context: container.mainContext,
                                   assistant: sorter, drafter: nil, sender: RecordingMailSender(),
                                   broker: RecordingPermissionBroker())
        let loading = Task { await model.load() }
        let deadline = ContinuousClock.now + .seconds(10)
        while !model.isSorting, ContinuousClock.now < deadline { try await Task.sleep(for: .milliseconds(5)) }
        try #require(model.isSorting)
        let review = try #require(model.messages(in: .bucket("other")).first { $0.messageID == "<review@example.com>" })
        model.markDone(review)
        await gate.open()
        await loading.value
        #expect(model.count(in: .done) == 1)
        #expect(model.messages(in: .bucket("important")).allSatisfy { $0.messageID != "<review@example.com>" })
    }

    /// Mail's live index, write-ahead log and shared memory are only ever
    /// copied: their bytes and modification dates do not change, and mail
    /// still in the log is read.
    @Test func readsAPrivateCopyAndLeavesMailsFilesUntouched() async throws {
        let fixture = try MailStoreFixture.make()
        defer { fixture.remove() }
        let index = fixture.root.appending(path: "V10/MailData/Envelope Index")
        let writer = try MailStoreFixture.WALWriter(index: index)
        try writer.insertInboxMessage(rowID: 900, messageID: "<fresh@example.com>", subject: "Only in the WAL")
        let files = ["", "-wal", "-shm"].map { URL(filePath: index.path + $0) }
        #expect(files.allSatisfy { FileManager.default.fileExists(atPath: $0.path) })
        func fingerprint() throws -> [String] {
            try files.map { url in
                let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
                let date = (attributes[.modificationDate] as? Date)?.timeIntervalSince1970 ?? 0
                return "\(url.lastPathComponent) \(try Data(contentsOf: url).hashValue) \(date)"
            }
        }
        let before = try fingerprint()
        let source = AppleMailSource(mailRoot: fixture.root)
        let inbox = try await source.messages(in: [.inbox], limit: 10)
        #expect(inbox.first?.subject == "Only in the WAL")
        await source.reload()
        _ = try await source.messages(in: [.inbox], limit: 10)
        #expect(try fingerprint() == before)
        writer.close()
    }

    @Test func mailboxNamesCannotLeaveTheMailFolder() async throws {
        #expect(!MailboxRef(url: "imap://ABC/..%2F..%2FDocuments").isSafePath)
        #expect(!MailboxRef(url: "imap://ABC/../INBOX").isSafePath)
        #expect(!MailboxRef(url: "imap://..%2F/INBOX").isSafePath)
        #expect(MailboxRef(url: "imap://ABC/%5BGmail%5D/All%20Mail").isSafePath)
        let huge = Data("99999999999999999999999\nSubject: x\n\nbody\n".utf8)
        #expect(MailMessageParser.parseEMLX(huge)?.text == "body")
        let overstated = Data("9000000\nSubject: x\n\nbody\n".utf8)
        #expect(MailMessageParser.parseEMLX(overstated)?.text == "body")
    }

    @Test func hostileHTMLIsReadInLinearTime() {
        let unclosed = String(repeating: "<div ", count: 100_000) + "text"
        let comments = String(repeating: "<!-- ", count: 100_000)
        let scripts = String(repeating: "<script>", count: 50_000)
        let start = ContinuousClock.now
        _ = MailHTML.text(from: unclosed)
        _ = MailHTML.text(from: comments)
        _ = MailHTML.text(from: scripts)
        _ = MailHTML.remoteImages(in: String(repeating: "<img src=\"https://x.example/a.gif\">", count: 20_000))
        #expect(ContinuousClock.now - start < .seconds(5))
        let long = String(repeating: "word ", count: 50_000)
        let body = MailMessageParser.parseMessage(Data("Subject: x\n\n\(long)".utf8))
        #expect((body?.text.count ?? 0) <= MailMessageParser.maximumTextCharacters + 40)
        #expect(MailHTML.text(from: "<p title=\"a > b\">Hi</p><SCRIPT>bad()</script>there") == "Hi\nthere")
    }

    @Test @MainActor func repliesHonourReplyToAndShowWhoTheyGoTo() async throws {
        let message = MailMessageSummary(rowID: 1, messageID: "<q@example.com>", senderName: "Lists",
                                         senderAddress: "noreply@lists.example.com", subject: "Re: Plans",
                                         snippet: "", date: .now, isRead: false,
                                         mailbox: MailboxRef(url: "imap://work/INBOX"))
        let headers = ["reply-to": "Team <team@example.com>", "to": "Me <me@example.com>, other@example.com",
                       "cc": "boss@example.com"]
        let draft = MailDraft.reply(to: message, headers: headers, accountAddresses: ["me@example.com"], body: "Ok")
        #expect(draft.to == "team@example.com")
        #expect(draft.from == "me@example.com")
        #expect(draft.subject == "Re: Plans")
        #expect(draft.inReplyTo == "<q@example.com>")
        let noAccount = MailDraft.reply(to: message, headers: [:], accountAddresses: ["x@example.com"], body: "")
        #expect(noAccount.to == "noreply@lists.example.com" && noAccount.from == nil)
    }

    @Test @MainActor func sendIsOffWhileARewriteRunsAndForDraftsOrSent() async throws {
        let fixture = try MailStoreFixture.make()
        defer { fixture.remove() }
        let container = try AppModelContainer.inMemory()
        let gate = ScanTestLatch()
        let engine = GatedEngine(gate: gate, answer: "A shorter reply.")
        let sender = RecordingMailSender()
        let model = MailInboxModel(source: AppleMailSource(mailRoot: fixture.root), context: container.mainContext,
                                   assistant: AppleModelService(primary: ScriptedAssistantTextModel(kind: .appleOnDevice,
                                                                                                    answer: "{}"),
                                                                fallback: nil),
                                   drafter: MailReplyDrafter(engine: engine), sender: sender,
                                   broker: RecordingPermissionBroker())
        await model.load()
        model.draft = "Hi Priya, 11 works."
        #expect(model.canSend)
        let rewrite = Task { await model.writeDraft(tone: .shorter) }
        let deadline = ContinuousClock.now + .seconds(10)
        while model.draftState != .writing, ContinuousClock.now < deadline { try await Task.sleep(for: .milliseconds(5)) }
        #expect(!model.canSend)
        await model.send()
        #expect(await sender.sent.isEmpty)
        await gate.open()
        await rewrite.value
        #expect(model.draftState == .ready && model.draft == "A shorter reply.")
        await model.send()
        #expect(await sender.sent.count == 1)
        #expect(!model.canSend)

        await model.select(.sent)
        #expect(model.selected?.mailbox.kind == .sent)
        #expect(!model.canReply && model.replyDraft == nil)
    }

    @Test @MainActor func backgroundSortingUsesAppleOnlyAndOffersQuick() async throws {
        let fixture = try MailStoreFixture.make()
        defer { fixture.remove() }
        let container = try AppModelContainer.inMemory()
        let apple = ScriptedAssistantTextModel(kind: .appleOnDevice, availability: .unavailable(reason: "off"), answer: "")
        let quick = ScriptedAssistantTextModel(kind: .quick, answers: [.success(#"{"1": "important", "2": "reading", "3": "other"}"#)])
        let model = MailInboxModel(source: AppleMailSource(mailRoot: fixture.root), context: container.mainContext,
                                   assistant: AppleModelService(primary: apple, fallback: quick), drafter: nil,
                                   sender: RecordingMailSender(), broker: RecordingPermissionBroker())
        await model.load()
        #expect(model.needsManualSort)
        #expect(quick.requests.isEmpty)  // opening the first message asks nothing of Quick either
        await model.sortWithQuick()
        #expect(!model.needsManualSort)
        #expect(model.count(in: .bucket("important")) == 1)
    }

    @Test func bodiesComeFromEMLXAsTextWithTrackersCounted() async throws {
        let fixture = try MailStoreFixture.make()
        defer { fixture.remove() }
        let source = AppleMailSource(mailRoot: fixture.root)
        let inbox = try await source.messages(in: [.inbox], limit: 10)

        let review = try await source.body(of: inbox[0])
        #expect(review.text == "Hi! Could we move Wednesday’s review to 11:00? I’ll bring the new onboarding screens.\nThanks, Priya")
        #expect(review.trackersRemoved == 2)
        #expect(review.remoteImagesBlocked == 1)

        // Row 2345 lives in a sharded folder; base64 Latin-1 HTML.
        let redlines = try await source.body(of: inbox[1])
        #expect(redlines.text == "Two changes left in section 4. Can you look before Friday?\nCafé at 3?")
        #expect(redlines.headers["from"].map(MailMessageParser.decodeHeader) == "Tom Okafor <tom@example.com>")
        #expect(MailMessageParser.decodeHeader(redlines.headers["subject"] ?? "") == "Contract redlines, round 2")

        let plain = try await source.body(of: inbox[2])
        #expect(plain.text == "Reply with a few times that work and we'll confirm.")
    }

    @Test func reportsMissingStoreAccessAndUnknownSchemas() async throws {
        let empty = URL.temporaryDirectory.appending(path: "no-mail-\(UUID().uuidString)")
        #expect(await AppleMailSource(mailRoot: empty).status() == .notFound)

        let old = try MailStoreFixture.make(schema: .missingDeleted)
        defer { old.remove() }
        guard case .unsupported(let detail) = await AppleMailSource(mailRoot: old.root).status() else {
            Issue.record("expected unsupported"); return
        }
        #expect(detail.contains("messages lacks deleted"))
        #expect(MailStoreStatus.unsupported(detail: detail).userMessage?.contains("cannot read") == true)

        let unreadable = try MailStoreFixture.make()
        defer { unreadable.remove() }
        let index = unreadable.root.appending(path: "V10/MailData/Envelope Index")
        try FileManager.default.setAttributes([.posixPermissions: 0o000], ofItemAtPath: index.path)
        defer { try? FileManager.default.setAttributes([.posixPermissions: 0o644], ofItemAtPath: index.path) }
        #expect(await AppleMailSource(mailRoot: unreadable.root).status() == .needsFullDiskAccess)
    }

    @Test func picksTheNewestStoreVersion() async throws {
        let fixture = try MailStoreFixture.make(version: "V11")
        defer { fixture.remove() }
        try FileManager.default.createDirectory(at: fixture.root.appending(path: "V9/MailData"),
                                                withIntermediateDirectories: true)
        #expect(await AppleMailSource(mailRoot: fixture.root).status() == .available(version: "V11"))
    }

    @Test func shardsAndMailboxURLs() {
        #expect(AppleMailSource.shardPath(for: 999) == "")
        #expect(AppleMailSource.shardPath(for: 2_345) == "/2")
        #expect(AppleMailSource.shardPath(for: 123_456) == "/3/2/1")
        let gmail = MailboxRef(url: "imap://ABC/%5BGmail%5D/All%20Mail")
        #expect(gmail.accountID == "ABC")
        #expect(gmail.path == ["[Gmail]", "All Mail"])
        #expect(gmail.kind == .archive)
        #expect(MailboxRef(url: "ews://X/Sent%20Items").kind == .sent)
    }

    @Test func htmlToTextAndTrackerRules() {
        let html = """
            <html><head><style>p{color:red}</style><title>t</title></head><body>
            <script>alert(1)</script><!-- note --><p>One &amp; two</p><ul><li>a</li><li>b</li></ul>
            <img src="https://x.example/pixel.gif"><img src="cid:logo"><img src='https://x.example/a.png' width="320">
            &#8364;5 &#x2014; ok</body></html>
            """
        let text = MailMessageParser.normalize(MailHTML.text(from: html))
        #expect(text == "One & two\n• a\n• b\n\n€5 — ok")
        let images = MailHTML.remoteImages(in: html)
        #expect(images.map(\.isTracker) == [true, false])
        #expect(MailMessageParser.decodeHeader("=?UTF-8?Q?Caf=C3=A9?= =?UTF-8?Q?_ouvert?=") == "Café ouvert")
    }

    @Test @MainActor func bucketsAreCreatedRenamedAndAssignmentsStick() throws {
        let container = try AppModelContainer.inMemory()
        let store = MailBucketStore(context: container.mainContext)
        #expect(try store.buckets().map(\.name) == ["Important", "Calendar", "Reading", "Other"])
        try store.rename(key: "reading", to: "  Later  ")
        try store.rename(key: "other", to: "   ")
        #expect(try store.buckets().map(\.name) == ["Important", "Calendar", "Later", "Other"])

        try store.assign("<a@example.com>", to: "other", manual: false)
        try store.assign("<a@example.com>", to: "important", manual: true)
        try store.assign("<a@example.com>", to: "reading", manual: false)  // sorter cannot undo a person's move
        #expect(try store.assignments(for: ["<a@example.com>"])["<a@example.com>"]?.bucketKey == "important")
        try store.removeAssignment("<a@example.com>")
        #expect(try store.assignments(for: ["<a@example.com>"]).isEmpty)
    }

    @Test @MainActor func inboxSortsReadsProposesDraftsAndSendsOnlyOnClick() async throws {
        let fixture = try MailStoreFixture.make()
        defer { fixture.remove() }
        let container = try AppModelContainer.inMemory()
        // The newest message opens (event card) before sorting runs.
        let sorter = ScriptedAssistantTextModel(kind: .appleOnDevice, answers: [
            .success(#"{"title": "Design review", "start": "2026-09-16T11:00", "end": "2026-09-16T11:45"}"#),
            .success(#"{"1": "important", "2": "important", "3": "calendar"}"#),
        ])
        let engine = AppleModelServiceTests.quickEngine(answer: "Hi Priya, 11 works. See you Wednesday!")
        let sender = RecordingMailSender()
        var proposed: [ProposedEvent] = []
        let model = MailInboxModel(source: AppleMailSource(mailRoot: fixture.root), context: container.mainContext,
                                   assistant: AppleModelService(primary: sorter, fallback: nil),
                                   drafter: MailReplyDrafter(engine: engine), sender: sender,
                                   broker: RecordingPermissionBroker(),
                                   timeZone: TimeZone(identifier: "UTC")!,
                                   onEventSuggested: { proposed.append($0) })
        await model.load()
        #expect(model.status == .available(version: "V10"))
        #expect(model.count(in: .bucket("important")) == 2)
        #expect(model.count(in: .bucket("calendar")) == 1)
        #expect(model.folderTitle == "Important")
        #expect(model.selected?.messageID == "<review@example.com>")
        #expect(model.body?.trackersRemoved == 2)
        let card = try #require(model.eventCard)
        #expect(card.title == "Design review")
        #expect(card.source.reference == "<review@example.com>")
        #expect(await sender.sent.isEmpty)

        await model.writeDraft()
        #expect(model.draftState == .ready)
        #expect(model.draft == "Hi Priya, 11 works. See you Wednesday!")
        #expect(model.draftModelLabel == "Quick 4B")
        #expect(await sender.sent.isEmpty)

        model.draft = "Hi Priya, 11 works."
        await model.send()
        #expect(model.draftState == .sent)
        #expect(await sender.sent == [MailDraft(to: "priya@example.com", subject: "Re: Wednesday design review",
                                                body: "Hi Priya, 11 works.", inReplyTo: "<review@example.com>")])
        #expect(!model.canSend)  // sent once; the button stays off

        model.addEventCardToCalendar()
        #expect(proposed.map(\.title) == ["Design review"])
        #expect(model.eventCard == nil)

        let review = try #require(model.messages.first)
        model.markDone(review)
        #expect(model.count(in: .done) == 1)
        #expect(model.count(in: .bucket("important")) == 1)
        model.undo()
        #expect(model.count(in: .done) == 0)
        #expect(model.count(in: .bucket("important")) == 2)

        // Assignments persist: a second load does not ask the sorter again.
        let before = sorter.requests.count
        await model.load()
        #expect(sorter.requests.count == before + 1)  // only the event card of the reopened message
        await engine.shutdown()
    }

    @Test @MainActor func missingAccessAsksThroughTheBroker() async throws {
        let container = try AppModelContainer.inMemory()
        let broker = RecordingPermissionBroker([.fullDiskAccess: .needsSystemSettings])
        let unreadable = try MailStoreFixture.make()
        defer { unreadable.remove() }
        let index = unreadable.root.appending(path: "V10/MailData/Envelope Index")
        try FileManager.default.setAttributes([.posixPermissions: 0o000], ofItemAtPath: index.path)
        defer { try? FileManager.default.setAttributes([.posixPermissions: 0o644], ofItemAtPath: index.path) }
        let model = MailInboxModel(source: AppleMailSource(mailRoot: unreadable.root), context: container.mainContext,
                                   assistant: AppleModelService(primary: ScriptedAssistantTextModel(kind: .quick, answer: "{}"),
                                                                fallback: nil),
                                   drafter: nil, sender: RecordingMailSender(), broker: broker)
        await model.load()
        #expect(model.status == .needsFullDiskAccess)
        #expect(broker.requests.isEmpty)
        await model.requestAccess()
        #expect(broker.requests == [.fullDiskAccess])
        #expect(broker.openedSettings.isEmpty)  // the broker's request opens the pane itself
    }

    @Test @MainActor func composeSendsOnlyAValidMessage() async throws {
        let container = try AppModelContainer.inMemory()
        let sender = RecordingMailSender()
        let empty = URL.temporaryDirectory.appending(path: "no-mail-\(UUID().uuidString)")
        let model = MailInboxModel(source: AppleMailSource(mailRoot: empty), context: container.mainContext,
                                   assistant: AppleModelService(primary: ScriptedAssistantTextModel(kind: .appleOnDevice,
                                                                                                    answer: "{}"),
                                                                fallback: nil),
                                   drafter: nil, sender: sender, broker: RecordingPermissionBroker())
        #expect(await model.sendNewMessage(to: "nobody", subject: "Hi", body: "Hello") != nil)
        #expect(await model.sendNewMessage(to: "a@example.com", subject: "Hi", body: "  ") != nil)
        #expect(await sender.sent.isEmpty)
        #expect(await model.sendNewMessage(to: " a@example.com ", subject: "Hi", body: "Hello") == nil)
        #expect(await sender.sent == [MailDraft(to: "a@example.com", subject: "Hi", body: "Hello")])
        model.isComposing = true
        #expect(model.isEditingText)
    }

    @Test func appleScriptEscapesAndRefusesOddAddresses() {
        let draft = MailDraft(to: "priya@example.com", from: "me@example.com", subject: "Re: \"Q3\" plan",
                              body: "Line \\ one\nLine two\u{0}\" & do shell script \"x", inReplyTo: "<q\"@example.com>")
        let script = AppleMailSender.script(for: draft)
        #expect(script.contains(#"subject:"Re: \"Q3\" plan""#))
        #expect(script.contains(#"content:"Line \\ one"#))
        #expect(!script.contains("\u{0}"))
        // The payload stays inside one string literal: every quote it holds is escaped.
        #expect(script.contains(#"Line two\" & do shell script \"x""#))
        #expect(!script.contains(#"two" & do shell"#))
        #expect(script.contains(#"whose message id is "q\"@example.com""#))
        #expect(script.contains(#"set sender of newMessage to "me@example.com""#))
        #expect(script.contains("reply (item 1 of originals) without opening window"))
        #expect(AppleMailSender.isPlausibleAddress("a@b.co"))
        #expect(!AppleMailSender.isPlausibleAddress("a\"@b.co"))
        #expect(!AppleMailSender.isPlausibleAddress("nobody"))
        #expect(MailDraft.replySubject("RE: x") == "RE: x")
        #expect(MailDraft.replySubject("x") == "Re: x")
    }
}

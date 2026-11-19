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

/// Sends once `gate` opens.
actor GatedSender: MailSending {
    let gate: ScanTestLatch
    private(set) var sent: [MailDraft] = []

    init(gate: ScanTestLatch) { self.gate = gate }

    func send(_ draft: MailDraft) async throws {
        await gate.wait()
        sent.append(draft)
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
        #expect(try await source.accounts().map(\.id) == [MailStoreFixture.account, MailStoreFixture.secondAccount])
    }

    /// The sidebar is Mail's own: Favorites across accounts, then each
    /// account's mailboxes in Mail's order, with Mail's counts. Inbox opens
    /// first; nothing is sorted or hidden.
    @Test @MainActor func sidebarListsMailsOwnMailboxes() async throws {
        let fixture = try MailStoreFixture.make()
        defer { fixture.remove() }
        let model = MailInboxModel(source: AppleMailSource(mailRoot: fixture.root),
                                   assistant: AppleModelService(primary: ScriptedAssistantTextModel(kind: .appleOnDevice,
                                                                                                    answer: "{}"),
                                                                fallback: nil),
                                   drafter: nil, sender: RecordingMailSender(), broker: RecordingPermissionBroker(),
                                   mailApp: FixedMailApp())
        await model.load()
        #expect(model.folder == .favorite(.inbox) && model.folderTitle == "Inbox" && model.folderScope == "All accounts")
        #expect(model.messages.map(\.messageID) == ["<review@example.com>", "<redlines@example.com>",
                                                    "<dental@example.com>"])
        #expect(model.selected?.messageID == "<review@example.com>")
        #expect(model.count(in: .favorite(.inbox)) == 2 && model.unreadInFolder == 2)
        #expect(model.offersFlagged && model.count(in: .flagged) == 1)
        #expect(model.localMailboxes.isEmpty)
        #expect(model.accountSections.map { $0.mailboxes.map(\.name) } == [["Inbox", "Sent", "Trash"], ["Junk"]])
        #expect(model.accountSections.map(\.account.title) == ["me@example.com", MailAccount.unnamed])

        await model.select(.favorite(.sent))
        #expect(model.messages.map(\.subject) == ["Re: Lisbon flights"])
        #expect(model.messages.first?.counterpart == "Lena Fischer" && !model.canReply)
        await model.select(.flagged)
        #expect(model.folderTitle == "Flagged" && model.messages.map(\.messageID) == ["<redlines@example.com>"])
        #expect(model.messages.first?.isFlagged == true && model.canReply)
        let junk = try #require(model.accountSections.last?.mailboxes.first)
        await model.select(.mailbox(junk.id))
        #expect(model.folderTitle == "Junk" && model.folderScope == MailAccount.unnamed)
        #expect(model.messages.map(\.subject) == ["You won!"] && !model.canReply)
        await model.select(.favorite(.drafts))
        #expect(model.messages.isEmpty && model.selected == nil)
    }

    /// Gmail stores each message once, in All Mail, and labels it into
    /// Inbox, Sent and Important: those lists come from the labels. Its
    /// standard mailboxes are the "[Gmail]" ones; a folder of the person's
    /// own called "Drafts" is not Drafts.
    @Test func gmailLabelsFillInboxSentAndImportant() async throws {
        let fixture = try MailStoreFixture.make(gmail: true)
        defer { fixture.remove() }
        let source = AppleMailSource(mailRoot: fixture.root)
        let inbox = try await source.messages(in: [.inbox], limit: 10)
        #expect(inbox.map(\.messageID) == ["<g-inbox@example.com>", "<review@example.com>", "<redlines@example.com>",
                                           "<dental@example.com>"])
        #expect(inbox[0].mailbox.url == MailStoreFixture.allMail && inbox[0].listedKind == .inbox)
        #expect(inbox[0].isFlagged && !inbox[0].isOutgoing)
        #expect(try await source.body(of: inbox[0]).text == "Noon at the usual place?")

        let sent = try await source.messages(in: [.sent], limit: 10)
        #expect(sent.map(\.messageID) == ["<g-sent@example.com>", "<sent@example.com>"])
        #expect(sent[0].isOutgoing && sent[0].recipients.map(\.display) == ["Lena Fischer", "tom@example.com"])
        #expect(sent[0].counterpart == "Lena Fischer, +1")
        #expect(try await source.messages(in: [.important], limit: 10).map(\.messageID) == ["<g-inbox@example.com>"])
        #expect(try await source.messages(in: [.drafts], limit: 10).isEmpty)
        #expect(try await source.messages(in: [.archive], limit: 10).count == 4)

        let listing = try await source.mailboxes()
        let gmail = listing.mailboxes.filter { $0.ref.accountID == MailStoreFixture.gmailAccount }
        #expect(gmail.map(\.name) == ["Important", "Inbox", "Sent", "Archive", "archinve", "Drafts"])
        #expect(gmail.map(\.kind) == [.important, .inbox, .sent, .archive, .other, .other])
        let counts = Dictionary(uniqueKeysWithValues: gmail.map { ($0.name, ($0.unreadCount, $0.totalCount)) })
        #expect(counts["Inbox"]! == (1, 1) && counts["Sent"]! == (0, 1) && counts["Archive"]! == (2, 4))
        #expect(counts["Drafts"]! == (1, 1) && counts["archinve"]! == (0, 0))
        #expect(listing.flaggedCount == 2)
        let folder = try #require(gmail.first { $0.name == "Drafts" })
        #expect(try await source.messages(in: .mailbox(folder.id), limit: 10).map(\.messageID)
                == ["<g-label@example.com>"])
    }

    @Test func mailboxClassifierFollowsMailsSidebar() {
        let rows: [(id: Int64, url: String)] = [
            (1, "imap://A/Sent%20Messages"), (2, "imap://A/INBOX"), (3, "imap://A/Sent"), (4, "imap://A/Zeta"),
            (5, "imap://A/alpha"), (6, "local:///Send%20Later"), (7, "imap://B/%5BGoogle%20Mail%5D/Bin"),
            (8, "imap://B/%5BGoogle%20Mail%5D"), (9, "imap://B/Work/Clients"), (10, "imap://A"),
        ]
        let classified = MailboxClassifier.classify(rows).map(\.mailbox)
        #expect(classified.map(\.name) == ["Inbox", "Sent", "alpha", "Sent", "Zeta", "Send Later", "Trash",
                                           "Work / Clients"])
        #expect(classified.map(\.kind) == [.inbox, .sent, .other, .other, .other, .other, .trash, .other])
        #expect(classified[5].isLocal && !classified[0].isLocal)
        #expect(Mailbox(ref: MailboxRef(url: "imap://A/Drafts"), kind: .drafts, name: "Drafts", unreadCount: 0,
                        totalCount: 6).badgeCount == 6)
    }

    /// While the screen is shown it follows Mail: a change Mail saves is
    /// read within one poll, the folder and open message stay, and the
    /// sidebar says when Mail is closed.
    @Test @MainActor func followsMailAsItSavesChanges() async throws {
        let inbox = MailboxRef(url: "imap://A/INBOX")
        func message(_ id: String, minutes: Double) -> MailMessageSummary {
            MailMessageSummary(rowID: 1, messageID: id, senderName: "Ana", senderAddress: "ana@example.com",
                               subject: id, snippet: "", date: MailStoreFixture.received.addingTimeInterval(minutes * 60),
                               isRead: false, mailbox: inbox)
        }
        let source = InMemoryMailSource(messages: [message("<a>", minutes: 0)])
        let refreshed = MailStoreFixture.received
        let model = MailInboxModel(source: source,
                                   assistant: AppleModelService(primary: ScriptedAssistantTextModel(kind: .appleOnDevice,
                                                                                                    answer: "{}"),
                                                                fallback: nil),
                                   drafter: nil, sender: RecordingMailSender(), broker: RecordingPermissionBroker(),
                                   mailApp: FixedMailApp(running: false), pollInterval: .milliseconds(10),
                                   now: { refreshed })
        await model.load()
        #expect(model.messages.map(\.messageID) == ["<a>"] && model.selected?.messageID == "<a>")
        #expect(model.syncState == .mailClosed && model.lastRefreshed == refreshed)

        await source.setMessages([message("<a>", minutes: 0), message("<b>", minutes: 5)])
        let watching = Task { await model.watch() }
        let deadline = ContinuousClock.now + .seconds(10)
        while model.messages.count < 2, ContinuousClock.now < deadline { try await Task.sleep(for: .milliseconds(5)) }
        watching.cancel()
        await watching.value
        #expect(model.messages.map(\.messageID) == ["<b>", "<a>"])
        #expect(model.selected?.messageID == "<a>")
        #expect(model.count(in: .favorite(.inbox)) == 2)
    }

    /// Mail's own writes change the token the screen polls.
    @Test func changeTokenMovesWhenMailSaves() async throws {
        let fixture = try MailStoreFixture.make()
        defer { fixture.remove() }
        let source = AppleMailSource(mailRoot: fixture.root)
        let before = try #require(await source.changeToken())
        let writer = try MailStoreFixture.WALWriter(index: fixture.root.appending(path: "V10/MailData/Envelope Index"))
        try writer.insertInboxMessage(rowID: 900, messageID: "<fresh@example.com>", subject: "New")
        let after = try #require(await source.changeToken())
        #expect(before != after)
        await source.reload()
        #expect(try await source.messages(in: [.inbox], limit: 1).first?.subject == "New")
        writer.close()
        #expect(await AppleMailSource(mailRoot: URL.temporaryDirectory.appending(path: "none-\(UUID())")).changeToken()
                == nil)
    }

    @Test func mailboxKindsCoverCommonProviderNames() {
        let kind = { (path: String) in MailboxRef(url: "imap://A/\(path)").kind }
        #expect(kind("%5BGmail%5D/Spam") == .junk)
        #expect(kind("Junk%20E-mail") == .junk)
        #expect(kind("Bulk%20Mail") == .junk)
        #expect(kind("%5BGmail%5D/Trash") == .trash)
        #expect(kind("Deleted%20Items") == .trash)
        #expect(kind("Sent%20Items") == .sent)
    }

    /// Accounts show an address or a name, never Mail's folder UUID: the
    /// system's account list first, then the account's Sent address, then
    /// "Mail account".
    @Test func accountsShowAnAddressOrANameNeverTheFolderID() async throws {
        let fixture = try MailStoreFixture.make()
        defer { fixture.remove() }
        let plain = try await AppleMailSource(mailRoot: fixture.root).accounts()
        #expect(plain.map(\.name) == ["me@example.com", MailAccount.unnamed])
        #expect(plain.map(\.address) == ["me@example.com", nil])

        let accounts = fixture.root.appending(path: "Accounts4.sqlite")
        try MailStoreFixture.writeAccounts(at: accounts, rows: [
            (pk: 1, id: "PARENT-1", user: "neel@work.example", description: "Work", parent: nil),
            (pk: 2, id: MailStoreFixture.account, user: nil, description: nil, parent: 1),
            (pk: 3, id: MailStoreFixture.secondAccount, user: nil, description: "iCloud", parent: nil),
        ])
        let named = try await AppleMailSource(mailRoot: fixture.root, accountsDatabase: accounts).accounts()
        #expect(named.map(\.name) == ["neel@work.example", "iCloud"])
        #expect(!named.contains { $0.name.contains(MailStoreFixture.account) })
    }

    @Test func aQuietMailboxIsNotCrowdedOutByNewerMail() async throws {
        let fixture = try MailStoreFixture.make(extraInbox: 300)
        defer { fixture.remove() }
        let source = AppleMailSource(mailRoot: fixture.root)
        #expect(try await source.messages(in: [.sent], limit: 50).map(\.subject) == ["Re: Lisbon flights"])
        #expect(try await source.messages(in: [.inbox], limit: 10).count == 10)
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
        let gate = ScanTestLatch()
        let engine = GatedEngine(gate: gate, answer: "A shorter reply.")
        let sender = RecordingMailSender()
        let model = MailInboxModel(source: AppleMailSource(mailRoot: fixture.root), assistant: AppleModelService(primary: ScriptedAssistantTextModel(kind: .appleOnDevice,
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

        await model.select(.favorite(.sent))
        #expect(model.selected?.mailbox.kind == .sent)
        #expect(!model.canReply && model.replyDraft == nil)
    }

    @Test @MainActor func aSendFinishingAfterAnotherMessageOpensLeavesThatOneAlone() async throws {
        let fixture = try MailStoreFixture.make()
        defer { fixture.remove() }
        let gate = ScanTestLatch()
        let sender = GatedSender(gate: gate)
        let model = MailInboxModel(source: AppleMailSource(mailRoot: fixture.root), assistant: AppleModelService(primary: ScriptedAssistantTextModel(kind: .appleOnDevice,
                                                                                                    answer: "{}"),
                                                                fallback: nil),
                                   drafter: nil, sender: sender, broker: RecordingPermissionBroker())
        await model.load()
        let first = try #require(model.selected)
        model.draft = "Sounds good."
        let sending = Task { await model.send() }
        let deadline = ContinuousClock.now + .seconds(10)
        while model.draftState != .sending, ContinuousClock.now < deadline { try await Task.sleep(for: .milliseconds(5)) }
        let other = try #require(model.messages.first { $0.messageID != first.messageID })
        await model.open(other)
        #expect(model.draftState == .idle)
        // Back to the first while its reply is still sending: it says so and
        // cannot be sent again.
        await model.open(first)
        #expect(model.draftState == .sending)
        model.draft = "A second reply."
        #expect(!model.canSend)
        await gate.open()
        await sending.value
        #expect(model.draftState == .sent)
        #expect(await sender.sent.count == 1)
        await model.open(other)
        #expect(model.draftState == .idle)
        await model.open(first)
        #expect(model.draftState == .sent)
    }

    /// Loading Mail asks no model to sort it: Quick is never loaded over
    /// the chat model just by opening the screen.
    @Test @MainActor func loadingAsksNoModelToSortMail() async throws {
        let fixture = try MailStoreFixture.make()
        defer { fixture.remove() }
        let apple = ScriptedAssistantTextModel(kind: .appleOnDevice, availability: .unavailable(reason: "off"), answer: "")
        let quick = ScriptedAssistantTextModel(kind: .quick, answer: "{}")
        let model = MailInboxModel(source: AppleMailSource(mailRoot: fixture.root),
                                   assistant: AppleModelService(primary: apple, fallback: quick), drafter: nil,
                                   sender: RecordingMailSender(), broker: RecordingPermissionBroker())
        await model.load()
        #expect(model.messages.count == 3)
        #expect(quick.requests.isEmpty)
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

    @Test @MainActor func inboxReadsProposesDraftsAndSendsOnlyOnClick() async throws {
        let fixture = try MailStoreFixture.make()
        defer { fixture.remove() }
        // The newest message opens with its event card.
        let sorter = ScriptedAssistantTextModel(kind: .appleOnDevice, answers: [
            .success(#"{"title": "Design review", "start": "2026-09-16T11:00", "end": "2026-09-16T11:45"}"#),
        ])
        let engine = AppleModelServiceTests.quickEngine(answer: "Hi Priya, 11 works. See you Wednesday!")
        let sender = RecordingMailSender()
        var proposed: [ProposedEvent] = []
        let model = MailInboxModel(source: AppleMailSource(mailRoot: fixture.root), assistant: AppleModelService(primary: sorter, fallback: nil),
                                   drafter: MailReplyDrafter(engine: engine), sender: sender,
                                   broker: RecordingPermissionBroker(),
                                   timeZone: TimeZone(identifier: "UTC")!,
                                   onEventSuggested: { proposed.append($0) })
        await model.load()
        #expect(model.status == .available(version: "V10"))
        #expect(model.folder == .favorite(.inbox) && model.folderTitle == "Inbox")
        #expect(model.messages.count == 3)
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
        // Sent from the account's own address (read from its Sent mailbox).
        #expect(await sender.sent == [MailDraft(to: "priya@example.com", from: "me@example.com",
                                                subject: "Re: Wednesday design review",
                                                body: "Hi Priya, 11 works.", inReplyTo: "<review@example.com>")])
        #expect(!model.canSend)  // sent once; the button stays off

        model.addEventCardToCalendar()
        #expect(proposed.map(\.title) == ["Design review"])
        #expect(model.eventCard == nil)

        // Loading again keeps the open message and asks the model nothing.
        let before = sorter.requests.count
        await model.load()
        #expect(model.selected?.messageID == "<review@example.com>")
        #expect(sorter.requests.count == before)
        await engine.shutdown()
    }

    @Test @MainActor func missingAccessAsksThroughTheBroker() async throws {
        let broker = RecordingPermissionBroker([.fullDiskAccess: .needsSystemSettings])
        let unreadable = try MailStoreFixture.make()
        defer { unreadable.remove() }
        let index = unreadable.root.appending(path: "V10/MailData/Envelope Index")
        try FileManager.default.setAttributes([.posixPermissions: 0o000], ofItemAtPath: index.path)
        defer { try? FileManager.default.setAttributes([.posixPermissions: 0o644], ofItemAtPath: index.path) }
        let model = MailInboxModel(source: AppleMailSource(mailRoot: unreadable.root), assistant: AppleModelService(primary: ScriptedAssistantTextModel(kind: .quick, answer: "{}"),
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
        let sender = RecordingMailSender()
        let empty = URL.temporaryDirectory.appending(path: "no-mail-\(UUID().uuidString)")
        let model = MailInboxModel(source: AppleMailSource(mailRoot: empty), assistant: AppleModelService(primary: ScriptedAssistantTextModel(kind: .appleOnDevice,
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

    @Test func sendingNeedsAGrantedAutomationPermission() async {
        let draft = MailDraft(to: "a@example.com", subject: "Hi", body: "Hello")
        for state in [PermissionState.notDetermined, .denied, .needsSystemSettings] {
            let broker = RecordingPermissionBroker([.mailAutomation: state])
            await #expect(throws: MailSendError.notAllowed) { try await AppleMailSender(broker: broker).send(draft) }
            #expect(broker.requests == [.mailAutomation])
        }
    }

    /// macOS applies Full Disk Access only from the next launch: the screen
    /// re-checks when the app becomes active, and after System Settings was
    /// opened and access still fails, it offers Quit and Reopen.
    @Test @MainActor func fullDiskAccessIsRecheckedAndOffersARelaunch() async throws {
        let source = InMemoryMailSource(status: .needsFullDiskAccess)
        let broker = RecordingPermissionBroker([.fullDiskAccess: .needsSystemSettings])
        let model = MailInboxModel(source: source, assistant: AppleModelService(primary: ScriptedAssistantTextModel(kind: .appleOnDevice,
                                                                                                    answer: "{}"),
                                                                fallback: nil),
                                   drafter: nil, sender: RecordingMailSender(), broker: broker)
        await model.load()
        #expect(model.status == .needsFullDiskAccess)
        #expect(!model.needsRelaunchForAccess)

        // "Not now" under the reason: Mference never left, so no relaunch.
        await model.requestAccess()
        #expect(!model.needsRelaunchForAccess)

        // Opened System Settings (Mference left the foreground right after
        // the request), but the grant is not applied yet.
        await model.requestAccess()
        model.appResignedActive()
        #expect(model.status == .needsFullDiskAccess)
        #expect(model.needsRelaunchForAccess)

        // Granted and applied: the next activation (or Check again) loads Mail.
        await source.setStatus(.available(version: "V10"))
        await model.recheckAccess()
        #expect(model.status == .available(version: "V10"))
        #expect(!model.needsRelaunchForAccess)
    }

    @Test @MainActor func connectionsOfferARelaunchWhenAccessIsStillMissing() async {
        let source = InMemoryMailSource(status: .needsFullDiskAccess)
        let connections = AppleConnections(calendar: InMemoryCalendarService(), mail: source,
                                           broker: RecordingPermissionBroker([.fullDiskAccess: .needsSystemSettings]))
        await connections.refresh()
        #expect(!connections.mailNeedsRelaunch)
        await connections.connectMail()
        #expect(!connections.mailNeedsRelaunch, "Not now: Settings never opened")
        await connections.connectMail()
        connections.appResignedActive()
        #expect(connections.mailNeedsRelaunch)
        await source.setStatus(.available(version: "V10"))
        await connections.refresh()
        #expect(!connections.mailNeedsRelaunch && connections.status.mail == .connected)
    }

    /// A switch away long after the request is not the Settings visit.
    @Test func aLateSwitchAwayIsNotASettingsVisit() {
        let start = ContinuousClock.now
        var late = SettingsVisit()
        late.begin()
        late.end(at: start)
        late.appLeft(at: start + .seconds(30))
        #expect(!late.happened)
        var during = SettingsVisit()
        during.begin()
        during.appLeft(at: start)
        #expect(during.happened)
        var unasked = SettingsVisit()
        unasked.appLeft()
        #expect(!unasked.happened)
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

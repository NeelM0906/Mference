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
        let sorter = ScriptedAssistantTextModel(kind: .appleOnDevice, answers: [
            .success(#"{"1": "important", "2": "important", "3": "calendar"}"#),
            .success(#"{"title": "Design review", "start": "2026-09-16T11:00", "end": "2026-09-16T11:45"}"#),
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
        #expect(broker.openedSettings == [.fullDiskAccess])
    }

    @Test func appleScriptEscapesAndRefusesOddAddresses() {
        let draft = MailDraft(to: "priya@example.com", subject: "Re: \"Q3\" plan", body: "Line \\ one\nLine two")
        let script = AppleMailSender.script(for: draft)
        #expect(script.contains(#"subject:"Re: \"Q3\" plan""#))
        #expect(script.contains(#"content:"Line \\ one"#))
        #expect(AppleMailSender.isPlausibleAddress("a@b.co"))
        #expect(!AppleMailSender.isPlausibleAddress("a\"@b.co"))
        #expect(!AppleMailSender.isPlausibleAddress("nobody"))
        #expect(MailDraft.replySubject("RE: x") == "RE: x")
        #expect(MailDraft.replySubject("x") == "Re: x")
    }
}

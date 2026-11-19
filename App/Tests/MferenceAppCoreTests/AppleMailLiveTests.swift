import Foundation
@testable import MferenceAppCore
import Testing

/// Checkpoint C-B3's live read of this Mac's Apple Mail, read-only. Gated:
/// it needs Full Disk Access for the process running the tests, which the
/// maintainer grants. Run only with `MFERENCE_MAIL_LIVE=1`. It records counts
/// and parse outcomes only, never subjects, senders or bodies, to
/// `MFERENCE_MAIL_LIVE_REPORT` when set.
@Suite struct AppleMailLiveTests {
    static let enabled = ProcessInfo.processInfo.environment["MFERENCE_MAIL_LIVE"] == "1"

    @Test(.enabled(if: enabled))
    func readsThisMacsMailWithoutContent() async throws {
        let source = AppleMailSource()
        let status = await source.status()
        var lines = ["status: \(Self.describe(status))"]
        defer {
            if let path = ProcessInfo.processInfo.environment["MFERENCE_MAIL_LIVE_REPORT"] {
                try? (lines.joined(separator: "\n") + "\n").write(toFile: path, atomically: true, encoding: .utf8)
            }
        }
        guard case .available = status else {
            Issue.record("Mail store not readable: \(Self.describe(status))")
            return
        }
        let start = ContinuousClock.now
        let inbox = try await source.messages(in: [.inbox], limit: 200)
        lines.append("inbox messages listed: \(inbox.count) in \(ContinuousClock.now - start)")
        lines.append("with Message-ID header: \(inbox.filter { !$0.messageID.hasPrefix("mail-row:") }.count)")
        lines.append("accounts: \(try await source.accounts().count)")
        var parsed = 0, empty = 0, missing = 0, trackers = 0
        for message in inbox.prefix(25) {
            do {
                let body = try await source.body(of: message)
                if body.text.isEmpty { empty += 1 } else { parsed += 1 }
                trackers += body.trackersRemoved
            } catch {
                missing += 1
            }
        }
        lines.append("bodies of newest 25: parsed \(parsed), empty \(empty), not found \(missing), trackers removed \(trackers)")
        #expect(parsed > 0)
    }

    static func describe(_ status: MailStoreStatus) -> String {
        switch status {
        case .available(let version): "available \(version)"
        case .needsFullDiskAccess: "needs Full Disk Access"
        case .notFound: "not found"
        case .unsupported(let detail): "unsupported: \(detail)"
        }
    }

    /// C-B3 "draft": writes one reply with Quick (a model run: run it under
    /// `lock.sh run` with preflight) to the newest inbox message. Records only
    /// the draft's length and timing; never sends.
    @Test(.enabled(if: ProcessInfo.processInfo.environment["MFERENCE_MAIL_LIVE_DRAFT"] == "1"))
    func draftsAReplyWithQuickWithoutSending() async throws {
        let source = AppleMailSource()
        guard case .available = await source.status() else {
            Issue.record("Mail store not readable: \(Self.describe(await source.status()))")
            return
        }
        let newest = try #require(try await source.messages(in: [.inbox], limit: 1).first)
        let body = try await source.body(of: newest)
        let engine = EngineService.live()
        let start = ContinuousClock.now
        let draft = try await MailReplyDrafter(engine: engine).draft(replyTo: newest, body: body.text)
        let elapsed = ContinuousClock.now - start
        await engine.shutdown()
        #expect(!draft.isEmpty)
        if let path = ProcessInfo.processInfo.environment["MFERENCE_MAIL_LIVE_REPORT"] {
            let line = "draft with Quick: \(draft.count) characters in \(elapsed); nothing sent\n"
            let existing = (try? String(contentsOfFile: path, encoding: .utf8)) ?? ""
            try (existing + line).write(toFile: path, atomically: true, encoding: .utf8)
        }
    }
}

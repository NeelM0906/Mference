import Foundation
import MferenceAppCore

extension AppleUIFixtures {
    // MARK: Mail

    actor FixedMailSource: MailSource {
        var status: MailStoreStatus
        var inbox: [MailMessageSummary]
        var bodies: [String: MailBody]

        init(status: MailStoreStatus, inbox: [MailMessageSummary], bodies: [String: MailBody]) {
            self.status = status
            self.inbox = inbox
            self.bodies = bodies
        }

        func reload() async {}
        func status() async -> MailStoreStatus { status }
        func messages(in kinds: Set<MailboxRef.Kind>, limit: Int) async throws -> [MailMessageSummary] {
            kinds.contains(.inbox) ? Array(inbox.prefix(limit)) : []
        }
        func body(of message: MailMessageSummary) async throws -> MailBody {
            bodies[message.messageID] ?? MailBody(text: message.snippet)
        }
        func accounts() async throws -> [MailAccount] {
            [MailAccount(id: "work", name: "work@example.com"), MailAccount(id: "home", name: "me@example.com")]
        }
    }

    static func message(_ row: Int64, _ id: String, _ name: String, _ subject: String, _ snippet: String,
                        _ date: Date) -> MailMessageSummary {
        MailMessageSummary(rowID: row, messageID: id, senderName: name,
                           senderAddress: "\(name.split(separator: " ").first!.lowercased())@example.com",
                           subject: subject, snippet: snippet, date: date, isRead: false,
                           mailbox: MailboxRef(url: "imap://work/INBOX"))
    }

    static var mailNow: Date { Calendar.current.date(bySettingHour: 10, minute: 0, second: 0, of: .now)! }

    static var inbox: [MailMessageSummary] {
        let today = Calendar.current.startOfDay(for: .now)
        return [
            message(1, "<review@example.com>", "Priya Raman", "Wednesday design review",
                    "Could we move it to 11? I’ll bring the new onboarding screens.",
                    today.addingTimeInterval(9 * 3_600 + 42 * 60)),
            message(2, "<redlines@example.com>", "Tom Okafor", "Contract redlines, round 2",
                    "Two changes left in section 4. Can you look before Friday?",
                    today.addingTimeInterval(9 * 3_600 + 18 * 60)),
            message(3, "<dental@example.com>", "Northside Dental", "Time for your six-month checkup",
                    "Reply with a few times that work and we’ll confirm.", today.addingTimeInterval(8 * 3_600 + 5 * 60)),
            message(4, "<flights@example.com>", "Lena Fischer", "Lisbon flights",
                    "Found two options under €300. Which dates are you set on?", today.addingTimeInterval(-6 * 3_600)),
            message(5, "<digest@example.com>", "Design Weekly", "This week in interface design",
                    "Five essays on calm software.", today.addingTimeInterval(-30 * 3_600)),
        ]
    }

    static let bodies = [
        "<review@example.com>": MailBody(
            text: "Hi! Could we move Wednesday’s review to 11:00? I’ll bring the new onboarding screens, and it would help to have the copy decisions settled before then.\nThanks, Priya",
            trackersRemoved: 2),
    ]

    static let sortAnswer = #"{"1": "important", "2": "important", "3": "important", "4": "important", "5": "reading"}"#
    static var eventAnswer: String {
        let wednesday = Calendar.current.nextDate(after: .now, matching: DateComponents(weekday: 4),
                                                  matchingPolicy: .nextTime)!
        let day = wednesday.formatted(.iso8601.year().month().day())
        return #"{"title": "Design review", "start": "\#(day)T11:00", "end": "\#(day)T11:45"}"#
    }

    /// An engine that answers every request with one fixed draft.
    nonisolated final class FixedEngine: EngineServicing, @unchecked Sendable {
        let answer: String
        init(answer: String) { self.answer = answer }
        func models() async -> [ModelOption] { [ModelOption(model: .quick, availability: .ready)] }
        func refreshCatalog() async {}
        var status: EngineServiceStatus { .empty }
        func statusUpdates() -> AsyncStream<EngineServiceStatus> { AsyncStream { $0.finish() } }
        func load(_ model: AppModel) async throws {}
        func unload() async throws {}
        func generate(model: AppModel, turns: [ChatTurn], options: GenerationOptions) -> GenerationStream {
            let answer = answer
            return GenerationStream { continuation in
                Task {
                    continuation.yield(.prepared(promptTokens: 10))
                    continuation.yield(.content(answer))
                    continuation.yield(.completed(GenerationSummary(content: answer, reasoning: nil, toolCalls: [],
                                                                    finish: .stop, promptTokens: 10,
                                                                    completionTokens: 20)))
                    continuation.finish()
                }
            }
        }
        func cancelActiveGeneration() async {}
        func shutdown() async {}
    }

    nonisolated struct NoPromptBroker: PermissionBroker {
        func state(of permission: AppPermission) async -> PermissionState { .notDetermined }
        func request(_ permission: AppPermission) async -> PermissionState { .denied }
        func openSystemSettings(for permission: AppPermission) async {}
    }
}

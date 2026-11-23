import Foundation
@testable import MferenceAppCore
import Testing

/// The 20-question check in both schedule formats, as requests for a run
/// through MferenceServer and as the app would show the replies. Nothing
/// here loads a model: the two file steps (export the requests, review the
/// replies) run only when their environment variable names a path.
@Suite struct WeekAskExportTests {
    @Test func requestsCoverEveryQuestionInBothFormats() throws {
        for format in WeekScheduleFormat.allCases {
            let requests = WeekAskCheckpoint.requests(format: format)
            #expect(requests.count == 20)
            #expect(requests.map(\.index) == Array(1...20))
            #expect(requests[8].messages.map(\.role) == ["system", "user"])
            #expect(requests[9].followUp)
            #expect(requests[9].messages.map(\.role) == ["user"])
        }
        let json = WeekAskCheckpoint.requests(format: .json)
        #expect(json[0].messages[1].content.hasPrefix("<USER WEEK>\n{"))
        #expect(json[9].messages[0].content == "<USER QUERY>\nonly mornings\n</USER QUERY>")
        let lines = WeekAskCheckpoint.requests(format: .lines)
        #expect(lines[9].messages[0].content == "Question: <untrusted>\nonly mornings\n</untrusted>")
    }

    @Test func theCheckpointWeeksFollowTheNewRules() {
        let work = WeekAskCheckpoint.contexts["work"]
        #expect(work?.text.contains("Team sync") == false)
        #expect(work?.text.contains("Thursday Sep 17: Offsite, all day") == true)
        #expect(work?.text.contains("Busy all day.") == true)
        let thanksgiving = WeekAskCheckpoint.contexts["thanksgiving"]
        #expect(thanksgiving?.text.contains("Thursday Nov 26: Thanksgiving, all day [E1]; Thanksgiving Day, all day [E2]. Open: 9 AM to 5 PM") == true)
    }

    @Test func reviewShowsARepliedAnswerBothWays() {
        let results = WeekAskCheckpoint.review([
            .init(format: "lines", index: 7, reply: "You are free Monday 9 AM to 5 PM and Sunday 9 PM to 12 AM. [F1]",
                  finishReason: "stop", latencyMilliseconds: 900, promptTokens: 500, completionTokens: 20),
            .init(format: "json", index: 3, reply: "", finishReason: "length", latencyMilliseconds: 10),
        ])
        #expect(results[0].shown == "You are free Monday 9 AM to 5 PM and Sunday 9 PM to 12 AM.")
        #expect(results[0].warnings == ["\u{201C}9 PM to 12 AM\u{201D} isn't in your calendar. Check it before relying on it."])
        #expect(results[0].cited == ["F1"])
        #expect(results[1].wasCut)
        #expect(results[1].error == "empty answer")
    }

    // MARK: File steps (internal testing)

    @Test(.enabled(if: ProcessInfo.processInfo.environment["MFERENCE_WEEK_ASK_EXPORT"] != nil))
    func exportRequests() throws {
        let path = try #require(ProcessInfo.processInfo.environment["MFERENCE_WEEK_ASK_EXPORT"])
        let requests = WeekScheduleFormat.allCases.flatMap { WeekAskCheckpoint.requests(format: $0) }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        try encoder.encode(requests).write(to: URL(fileURLWithPath: path))
    }

    /// Reads the server's replies and writes `<replies>.reviewed.json`: each
    /// answer as the app shows it, and whether every follow-up was sent the
    /// earlier answers exactly as the app would send them.
    @Test(.enabled(if: ProcessInfo.processInfo.environment["MFERENCE_WEEK_ASK_REPLIES"] != nil))
    func reviewReplies() throws {
        let path = try #require(ProcessInfo.processInfo.environment["MFERENCE_WEEK_ASK_REPLIES"])
        let replies = try JSONDecoder().decode([WeekAskCheckpoint.Reply].self,
                                               from: Data(contentsOf: URL(fileURLWithPath: path)))
        var reviewed: [String: [WeekAskCheckpoint.Result]] = [:]
        var historyMismatches: [String] = []
        for format in WeekScheduleFormat.allCases {
            let mine = replies.filter { $0.format == format.rawValue }.sorted { $0.index < $1.index }
            let results = WeekAskCheckpoint.review(mine)
            reviewed[format.rawValue] = results
            var thread: [String] = []
            for (reply, result) in zip(mine, results) {
                if result.followUp {
                    if reply.history != thread { historyMismatches.append("\(format.rawValue) Q\(reply.index)") }
                } else {
                    thread = []
                }
                thread.append(result.shown)
            }
        }
        struct Output: Encodable {
            var results: [String: [WeekAskCheckpoint.Result]]
            var historyMismatches: [String]
        }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        try encoder.encode(Output(results: reviewed, historyMismatches: historyMismatches))
            .write(to: URL(fileURLWithPath: path + ".reviewed.json"))
    }
}

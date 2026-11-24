import Foundation
import MferenceAppCore
import MferenceWeekAskFixtures
import Synchronization
import Testing

/// The real-model checkpoint's questions, key and grader, run with a
/// scripted model: nothing is loaded.
@Suite struct WeekAskCheckpointTests {
    /// Answers each question with the key's expected reply.
    final class KeyModel: WeekLanguageModel {
        let replies: [String: String]
        let asked = Mutex<[[ChatTurn]]>([])

        init() {
            var replies: [String: String] = [:]
            for question in WeekAskCheckpoint.questions {
                guard let expected = question.expected else { continue }
                replies[question.text] = expected.routes.first == .freeQuery
                    ? expected.queries.first?.json ?? #"{"free_query": {}}"#
                    : #"{"answer": "Per the key.", "refs": []}"#
            }
            self.replies = replies
        }

        var label: String { "Key" }

        func respond(turns: [ChatTurn], maximumTokens: Int,
                     started: @escaping @Sendable () async -> Void) async throws -> WeekModelReply {
            asked.withLock { $0.append(turns) }
            let last = turns.last?.content ?? ""
            let question = last.components(separatedBy: "<USER QUERY>\n").last?
                .components(separatedBy: "\n</USER QUERY>").first ?? ""
            return WeekModelReply(content: replies[question] ?? "")
        }
    }

    @Test func theSetCoversAboutSixtyQuestionsOverThreeWeeks() {
        let questions = WeekAskCheckpoint.questions
        let typed = questions.filter { $0.chip == nil }
        #expect(typed.count >= 60)
        #expect(Set(questions.map(\.id)).count == questions.count)
        #expect(Set(questions.map(\.week)) == ["thanksgiving", "work", "workThursday", "busy"])
        #expect(typed.allSatisfy { $0.expected != nil })
        // The past-day gate, with today Tuesday and Thursday: five each.
        let gate = typed.filter { $0.id.hasPrefix("G") }
        #expect(gate.filter { $0.week == "work" }.map(\.text) == gate.filter { $0.week == "workThursday" }.map(\.text))
        #expect(gate.count == 10)
        // A follow-up always continues a thread in the same week.
        for (index, question) in questions.enumerated() where question.followUp {
            #expect(index > 0 && questions[index - 1].week == question.week)
        }
    }

    @Test func everyFreeTimeKeyHasASwiftAnswer() throws {
        let contexts = WeekAskCheckpoint.contexts
        for question in WeekAskCheckpoint.questions {
            guard let expected = question.expected, expected.routes.first == .freeQuery else { continue }
            let context = try #require(contexts[question.week])
            let query = try #require(expected.queries.first)
            #expect(!FreeTimeSearch.run(query, in: context.week).headline.isEmpty)
        }
        #expect(WeekAskCheckpoint.keyLines().count == WeekAskCheckpoint.questions.count)
    }

    @Test func theBusyWeekIsRealistic() throws {
        let context = try #require(WeekAskCheckpoint.contexts["busy"])
        #expect(context.week.events.count == 35)
        #expect(context.overlaps.count == 1)
        #expect(context.busiest?.days.map { context.week.text.weekday($0.date) } == ["Friday"])
        #expect(context.pastDays.count == 2)
    }

    @Test func normalizingTreatsDefaultsWrittenOutAsLeftOut() throws {
        let week = WeekAskCheckpoint.workWeek
        let spelled = FreeQuery(minMinutes: 30, days: ["2026-09-18", "2026-09-15", "2026-09-16", "2026-09-17"])
        #expect(WeekAskCheckpoint.normalized(spelled, in: week) == FreeQuery())
        #expect(WeekAskCheckpoint.normalized(FreeQuery(days: ["2026-09-18"]), in: week) == FreeQuery(days: ["2026-09-18"]))
    }

    @Test func aRunGradesShapeFieldsAndTheGate() async throws {
        let model = KeyModel()
        let results = await WeekAskCheckpoint.run(model: model, log: nil)
        #expect(results.count == WeekAskCheckpoint.questions.count)
        let typed = results.filter { !$0.chip }
        #expect(typed.allSatisfy { $0.error == nil })
        #expect(typed.allSatisfy { $0.shapeCorrect == true })
        #expect(typed.allSatisfy { $0.jsonParsed == true })
        #expect(typed.filter { $0.fieldsCorrect != nil }.allSatisfy { $0.fieldsCorrect == true })
        // The key model cites nothing: "must not cite" passes, "must cite"
        // does not.
        #expect(typed.filter { $0.gate == "mustNotCite" }.allSatisfy { $0.gatePassedByRefs == true })
        #expect(typed.filter { $0.gate == "mustCite" }.allSatisfy { $0.gatePassedByRefs == false })
        let summary = WeekAskCheckpoint.summary(results)
        #expect(summary.typed == typed.count)
        #expect(summary.shapeCorrect == typed.count)
        #expect(summary.jsonParsed == typed.count)
        // Chips never reach the model; follow-ups replay the thread.
        let asked = model.asked.withLock { $0 }
        #expect(asked.count == typed.count)
        let narrowing = try #require(asked.first { $0.last?.content?.contains("only afternoons") == true
            && $0.count == 4 && $0[2].content == #"{"free_query": {"min_minutes": 120}}"# })
        #expect(narrowing.map(\.role) == [.system, .user, .assistant, .user])
    }

    @Test func aWrongShapeOrFieldIsGraded() throws {
        let context = try #require(WeekAskCheckpoint.contexts["work"])
        let question = try #require(WeekAskCheckpoint.questions.first { $0.id == "W09" })
        var answer = WeekAnswerer.freeTime(FreeQuery(minMinutes: 90, days: ["2026-09-16"]), question: question.text,
                                           context: context, askedBy: "m")
        answer.trace = WeekAnswerTrace(model: "m", reply: "", route: .freeQuery, jsonParsed: true, wasCut: false,
                                       latencyMilliseconds: 1)
        let wrongField = WeekAskCheckpoint.grade(answer, question: question, context: context)
        #expect(wrongField.shapeCorrect == true)
        #expect(wrongField.fieldsCorrect == false)
        answer.trace?.route = .answer
        let wrongShape = WeekAskCheckpoint.grade(answer, question: question, context: context)
        #expect(wrongShape.shapeCorrect == false)
    }

    @Test func onlyAsksTheNamedQuestionsWithTheirThreads() {
        #expect(WeekAskCheckpoint.selected(nil).count == WeekAskCheckpoint.questions.count)
        #expect(WeekAskCheckpoint.selected(["G04", "G06", "G07"]).map(\.id) == ["G04", "G06", "G07"])
        // A follow-up brings the questions before it in its thread.
        let thread = WeekAskCheckpoint.selected(["C03"]).map(\.id)
        #expect(thread == ["C01", "C02", "C03"])
        #expect(WeekAskCheckpoint.selected(["nope"]).isEmpty)
    }

    /// Writes the answer key (and each free-time question's Swift answer)
    /// for the record: `MFERENCE_WEEK_ASK_KEY=<file> Scripts/test.sh ...`.
    @Test(.enabled(if: ProcessInfo.processInfo.environment["MFERENCE_WEEK_ASK_KEY"] != nil))
    func exportKey() throws {
        let path = try #require(ProcessInfo.processInfo.environment["MFERENCE_WEEK_ASK_KEY"])
        try (WeekAskCheckpoint.keyLines().joined(separator: "\n") + "\n")
            .write(toFile: path, atomically: true, encoding: .utf8)
    }
}

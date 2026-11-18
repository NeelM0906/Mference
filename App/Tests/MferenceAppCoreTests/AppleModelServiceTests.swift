import Foundation
@testable import MferenceAppCore
import MferenceEngine
import Testing

@Suite struct AppleModelServiceTests {
    static let unavailableApple = AssistantAvailability.unavailable(reason: "Apple Intelligence is turned off.")

    static func quickEngine(answer: String, installed: Bool = true) -> EngineService {
        let script = FakeEngineScript(events: [.content(answer)])
        let journal = FakeEngineJournal()
        let llama = FakeEngineBackend(engine: .llamaCpp, journal: journal,
                                      scripts: [AppModel.quick.modelID: script])
        let catalog = EngineCatalog(models: installed ? [
            FakeEngineBackend.descriptor(id: AppModel.quick.modelID, engine: .llamaCpp,
                                         product: EngineProductSlot(modelID: AppModel.quick.modelID)),
        ] : [])
        return EngineService(discover: { catalog }) { catalog, observer in
            EngineHost(catalog: catalog, backends: [llama], idleUnloadAfter: nil, observer: observer)
        }
    }

    @Test func appleModelAnswersWhenAvailable() async throws {
        let apple = ScriptedAssistantTextModel(kind: .appleOnDevice, answer: "\"Wednesday design review.\"")
        let quick = ScriptedAssistantTextModel(kind: .quick, answer: "unused")
        let service = AppleModelService(primary: apple, fallback: quick)
        let result = try await service.title(for: "Could we move Wednesday's review to 11?")
        #expect(result == AssistantResult(value: "Wednesday design review", model: .appleOnDevice))
        #expect(quick.requests.isEmpty)
        #expect(try await service.activeModel().get() == .appleOnDevice)
    }

    @Test func fallsBackToQuickThroughTheEngineWhenAppleIsOff() async throws {
        let apple = ScriptedAssistantTextModel(kind: .appleOnDevice, availability: Self.unavailableApple,
                                               answer: "unused")
        let engine = Self.quickEngine(answer: "Lisbon flights under 300 euros")
        let service = AppleModelService(primary: apple, fallback: QuickTextModel(engine: engine))
        #expect(try await service.activeModel().get() == .quick)
        let result = try await service.title(for: "Found two options under €300.")
        #expect(result == AssistantResult(value: "Lisbon flights under 300 euros", model: .quick))
        #expect(apple.requests.isEmpty)
        await engine.shutdown()
    }

    @Test func fallsBackToQuickWhenAppleDeclinesOrAnswersBadly() async throws {
        let apple = ScriptedAssistantTextModel(kind: .appleOnDevice, answers: [
            .failure(.declined("Apple's on-device model declined this text.")),
            .success("no json here"),
        ])
        let quick = ScriptedAssistantTextModel(kind: .quick, answers: [
            .success("A short summary."),
            .success(#"{"merchant": "Corner Hardware", "total": "44.49"}"#),
        ])
        let service = AppleModelService(primary: apple, fallback: quick)
        #expect(try await service.summary(of: "Long text").model == .quick)
        let receipt = try await service.fillReceipt(from: "CORNER HARDWARE TOTAL 44.49")
        #expect(receipt.model == .quick)
        #expect(receipt.value.merchant == "Corner Hardware")
        #expect(receipt.value.total == Decimal(string: "44.49"))
    }

    @Test func explainsWhyNothingCanRun() async {
        let apple = ScriptedAssistantTextModel(kind: .appleOnDevice, availability: Self.unavailableApple,
                                               answer: "unused")
        let engine = Self.quickEngine(answer: "unused", installed: false)
        let service = AppleModelService(primary: apple, fallback: QuickTextModel(engine: engine))
        guard case .failure(.noModelAvailable(let reason)) = await service.activeModel() else {
            Issue.record("expected no model"); return
        }
        #expect(reason.hasPrefix("Apple Intelligence is off and Quick is not installed."))
        await #expect(throws: AssistantModelError.self) { try await service.title(for: "x") }
        await engine.shutdown()
    }

    @Test func keepsTheRealFailureOverALaterUnavailability() async {
        let apple = ScriptedAssistantTextModel(kind: .appleOnDevice,
                                               answers: [.failure(.declined("Declined."))])
        let quick = ScriptedAssistantTextModel(kind: .quick, availability: .unavailable(reason: "No Quick."),
                                               answer: "unused")
        let service = AppleModelService(primary: apple, fallback: quick)
        await #expect(throws: AssistantModelError.declined("Declined.")) { try await service.title(for: "x") }
    }

    @Test func sortsMailInBatchesAndDropsUnknownBuckets() async throws {
        let messages = (1...10).map {
            MailSortItem(messageID: "<m\($0)@example.com>", sender: "Sender \($0)", subject: "Subject \($0)",
                         snippet: "Body \($0)")
        }
        let buckets = [MailSortBucket(key: "important", description: "Needs the person's attention"),
                       MailSortBucket(key: "other", description: "Everything else")]
        let first = (1...8).map { "\"\($0)\": \"\($0 == 3 ? "nonsense" : ($0.isMultiple(of: 2) ? "important" : "other"))\"" }
        let apple = ScriptedAssistantTextModel(kind: .appleOnDevice, answers: [
            .success("```json\n{\(first.joined(separator: ", "))}\n```"),
            .success(#"Sure! {"1": "important", "2": "other", "9": "important"}"#),
        ])
        let service = AppleModelService(primary: apple, fallback: nil)
        let result = try await service.sortMail(messages, into: buckets)
        #expect(apple.requests.count == 2)
        #expect(result.model == .appleOnDevice)
        #expect(result.value["<m2@example.com>"] == "important")
        #expect(result.value["<m1@example.com>"] == "other")
        #expect(result.value["<m3@example.com>"] == nil)
        #expect(result.value["<m9@example.com>"] == "important")
        #expect(result.value["<m10@example.com>"] == "other")
        #expect(result.value.count == 9)
        #expect(apple.requests[0].prompt.contains("important: Needs the person's attention"))
    }

    @Test func readsReceiptJSONWithNullsAndFormats() async throws {
        let answer = """
            Here are the fields:
            {"merchant": "Corner Hardware", "date": "09/12/2026", "total": "$44.49", "tax": 3.53,
             "currency": "usd", "category": "Home improvement", "paidWith": "Visa",
             "lineItems": [{"name": "Wood glue 8oz", "amount": "6.49"}, {"name": "Note {x}", "amount": null}]}
            """
        let service = AppleModelService(primary: ScriptedAssistantTextModel(kind: .appleOnDevice, answer: answer),
                                        fallback: nil)
        let draft = try await service.fillReceipt(from: "text").value
        #expect(draft.date == "2026-09-12")
        #expect(draft.total == Decimal(string: "44.49"))
        #expect(draft.tax == Decimal(string: "3.53"))
        #expect(draft.currency == "USD")
        #expect(draft.lineItems == [ReceiptLineItem(name: "Wood glue 8oz", amount: Decimal(string: "6.49")),
                                    ReceiptLineItem(name: "Note {x}", amount: nil)])
    }

    @Test func parsesAmountsAndDates() {
        #expect(ReceiptValueParser.decimal("1,234.50") == Decimal(string: "1234.50"))
        #expect(ReceiptValueParser.decimal("44,49 €") == Decimal(string: "44.49"))
        #expect(ReceiptValueParser.decimal("1.234,50") == Decimal(string: "1234.50"))
        #expect(ReceiptValueParser.decimal("n/a") == nil)
        #expect(ReceiptValueParser.isoDate("2026-09-12") == "2026-09-12")
        #expect(ReceiptValueParser.isoDate("12.09.2026") == "2026-09-12")
        #expect(ReceiptValueParser.isoDate("09/12/26") == "2026-09-12")
        #expect(ReceiptValueParser.isoDate("02/30/2026") == nil)
    }

    @Test func clipsLongInputAtAWord() {
        let clipped = AssistantPrompts.clip(String(repeating: "word ", count: 100), to: 22)
        #expect(clipped == "word word word word…")
    }

    @Test func eventSuggestionsParseTimesInTheMailTimeZone() async throws {
        let model = ScriptedAssistantTextModel(kind: .appleOnDevice, answers: [
            .success(#"{"title": "Checkup", "start": "2026-09-17T15:30", "end": null}"#),
            .success(#"{"title": null}"#),
        ])
        let service = AppleModelService(primary: model, fallback: nil)
        let zone = TimeZone(identifier: "Europe/Lisbon")!
        let found = try await service.suggestEvent(subject: "Checkup", body: "Thursday 3:30?", received: .now,
                                                   timeZone: zone).value
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = zone
        #expect(found?.start == calendar.date(from: DateComponents(year: 2026, month: 9, day: 17, hour: 15, minute: 30)))
        #expect(abs((found.map { $0.end.timeIntervalSince($0.start) } ?? 0) - 45 * 60) < 0.001)
        let none = try await service.suggestEvent(subject: "Hi", body: "No meeting.", received: .now, timeZone: zone)
        #expect(none.value == nil)
        #expect(model.requests[0].prompt.contains("Received: "))
    }
}

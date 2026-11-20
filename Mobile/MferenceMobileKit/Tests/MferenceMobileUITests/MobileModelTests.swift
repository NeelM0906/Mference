import Foundation
import Testing
@testable import MferenceMobileUI
import MferenceMobileSamples

@Suite("Fold geometry")
struct FoldGeometryTests {
    @Test func phonesAndClosedDuoAreCompact() {
        #expect(FoldGeometry(width: 402).mode == .compact)
        #expect(FoldGeometry(width: 440).mode == .compact)
        #expect(FoldGeometry(width: 466).mode == .compact)
    }

    @Test func openDuoSplitsAtTheFoldWithAKeepOut() {
        let fold = FoldGeometry(width: 951)
        #expect(fold.mode == .twoPane)
        #expect(fold.foldX == 475.5)
        #expect(fold.leadingPaneWidth == 463.5)
        #expect(fold.trailingPaneX == 487.5)
        #expect(fold.trailingPaneWidth == 463.5)
    }

    @Test func aMeasuredFoldOverridesTheMidline() {
        let fold = FoldGeometry(width: 900, foldX: 440)
        #expect(fold.leadingPaneWidth == 428)
        #expect(fold.trailingPaneX == 452)
    }
}

@Suite("Chat helpers")
struct ChatHelperTests {
    @Test func manyDocumentsGoToTheMac() {
        #expect(HandoffAdvisor.reason(for: "Go through the 40 board PDFs and pull every revenue figure") != nil)
        #expect(HandoffAdvisor.reason(for: "Summarise these 120 documents") != nil)
    }

    @Test func everydayRequestsStayOnTheIPhone() {
        #expect(HandoffAdvisor.reason(for: "What did Priya ask for?") == nil)
        #expect(HandoffAdvisor.reason(for: "Read this one PDF") == nil)
    }

    @Test func citationsSplitOut() {
        let runs = CitationText.runs("On sale through Sunday [2], cheaper [10].")
        #expect(runs == [.text("On sale through Sunday "), .citation(2), .text(", cheaper "), .citation(10), .text(".")])
        #expect(CitationText.plain("Best price [2].") == "Best price.")
    }

    @Test func engineLabelsSayWhereTheModelRan() {
        #expect(PhoneModel.quick.engineLabel == "Quick 4B · on this iPhone")
        #expect(PhoneModel.deep.engineLabel == "Deep 27B · on your Mac")
    }
}

@Suite("Month grid")
struct MonthGridTests {
    private func calendar(firstWeekday: Int) -> Calendar {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(identifier: "UTC")!
        calendar.locale = Locale(identifier: "en_US")
        calendar.firstWeekday = firstWeekday
        return calendar
    }

    @Test func september2026StartsOnTuesday() throws {
        let calendar = calendar(firstWeekday: 2)
        let month = try #require(calendar.date(from: DateComponents(year: 2026, month: 9, day: 15)))
        let grid = MonthGrid(month: month, calendar: calendar)
        #expect(grid.cells.count % 7 == 0)
        #expect(grid.cells.prefix(1).allSatisfy { $0 == nil })
        #expect(grid.cells[1].map { calendar.component(.day, from: $0) } == 1)
        #expect(grid.cells.compactMap { $0 }.count == 30)
        #expect(grid.weekdaySymbols.first == "M")
    }

    @Test func sundayFirstWeekPads2Days() throws {
        let calendar = calendar(firstWeekday: 1)
        let month = try #require(calendar.date(from: DateComponents(year: 2026, month: 9, day: 1)))
        let grid = MonthGrid(month: month, calendar: calendar)
        #expect(grid.cells.prefix(2).allSatisfy { $0 == nil })
        #expect(grid.weekdaySymbols.first == "S")
    }

    @Test func aWeekHasSevenDays() throws {
        let calendar = calendar(firstWeekday: 2)
        let day = try #require(calendar.date(from: DateComponents(year: 2026, month: 9, day: 15)))
        let week = MonthGrid.week(containing: day, calendar: calendar)
        #expect(week.count == 7)
        #expect(calendar.component(.day, from: week[0]) == 14)
    }
}

@Suite("App model")
struct MobileAppModelTests {
    private func makeModel(pairedMac: PairedMac? = .sample) -> MobileAppModel {
        MobileAppModel(calendar: SampleCalendarModel(), scan: SampleScanModel(), pairedMac: pairedMac,
                       wordDelay: .zero)
    }

    @Test func calendarAndScanComeInAsTheirScreenModels() {
        let calendar = SampleCalendarModel()
        let scan = SampleScanModel()
        let model = MobileAppModel(calendar: calendar, scan: scan, wordDelay: .zero)
        #expect(model.calendar === calendar)
        #expect(model.scan === scan)
        // The views' `select(day:)` keeps the current view.
        calendar.calendarMode = .month
        model.calendar.select(day: .now)
        #expect(calendar.calendarMode == .month)
    }

    private func settle(_ model: MobileAppModel) async {
        for _ in 0..<200 where model.currentMessages.contains(where: \.isStreaming) {
            try? await Task.sleep(for: .milliseconds(5))
        }
    }

    @Test func chatOpensToANewChat() {
        let model = makeModel()
        #expect(model.currentConversationID == nil)
        #expect(model.chatTitle == "New chat")
        #expect(model.selection == .chat)
    }

    @Test func sendingStartsAConversationAndStreamsAReply() async {
        let model = makeModel()
        model.draft = "What did Priya ask for in her email?"
        model.send()
        #expect(model.draft.isEmpty)
        #expect(model.currentConversation?.title == "What did Priya ask for")
        await settle(model)
        let reply = model.currentMessages.last
        #expect(reply?.role == .assistant)
        #expect(reply?.isStreaming == false)
        #expect(reply?.engineLabel == "Quick 4B · on this iPhone")
        #expect(reply?.text.contains("11:00") == true)
    }

    @Test func webAnswersSayTheyWentThroughTheMac() async {
        let model = makeModel()
        model.webSearchViaMac = true
        model.send("Compare standing desk warranties")
        await settle(model)
        #expect(model.currentMessages.last?.webSteps?.hasPrefix("Searched via paired Mac") == true)
        #expect(model.currentMessages.last?.sources.count == 3)
    }

    @Test func bigJobsOfferTheMacAndRunThereWhenSent() async throws {
        let model = makeModel()
        model.send("Go through the 40 board PDFs and pull every revenue figure into one table.")
        let offer = try #require(model.currentMessages.last)
        #expect(offer.handoff?.state == .offered)
        model.resolveHandoff(offer.id, sendToMac: true)
        #expect(model.currentMessages.first { $0.id == offer.id }?.handoff?.state == .sentToMac)
        await settle(model)
        #expect(model.currentMessages.last?.engineLabel == "Deep 27B · on your Mac")
        #expect(model.currentConversation?.isRunningOnMac == false)
        #expect(model.currentMessages.first { $0.id == offer.id }?.handoff?.state == .returnedFromMac)
    }

    @Test func aLateHandoffAnswersItsOwnRequest() async throws {
        let model = makeModel()
        model.send("Go through the 40 board PDFs and pull every revenue figure into one table.")
        let offer = try #require(model.currentMessages.last)
        #expect(offer.handoff?.state == .offered)
        model.send("What did Priya ask for in her email?")
        await settle(model)
        model.resolveHandoff(offer.id, sendToMac: true)
        await settle(model)
        let last = try #require(model.currentMessages.last)
        #expect(last.engineLabel == "Deep 27B · on your Mac")
        #expect(last.text.contains("Revenue"))
        #expect(!last.text.contains("Priya"))
    }

    @Test func withoutAPairedMacBigJobsStayOnTheIPhone() async {
        let model = makeModel(pairedMac: nil)
        #expect(model.macModels.isEmpty)
        model.send("Go through the 40 board PDFs and pull every revenue figure into one table.")
        #expect(model.currentMessages.last?.handoff == nil)
        await settle(model)
    }
}

@Suite("Time ranges")
struct TimeRangeTextTests {
    private let calendar: Calendar = {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(identifier: "UTC")!
        return calendar
    }()

    private func at(_ hour: Int, _ minute: Int = 0) -> Date {
        calendar.date(from: DateComponents(year: 2026, month: 9, day: 15, hour: hour, minute: minute))!
    }

    @Test func twelveHourRangesReadLikeTheMac() {
        let us = Locale(identifier: "en_US")
        #expect(TimeRangeText.text(from: at(10), to: at(11), calendar: calendar, locale: us) == "10 to 11 AM")
        #expect(TimeRangeText.text(from: at(9, 30), to: at(10), calendar: calendar, locale: us) == "9:30 to 10 AM")
        #expect(TimeRangeText.text(from: at(11, 30), to: at(13), calendar: calendar, locale: us) == "11:30 AM to 1 PM")
        #expect(TimeRangeText.text(from: at(12), to: at(13), calendar: calendar, locale: us) == "12 to 1 PM")
    }
}

import Foundation
@testable import MferenceAppCore
import Testing

/// Every chip is answered in Swift, with no model call, and stores the
/// record a typed question would (spec "Chips"). `CalendarFixtures`: now
/// Tuesday Sep 15, 8 AM; Offsite takes Thursday.
@Suite struct WeekChipsTests {
    static let context = WeekContextTests.context()
    static var text: WeekText { context.week.text }

    static func answer(_ chip: WeekChip, in context: WeekContext = context) -> WeekAnswer {
        WeekAnswerer.answer(chip, context: context)
    }

    @Test func freeForTwoHoursRunsTheFreeTimeSearch() {
        let answer = Self.answer(.freeFor(minutes: 120))
        #expect(answer.question == "When am I free for 2 hours?")
        #expect(answer.sentence == "6 openings of 2 h or more: Tue, Wed, Fri.")
        #expect(answer.openTimes.count == 6)
        #expect(answer.record == .freeQuery(FreeQuery(minMinutes: 120)))
        #expect(answer.source == .calendar(askedBy: nil))
        #expect(answer.trace == nil)
        #expect(WeekChip.freeFor(minutes: 90).title(Self.text) == "When am I free for 90 minutes?")
    }

    @Test func freeOnADaySearchesThatDay() {
        let today = Self.answer(.freeOn(CalendarFixtures.date(15, 0)))
        #expect(today.question == "Am I free today?")
        #expect(today.sentence == "3 openings today: 9 to 9:30 AM, 10 AM to 12 PM and 1 to 5 PM.")
        #expect(today.freeQuery == FreeQuery(days: ["2026-09-15"]))
        #expect(WeekChip.freeOn(CalendarFixtures.date(18, 0)).title(Self.text) == "Am I free on Friday?")
        #expect(WeekChip.freeOn(CalendarFixtures.date(16, 0)).title(Self.text) == "Am I free tomorrow?")
    }

    @Test func whatsOnADayListsItsEventsBookedAndOpenTime() {
        let today = Self.answer(.whatsOn(CalendarFixtures.date(15, 0)))
        #expect(today.question == "What's on today?")
        #expect(today.sentence == "Today: 2 events, 1 h 30 min booked, 6 h 30 min open.")
        #expect(today.rows.prefix(2).map(\.id) == ["E1", "E2"])
        #expect(today.openTimes.count == 3)
        #expect(today.record == .lookup(answer: today.sentence, refs: Array(today.rows.prefix(2))))
        let thursday = Self.answer(.whatsOn(CalendarFixtures.date(17, 0)))
        #expect(thursday.sentence == "Thursday: 2 events, busy all day with Offsite.")
        #expect(thursday.eventIDs == ["holiday", "launch"])
        let monday = Self.answer(.whatsOn(CalendarFixtures.date(14, 0)))
        #expect(monday.sentence == "Monday: 2 events, 2 h 30 min booked.")
        #expect(monday.openTimes.isEmpty)
    }

    @Test func tomorrowIsWhatsOnForTomorrow() {
        let answer = Self.answer(.tomorrow)
        #expect(answer.question == "What's tomorrow like?")
        #expect(answer.sentence == "Tomorrow: 2 events, 2 h booked, 6 h open.")
        // On the week's last day, tomorrow is in the next week.
        let sunday = WeekContextTests.context(now: CalendarFixtures.date(20, 9))
        #expect(Self.answer(.tomorrow, in: sunday).sentence
            == "I'm looking at Sep 14 to 20. Go to that week to check tomorrow.")
    }

    @Test func busiestDayIsOwnAllDayFirstElseMostBooked() {
        let answer = Self.answer(.busiest)
        #expect(answer.sentence == "Thursday is your busiest day left: Offsite takes the whole day.")
        #expect(answer.eventIDs == ["holiday", "launch"])
        let open = WeekContextTests.context(events: WeekContextTests.subscribedOffsite)
        #expect(Self.answer(.busiest, in: open).sentence == "Friday is your busiest day left, with 2 h 30 min booked.")
        let empty = WeekContextTests.context(events: [])
        #expect(Self.answer(.busiest, in: empty).sentence == "Nothing is booked for the rest of this week.")
    }

    @Test func dueListsDueAndOverdueToDos() {
        let answer = Self.answer(.due)
        #expect(answer.sentence == "2 to-dos are due this week: Send copy decisions to Priya (today) and "
            + "Review contract section 4 (Friday). Overdue: Renew passport.")
        #expect(answer.rows.map(\.id) == ["T1", "T2", "T3"])
        #expect(Self.answer(.due, in: WeekContextTests.context(todos: [])).sentence == "Nothing is due this week.")
    }

    @Test func overlapsNameBothEvents() {
        #expect(Self.answer(.overlaps).sentence == "Nothing overlaps for the rest of this week.")
        var events = CalendarFixtures.events
        events.append(CalendarEvent(id: "lunch-2", title: "Lunch with Ana", start: CalendarFixtures.date(15, 12, 30),
                                    end: CalendarFixtures.date(15, 13, 30)))
        let answer = Self.answer(.overlaps, in: WeekContextTests.context(events: events))
        #expect(answer.sentence == "Lunch and Lunch with Ana overlap today, 12:30 to 1 PM.")
        #expect(answer.eventIDs == ["lunch", "lunch-2"])
    }

    @Test func chipsFollowTheView() {
        let text = Self.text
        #expect(WeekChip.suggestions(dayView: false, selected: CalendarFixtures.now, text: text)
            == [.freeFor(minutes: 120), .busiest, .due, .overlaps])
        let today = CalendarFixtures.date(15, 0)
        #expect(WeekChip.suggestions(dayView: true, selected: today, text: text)
            == [.whatsOn(today), .freeOn(today), .tomorrow, .due, .busiest])
        let friday = CalendarFixtures.date(18, 0)
        #expect(WeekChip.suggestions(dayView: true, selected: friday, text: text)
            == [.whatsOn(friday), .freeOn(friday), .due, .busiest])
        #expect(WeekChip.whatsOn(friday).title(text) == "What's on Friday?")
    }
}

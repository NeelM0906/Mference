import Foundation
@testable import MferenceAppCore
import Testing

/// The facts behind Ask about your week: plain Swift, checked against the
/// mockup's week (`CalendarFixtures`, now = Tuesday Sep 15, 8 AM).
@Suite struct WeekFactsTests {
    static let calendar = CalendarFixtures.calendar
    static func date(_ day: Int, _ hour: Int) -> Date { CalendarFixtures.date(day, hour) }

    static let todos: [TodoItem] = [
        TodoItem(title: "Send copy decisions to Priya", due: date(15, 0)),
        TodoItem(title: "Review contract section 4", due: date(18, 0)),
        TodoItem(title: "Expense the team lunch", isDone: true, due: date(16, 0)),
        TodoItem(title: "Pick a dentist appointment"),
        TodoItem(title: "Renew passport", due: date(10, 0)),
    ]

    static func week(now: Date = CalendarFixtures.now, events: [CalendarEvent] = CalendarFixtures.events,
                     todos: [TodoItem] = todos, calendar: Calendar = calendar) -> WeekSnapshot {
        let layout = WeekLayout(calendar: calendar)
        return WeekSnapshot(interval: layout.interval(from: layout.weekStart(containing: now)), now: now,
                            calendar: calendar, events: events, todos: todos)
    }

    static func ranges(_ answer: WeekAnswerFacts) -> [String] {
        let text = WeekText(calendar: calendar, now: CalendarFixtures.now)
        return answer.openTimes.map { "\(text.shortWeekday($0.start)) \(text.range($0.start, $0.end))" }
    }

    @Test func weekLeavesOutOtherWeeksEvents() {
        let week = Self.week()
        #expect(!week.events.contains { $0.id == "next-week" })
        #expect(week.days.count == 7)
        #expect(week.text.weekRange(week.interval) == "Sep 14 to 20")
    }

    @Test func freeTimeLooksFromNowInWorkHoursOnWeekdays() {
        let answer = WeekFacts.answer(.freeTime(FreeTimeRequest(minimumMinutes: 120)), in: Self.week())
        #expect(Self.ranges(answer) == ["Tue 10 AM to 12 PM", "Tue 1 to 5 PM", "Wed 11 AM to 1 PM", "Wed 2 to 5 PM",
                                        "Thu 9 to 11 AM", "Thu 12 to 5 PM", "Fri 11 AM to 1 PM", "Fri 1:30 to 5 PM"])
        #expect(answer.sentence == "You have 8 open blocks of 2 h or more for the rest of the week. "
            + "The earliest is today, 10 AM to 12 PM.")
        #expect(answer.facts.map(\.id).prefix(2) == ["F1", "F2"])
        #expect(answer.facts[0].line == "F1: today (Tuesday Sep 15), 10 AM to 12 PM, 2 h")
        #expect(answer.notes.contains("longest: F6"))
    }

    @Test func freeTimeStartsAtTheNextQuarterHour() {
        let week = Self.week(now: CalendarFixtures.date(15, 10, 17))
        let answer = WeekFacts.answer(.freeTime(FreeTimeRequest(minimumMinutes: 60)), in: week)
        #expect(Self.ranges(answer).first == "Tue 10:30 AM to 12 PM")
    }

    @Test func morningsEndAtNoon() {
        let request = FreeTimeRequest(minimumMinutes: 120, part: .morning)
        let answer = WeekFacts.answer(.freeTime(request), in: Self.week())
        #expect(Self.ranges(answer) == ["Tue 10 AM to 12 PM", "Thu 9 to 11 AM"])
        #expect(answer.sentence.contains("in the mornings"))
    }

    @Test func beforeAndOneDayNarrowTheSearch() {
        let before = FreeTimeRequest(minimumMinutes: 120, before: CalendarFixtures.date(17, 12))
        #expect(Self.ranges(WeekFacts.answer(.freeTime(before), in: Self.week()))
            == ["Tue 10 AM to 12 PM", "Tue 1 to 5 PM", "Wed 11 AM to 1 PM", "Wed 2 to 5 PM", "Thu 9 to 11 AM"])
        let wednesday = FreeTimeRequest(minimumMinutes: 60, day: CalendarFixtures.date(16, 0))
        let answer = WeekFacts.answer(.freeTime(wednesday), in: Self.week())
        #expect(Self.ranges(answer) == ["Wed 9 to 10 AM", "Wed 11 AM to 1 PM", "Wed 2 to 5 PM"])
        #expect(answer.sentence.hasPrefix("You have 3 open blocks of 1 h or more tomorrow."))
    }

    @Test func noLongEnoughBlockOffersTheLongest() {
        let answer = WeekFacts.answer(.freeTime(FreeTimeRequest(minimumMinutes: 360)), in: Self.week())
        #expect(answer.sentence == "No open block of 6 h for the rest of the week. "
            + "The longest is Thursday, 12 to 5 PM (5 h).")
        #expect(Self.ranges(answer) == ["Thu 12 to 5 PM"])
    }

    @Test func overlappingEventsBlockTimeOnce() {
        var events = CalendarFixtures.events
        events.append(CalendarEvent(id: "lunch-2", title: "Lunch with Ana", start: CalendarFixtures.date(15, 12, 30),
                                    end: CalendarFixtures.date(15, 13, 30)))
        let request = FreeTimeRequest(minimumMinutes: 60, day: CalendarFixtures.date(15, 0))
        #expect(Self.ranges(WeekFacts.answer(.freeTime(request), in: Self.week(events: events)))
            == ["Tue 10 AM to 12 PM", "Tue 1:30 to 5 PM"])
    }

    @Test func compareDaysSkipsPastDaysAndNamesTies() {
        let answer = WeekFacts.answer(.compareDays, in: Self.week())
        #expect(answer.sentence == "Friday is your busiest day left, with 2 h 30 min booked.")
        #expect(answer.facts.count == 5)
        #expect(answer.facts[0].line == "D1: Monday Sep 14 (past), 2 h 30 min booked, 2 events")
        #expect(answer.notes == ["busiest days left: D5"])

        var events = CalendarFixtures.events
        events.append(CalendarEvent(id: "extra", title: "Review", start: CalendarFixtures.date(16, 15),
                                    end: CalendarFixtures.date(16, 15, 30)))
        #expect(WeekFacts.answer(.compareDays, in: Self.week(events: events)).sentence
            == "Tomorrow and Friday tie as your busiest days left, with 2 h 30 min booked each.")
    }

    @Test func findEventMatchesWordsAndListsEveryMatch() {
        #expect(WeekFacts.answer(.findEvent(name: "contract"), in: Self.week()).sentence
            == "Contract call is Friday, 1 to 1:30 PM.")
        let focus = WeekFacts.answer(.findEvent(name: "focus"), in: Self.week())
        #expect(focus.sentence == "2 events match \u{201C}focus\u{201D}: Monday, 1 to 3 PM and Friday, 9 to 11 AM.")
        #expect(focus.eventIDs == ["focus-mon", "focus-fri"])
        #expect(WeekFacts.answer(.findEvent(name: "check launch"), in: Self.week()).facts.count == 1)
        #expect(WeekFacts.answer(.findEvent(name: "dentist"), in: Self.week()).sentence
            == "No event called \u{201C}dentist\u{201D} this week.")
        #expect(WeekFacts.answer(.findEvent(name: "team sync"), in: Self.week()).sentence
            == "Team sync was Monday, 10 to 10:30 AM.")
    }

    @Test func daySummaryPutsAllDayEventsFirst() {
        let answer = WeekFacts.answer(.daySummary(day: CalendarFixtures.date(17, 9)), in: Self.week())
        #expect(answer.sentence == "Thursday has 2 events: Offsite, all day and Launch check-in at 11 AM.")
        #expect(WeekFacts.answer(.daySummary(day: CalendarFixtures.date(19, 0)), in: Self.week()).sentence
            == "Saturday has nothing booked.")
    }

    @Test func todosDueLeaveOutDoneOnesAndCountTheRest() {
        let answer = WeekFacts.answer(.todosDue, in: Self.week())
        #expect(answer.sentence == "2 to-dos are due this week: Send copy decisions to Priya (today) and "
            + "Review contract section 4 (Friday). 1 is overdue: Renew passport. 1 more has no date.")
        #expect(answer.facts.map(\.line) == ["T1: Send copy decisions to Priya, due today (Tuesday Sep 15)",
                                             "T2: Review contract section 4, due Friday Sep 18",
                                             "T3: Renew passport, overdue since Thursday Sep 10"])
        #expect(WeekFacts.answer(.todosDue, in: Self.week(todos: [])).sentence == "Nothing is due this week.")
    }

    @Test func overlapsFindSharedTime() {
        #expect(WeekFacts.answer(.overlaps, in: Self.week()).sentence == "Nothing overlaps this week.")
        var events = CalendarFixtures.events
        events.append(CalendarEvent(id: "lunch-2", title: "Lunch with Ana", start: CalendarFixtures.date(15, 12, 30),
                                    end: CalendarFixtures.date(15, 13, 30)))
        let answer = WeekFacts.answer(.overlaps, in: Self.week(events: events))
        #expect(answer.sentence == "Lunch and Lunch with Ana overlap today, 12:30 to 1 PM.")
        #expect(answer.eventIDs == ["lunch", "lunch-2"])
    }

    @Test func turnedAwayKindsNeedNoWording() {
        let outside = WeekFacts.answer(.outsideWeek, in: Self.week())
        #expect(outside.sentence == "I can only see Sep 14 to 20 here. Go to that week and ask there.")
        #expect(!outside.wantsWording)
        #expect(!WeekFacts.answer(.other, in: Self.week()).wantsWording)
    }

    @Test func sundayStartWeekAcrossTheClockChange() {
        var calendar = CalendarFixtures.calendar
        calendar.firstWeekday = 1
        // Saturday Oct 31 2026; clocks go back on Sunday Nov 1 in Los Angeles.
        let now = calendar.date(from: DateComponents(year: 2026, month: 10, day: 31, hour: 20))!
        let sunday = calendar.date(from: DateComponents(year: 2026, month: 11, day: 1))!
        let layout = WeekLayout(calendar: calendar)
        let week = WeekSnapshot(interval: layout.interval(from: layout.weekStart(containing: sunday)), now: now,
                                calendar: calendar, events: [], todos: [])
        #expect(week.days.first == sunday)
        #expect(week.days.count == 7)
        let answer = WeekFacts.answer(.freeTime(FreeTimeRequest(minimumMinutes: 60, day: sunday)), in: week)
        #expect(answer.openTimes.count == 1)
        #expect(answer.openTimes.first.map { WeekFacts.minutes($0) } == 8 * 60)
        #expect(answer.facts.first?.line.hasSuffix("9 AM to 5 PM, 8 h") == true)
    }

    @Test func twentyFourHourRegionsWriteTwentyFourHourTimes() {
        var calendar = CalendarFixtures.calendar
        calendar.locale = Locale(identifier: "de_DE")
        let text = WeekText(calendar: calendar, now: CalendarFixtures.now)
        #expect(!text.usesTwelveHours)
        #expect(text.range(CalendarFixtures.date(15, 13), CalendarFixtures.date(15, 17)) == "13:00 to 17:00")
    }

    @Test func termsHoldEveryNumberAndDayTheFactsMention() {
        let answer = WeekFacts.answer(.freeTime(FreeTimeRequest(minimumMinutes: 120)), in: Self.week())
        #expect(answer.terms.numbers.isSuperset(of: [8, 10, 12, 13, 17, 5, 2, 15, 30]))
        #expect(answer.terms.dayWords.isSuperset(of: ["today", "tuesday", "friday"]))
        let found = WeekTerms.found(in: "Friday at 1:30 PM, two blocks; tue too", calendar: Self.calendar)
        #expect(found.numbers == [1, 30, 2])
        #expect(found.dayWords == ["friday", "tuesday"])
    }
}

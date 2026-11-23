import EventKit
import Foundation
@testable import MferenceAppCore
import Testing

/// The arithmetic behind the schedule: open blocks and booked time, worked
/// out in Swift so the model never has to (`CalendarFixtures`, now =
/// Tuesday Sep 15, 8 AM).
@Suite struct WeekFactsTests {
    static func week(now: Date = CalendarFixtures.now, events: [CalendarEvent] = CalendarFixtures.events,
                     todos: [TodoItem] = [], calendar: Calendar = CalendarFixtures.calendar) -> WeekSnapshot {
        let layout = WeekLayout(calendar: calendar)
        return WeekSnapshot(interval: layout.interval(from: layout.weekStart(containing: now)), now: now,
                            calendar: calendar, events: events, todos: todos)
    }

    static func ranges(_ blocks: [DateInterval]) -> [String] {
        let text = WeekText(calendar: CalendarFixtures.calendar, now: CalendarFixtures.now)
        return blocks.map { text.range($0.start, $0.end) }
    }

    @Test func weekLeavesOutOtherWeeksEvents() {
        let week = Self.week()
        #expect(!week.events.contains { $0.id == "next-week" })
        #expect(week.days.count == 7)
        #expect(week.text.weekRange(week.interval) == "Sep 14 to 20")
    }

    @Test func openBlocksRunFromNowInsideWorkHours() {
        let week = Self.week()
        #expect(Self.ranges(WeekFacts.openBlocks(on: CalendarFixtures.date(15, 0), in: week))
            == ["9 to 9:30 AM", "10 AM to 12 PM", "1 to 5 PM"])
        // The person's own all-day event (the Offsite) takes the whole day.
        #expect(WeekFacts.openBlocks(on: CalendarFixtures.date(17, 0), in: week).isEmpty)
        #expect(WeekFacts.openBlocks(on: CalendarFixtures.date(14, 0), in: week).isEmpty)
    }

    @Test func subscribedAllDayEventsMarkADayWithoutFillingIt() {
        let events = WeekContextTests.subscribedOffsite
        let week = Self.week(events: events)
        #expect(Self.ranges(WeekFacts.openBlocks(on: CalendarFixtures.date(17, 0), in: week))
            == ["9 to 11 AM", "12 to 5 PM"])
        #expect(!week.isBlocked(CalendarFixtures.date(17, 0)))
        #expect(Self.week().isBlocked(CalendarFixtures.date(17, 0)))
        #expect(!Self.week().isBlocked(CalendarFixtures.date(18, 0)))
    }

    @Test func onlyCalendarsThePersonAddsToCountAsTheirs() {
        #expect(EventKitService.isUserCalendar(type: .calDAV, allowsContentModifications: true))
        #expect(EventKitService.isUserCalendar(type: .local, allowsContentModifications: true))
        #expect(!EventKitService.isUserCalendar(type: .subscription, allowsContentModifications: false))
        #expect(!EventKitService.isUserCalendar(type: .birthday, allowsContentModifications: false))
        #expect(!EventKitService.isUserCalendar(type: .calDAV, allowsContentModifications: false))
    }

    @Test func openBlocksStartAtTheNextQuarterHourAndSkipShortGaps() {
        let week = Self.week(now: CalendarFixtures.date(15, 10, 17))
        #expect(Self.ranges(WeekFacts.openBlocks(on: CalendarFixtures.date(15, 0), in: week))
            == ["10:30 AM to 12 PM", "1 to 5 PM"])
        let late = Self.week(now: CalendarFixtures.date(15, 16, 40))
        #expect(WeekFacts.openBlocks(on: CalendarFixtures.date(15, 0), in: late).isEmpty)
    }

    @Test func overlappingEventsBlockTimeOnce() {
        var events = CalendarFixtures.events
        events.append(CalendarEvent(id: "lunch-2", title: "Lunch with Ana", start: CalendarFixtures.date(15, 12, 30),
                                    end: CalendarFixtures.date(15, 13, 30)))
        #expect(Self.ranges(WeekFacts.openBlocks(on: CalendarFixtures.date(15, 0), in: Self.week(events: events)))
            == ["9 to 9:30 AM", "10 AM to 12 PM", "1:30 to 5 PM"])
        let tuesday = WeekFacts.loads(in: Self.week(events: events))[1]
        #expect(tuesday.bookedMinutes == 120)
        #expect(tuesday.eventCount == 3)
    }

    @Test func loadsMarkPastDays() {
        let loads = WeekFacts.loads(in: Self.week())
        #expect(loads.map(\.bookedMinutes) == [150, 90, 120, 60, 150, 0, 0])
        #expect(loads.map(\.isPast) == [true, false, false, false, false, false, false])
        #expect(loads.map(\.isBlocked) == [false, false, false, true, false, false, false])
    }

    @Test func aClockChangeDayStillHasEightWorkHours() {
        var calendar = CalendarFixtures.calendar
        calendar.firstWeekday = 1
        // Clocks go back on Sunday Nov 1 2026 in Los Angeles.
        let now = calendar.date(from: DateComponents(year: 2026, month: 10, day: 31, hour: 20))!
        let sunday = calendar.date(from: DateComponents(year: 2026, month: 11, day: 1))!
        let layout = WeekLayout(calendar: calendar)
        let week = WeekSnapshot(interval: layout.interval(from: layout.weekStart(containing: sunday)), now: now,
                                calendar: calendar, events: [], todos: [])
        #expect(week.days.first == sunday)
        let blocks = WeekFacts.openBlocks(on: sunday, in: week)
        #expect(blocks.count == 1)
        #expect(blocks.first.map { WeekFacts.minutes($0) } == 8 * 60)
        #expect(blocks.first.map { calendar.component(.hour, from: $0.start) } == 9)
    }

    @Test func twentyFourHourRegionsWriteTwentyFourHourTimes() {
        var calendar = CalendarFixtures.calendar
        calendar.locale = Locale(identifier: "de_DE")
        let text = WeekText(calendar: calendar, now: CalendarFixtures.now)
        #expect(!text.usesTwelveHours)
        #expect(text.range(CalendarFixtures.date(15, 13), CalendarFixtures.date(15, 17)) == "13:00 to 17:00")
    }

}

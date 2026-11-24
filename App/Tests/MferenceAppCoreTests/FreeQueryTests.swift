import Foundation
@testable import MferenceAppCore
import Testing

/// Free time, answered by Swift with fixed windows (spec "Free-time
/// rules"): reading the model's fields, the windows, the search and the
/// headlines (`CalendarFixtures`: now Tuesday Sep 15, 8 AM; Offsite takes
/// Thursday).
@Suite struct FreeQueryTests {
    static let week = WeekFactsTests.week()

    static func decode(_ json: String, in week: WeekSnapshot = week) throws -> FreeQuery {
        let object = try #require(AssistantAnswerParser.jsonObject(in: json))
        return FreeQuery.decode(object["free_query"], in: week)
    }

    static func run(_ query: FreeQuery, in week: WeekSnapshot = week) -> FreeTimeResult {
        FreeTimeSearch.run(query, in: week)
    }

    // MARK: Reading the fields

    @Test func fieldsAreReadLeniently() throws {
        let query = try Self.decode("""
            {"free_query": {"min_minutes": "90", "days": ["Friday", "tomorrow", "2026-09-19", "2026-10-12"],
             "part": "Morning", "after": "3 PM", "before": "17:00"}}
            """)
        #expect(query.minMinutes == 90)
        #expect(query.days == ["2026-09-16", "2026-09-18", "2026-09-19"])
        #expect(query.outside == ["2026-10-12"])
        #expect(query.part == .morning)
        #expect(query.after == 15 * 60)
        #expect(query.before == 17 * 60)
    }

    @Test func dayWordsResolveInsideTheAskedWeek() throws {
        #expect(try Self.decode(#"{"free_query": {"days": ["today"]}}"#).days == ["2026-09-15"])
        #expect(try Self.decode(#"{"free_query": {"days": ["fri", "Wed"]}}"#).days == ["2026-09-16", "2026-09-18"])
        #expect(try Self.decode(#"{"free_query": {"days": "weekend"}}"#).days == ["2026-09-19", "2026-09-20"])
        // "Monday" is this week's Monday, even though it is over.
        #expect(try Self.decode(#"{"free_query": {"days": ["Monday"]}}"#).days == ["2026-09-14"])
        // Nothing readable: the default days.
        #expect(try Self.decode(#"{"free_query": {"days": ["Funday"], "min_minutes": 0}}"#) == FreeQuery())
        #expect(try Self.decode(#"{"free_query": "tomorrow"}"#) == FreeQuery())
    }

    @Test func timesComeInManyForms() {
        #expect(FreeQuery.clockMinutes("15:00") == 900)
        #expect(FreeQuery.clockMinutes("15") == 900)
        #expect(FreeQuery.clockMinutes("3pm") == 900)
        #expect(FreeQuery.clockMinutes("3:30 p.m.") == 930)
        #expect(FreeQuery.clockMinutes("12 AM") == 0)
        #expect(FreeQuery.clockMinutes("noon") == 720)
        #expect(FreeQuery.clockMinutes("24:00") == 1440)
        #expect(FreeQuery.clockMinutes(NSNumber(value: 18)) == 1080)
        #expect(FreeQuery.clockMinutes("25:00") == nil)
        #expect(FreeQuery.clockMinutes("13 PM") == nil)
        #expect(FreeQuery.clockMinutes("lunch") == nil)
    }

    @Test func tonightIsTheEvening() throws {
        #expect(try Self.decode(#"{"free_query": {"part": "tonight"}}"#).part == .evening)
    }

    @Test func theQueryIsReplayedAsTheJSONTheModelWrites() {
        let query = FreeQuery(minMinutes: 120, days: ["2026-09-18"], part: .morning, after: 15 * 60, before: 24 * 60,
                              outside: ["2026-10-12"])
        #expect(query.json == #"{"free_query": {"min_minutes": 120, "days": ["2026-09-18", "2026-10-12"], "#
            + #""part": "morning", "after": "15:00", "before": "24:00"}}"#)
        #expect(FreeQuery().json == #"{"free_query": {}}"#)
        #expect(FreeQuery(minMinutes: 120).json == #"{"free_query": {"min_minutes": 120}}"#)
    }

    // MARK: Windows

    @Test func partsHaveFixedWindows() {
        #expect(FreeQuery().window() == 540..<1020)
        #expect(FreeQuery(part: .morning).window() == 540..<720)
        #expect(FreeQuery(part: .afternoon).window() == 720..<1020)
        #expect(FreeQuery(part: .evening).window() == 1020..<1260)
        #expect(FreeQuery(part: .night).window() == 1260..<1440)
    }

    @Test func explicitTimesWinAndOneSidedTimesGetAFixedOtherEnd() {
        #expect(FreeQuery(after: 15 * 60).window() == 900..<1020)
        // "Free after 6 PM?" searches 6 to 9 PM.
        #expect(FreeQuery(after: 18 * 60).window() == 1080..<1260)
        #expect(FreeQuery(after: 22 * 60).window() == 1320..<1440)
        // "Free before 9 AM?" searches 5 to 9 AM.
        #expect(FreeQuery(before: 9 * 60).window() == 300..<540)
        #expect(FreeQuery(before: 12 * 60).window() == 540..<720)
        #expect(FreeQuery(after: 14 * 60, before: 16 * 60).window() == 840..<960)
        // after/before replace the part's window.
        #expect(FreeQuery(part: .morning, after: 18 * 60).window() == 1080..<1260)
        // before <= after is refused.
        #expect(FreeQuery(after: 16 * 60, before: 14 * 60).window() == nil)
        #expect(FreeQuery(after: 14 * 60, before: 14 * 60).window() == nil)
    }

    // MARK: The search and its headline

    @Test func twoHoursThisWeekSearchesWeekdaysFromToday() {
        let result = Self.run(FreeQuery(minMinutes: 120))
        #expect(result.days.map { WeekContext.isoDate($0, in: Self.week.calendar) }
            == ["2026-09-15", "2026-09-16", "2026-09-17", "2026-09-18"])
        #expect(WeekFactsTests.ranges(result.openings)
            == ["10 AM to 12 PM", "1 to 5 PM", "11 AM to 1 PM", "2 to 5 PM", "11 AM to 1 PM", "1:30 to 5 PM"])
        #expect(result.headline == "6 openings of 2 h or more: Tue, Wed, Fri.")
    }

    /// Q19: blocks are clipped to the morning, so 90 minutes tomorrow
    /// morning is a clear No.
    @Test func ninetyMinutesTomorrowMorningIsNo() {
        let result = Self.run(FreeQuery(minMinutes: 90, days: ["2026-09-16"], part: .morning))
        #expect(result.openings.isEmpty)
        #expect(result.headline == "No 90-minute opening tomorrow morning. The longest is 1 h, 9 to 10 AM.")
    }

    @Test func weekendsAreSearchedOnlyWhenNamed() {
        #expect(!Self.run(FreeQuery()).days.contains { Self.week.calendar.isDateInWeekend($0) })
        let saturday = Self.run(FreeQuery(days: ["2026-09-19"]))
        #expect(saturday.headline == "1 opening on Saturday: 9 AM to 5 PM.")
        let weekend = Self.run(FreeQuery(minMinutes: 60, days: ["2026-09-19", "2026-09-20"]))
        #expect(weekend.headline == "2 openings of 1 h or more: Sat, Sun.")
    }

    @Test func todayStartsAtNowRoundedUpAndADayOverHasNothing() {
        let late = WeekFactsTests.week(now: CalendarFixtures.date(15, 10, 17))
        let today = Self.run(FreeQuery(days: ["2026-09-15"], part: .morning), in: late)
        #expect(WeekFactsTests.ranges(today.openings) == ["10:30 AM to 12 PM"])
        #expect(today.headline == "1 opening this morning: 10:30 AM to 12 PM.")
        #expect(Self.run(FreeQuery(days: ["2026-09-14"])).headline == "Monday is over.")
    }

    @Test func headlinesNameTheWindowTheySearched() {
        #expect(Self.run(FreeQuery(days: ["2026-09-15"], after: 15 * 60)).headline
            == "1 opening today after 3 PM: 3 to 5 PM.")
        #expect(Self.run(FreeQuery(days: ["2026-09-18"], after: 14 * 60, before: 16 * 60)).headline
            == "1 opening on Friday between 2 and 4 PM: 2 to 4 PM.")
        #expect(Self.run(FreeQuery(days: ["2026-09-16"], before: 12 * 60)).headline
            == "2 openings tomorrow before noon: 9 to 10 AM and 11 AM to 12 PM.")
        #expect(Self.run(FreeQuery(days: ["2026-09-15"], part: .evening)).headline
            == "1 opening this evening: 5 to 9 PM.")
        #expect(Self.run(FreeQuery(days: ["2026-09-15"], part: .night)).headline
            == "1 opening tonight: 9 PM to 12 AM.")
        #expect(Self.run(FreeQuery(minMinutes: 180, part: .morning)).headline
            == "No 3-hour opening in the morning this week. The longest is 2 h, Tue 10 AM to 12 PM.")
        #expect(Self.run(FreeQuery(days: ["2026-09-16", "2026-09-18"], part: .afternoon)).headline
            == "4 openings in the afternoon: Wed, Fri.")
    }

    @Test func aTakenDayHasNoOpenTime() {
        #expect(Self.run(FreeQuery(days: ["2026-09-17"])).headline == "No open time on Thursday.")
    }

    @Test func aDayOutsideTheWeekPointsToItsWeek() {
        let outside = Self.run(FreeQuery(minMinutes: 60, outside: ["2026-10-12"]))
        #expect(outside.openings.isEmpty)
        #expect(outside.headline == "I'm looking at Sep 14 to 20. Go to that week to check Oct 12.")
        let both = Self.run(FreeQuery(days: ["2026-09-19"], outside: ["2026-09-22"]))
        #expect(both.headline == "1 opening on Saturday: 9 AM to 5 PM. I'm looking at Sep 14 to 20. "
            + "Go to that week to check Sep 22.")
    }

    @Test func anEmptyWindowIsSaidPlainly() {
        #expect(Self.run(FreeQuery(after: 16 * 60, before: 14 * 60)).headline
            == "There is no time after 4 PM and before 2 PM.")
    }

    @Test func withNoWeekdaysLeftTheHeadlineSaysToNameTheWeekend() {
        let saturday = WeekFactsTests.week(now: CalendarFixtures.date(19, 10))
        #expect(Self.run(FreeQuery(), in: saturday).headline
            == "No weekdays are left this week. Name Saturday or Sunday to look there.")
    }
}

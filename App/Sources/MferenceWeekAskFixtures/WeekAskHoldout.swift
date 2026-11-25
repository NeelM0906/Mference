import Foundation
import MferenceAppCore

/// A holdout set: questions and a week the prompt was never tuned on, asked
/// once as a check that the v1 results carry over (`MferenceSmoke
/// --week-ask --holdout`). The key was written before the first run.
extension WeekAskCheckpoint {
    /// Monday Mar 1 to Sunday Mar 7 2027, today Wednesday 1:40 PM (the
    /// morning is over). Mom's visit runs Thursday to Saturday, all day and
    /// shown as Free; a retreat runs from Thursday 3 PM to Friday 1 PM, and
    /// the Quarterly review fills Friday afternoon, so Friday has no open
    /// time without being an all-day day. One overlap on Thursday.
    public static var holdoutWeek: WeekSnapshot {
        let calendar = calendar(firstWeekday: 2)
        func at(_ day: Int, _ hour: Int, _ minute: Int = 0) -> Date {
            calendar.date(from: DateComponents(year: 2027, month: 3, day: day, hour: hour, minute: minute)) ?? .now
        }
        func event(_ id: String, _ title: String, _ day: Int, _ from: (Int, Int), _ to: (Int, Int)) -> CalendarEvent {
            CalendarEvent(id: id, title: title, start: at(day, from.0, from.1), end: at(day, to.0, to.1))
        }
        return week(now: at(3, 13, 40), calendar: calendar, events: [
            event("poker", "Planning poker", 1, (10, 0), (11, 0)),
            event("priya", "Lunch with Priya", 1, (12, 0), (13, 0)),
            event("gym-1", "Gym", 1, (18, 0), (19, 0)),
            event("doctor", "Doctor checkup", 2, (8, 30), (9, 30)),
            event("budget", "Budget sync", 2, (14, 0), (15, 0)),
            event("standup-3", "Standup", 3, (9, 0), (9, 15)),
            event("vendor", "Vendor contract review", 3, (10, 0), (11, 30)),
            event("pairing", "Pairing with Lee", 3, (14, 0), (15, 0)),
            event("piano", "Piano lesson", 3, (18, 0), (18, 45)),
            event("standup-4", "Standup", 4, (9, 0), (9, 15)),
            event("workshop", "Client workshop", 4, (10, 0), (12, 0)),
            event("jordan", "1:1 with Jordan", 4, (11, 30), (12, 15)),
            CalendarEvent(id: "retreat", title: "Team retreat", start: at(4, 15), end: at(5, 13)),
            event("quarterly", "Quarterly review", 5, (13, 0), (17, 0)),
            CalendarEvent(id: "mom", title: "Mom visiting", start: at(4, 0), end: at(7, 0), isAllDay: true,
                          isFree: true),
            event("market", "Farmers market", 6, (9, 0), (10, 30)),
            event("movie", "Movie night", 6, (19, 0), (22, 0)),
        ], todos: [
            TodoItem(title: "Buy train tickets", due: at(1, 0)),
            TodoItem(title: "Return library books", due: at(3, 0)),
            TodoItem(title: "File taxes", due: at(5, 0)),
            TodoItem(title: "Water the plants", isDone: true, due: at(2, 0)),
            TodoItem(title: "Plan the garden"),
        ])
    }

    /// 18 typed questions and one chip, in new phrasings.
    public static let holdoutQuestions: [Question] = [
        q("X01", "holdout", "Do I have a spare 45 minutes this afternoon?",
          .free(FreeQuery(minMinutes: 45, days: days("2027-03-03"), part: .afternoon))),
        q("X02", "holdout", "Is Friday completely booked?",
          .free(FreeQuery(days: days("2027-03-05")), alsoAnswer: true,
                notes: "Yes: Team retreat until 1 PM, then Quarterly review 1 to 5 PM; no open time.")),
        q("X03", "holdout", "When does the retreat end?",
          .answer("Friday 1 PM (Team retreat, from Thursday 3 PM).", single: true)),
        q("X04", "holdout", "How many events do I have tomorrow?",
          .answer("Thursday: Standup, Client workshop, 1:1 with Jordan, Team retreat (from 3 PM), plus Mom visiting "
                  + "(all day, shown as free): 5 listed, or 4 leaving out the free banner. Either is a 2.",
                  single: true)),
        q("X05", "holdout", "What did I have on Monday?",
          .answer("Planning poker 10 to 11 AM, Lunch with Priya 12 to 1 PM, Gym 6 to 7 PM.", gate: .mustCite)),
        q("X06", "holdout", "What time was the doctor?",
          .answer("Tuesday 8:30 to 9:30 AM (Doctor checkup).", single: true, gate: .mustCite)),
        q("X07", "holdout", "Give me a rundown of the rest of my week",
          .answer("From today: Pairing with Lee, Piano lesson; Thursday Standup, Client workshop, 1:1 with Jordan, "
                  + "Team retreat; Friday retreat and Quarterly review; Saturday Farmers market, Movie night; Mom "
                  + "visiting Thursday to Saturday. Nothing from Monday or Tuesday.", gate: .mustNotCite)),
        q("X08", "holdout", "Any gap for a 30 min call on Thursday before noon?",
          .free(FreeQuery(minMinutes: 30, days: days("2027-03-04"), part: .morning),
                FreeQuery(minMinutes: 30, days: days("2027-03-04"), before: 12 * 60))),
        q("X09", "holdout", "Am I free Saturday evening?",
          .free(FreeQuery(days: days("2027-03-06"), part: .evening))),
        q("X10", "holdout", "Could I squeeze in 3 hours on Sunday?",
          .free(FreeQuery(minMinutes: 180, days: days("2027-03-07")))),
        q("X11", "holdout", "Is my mom visiting this week?",
          .answer("Yes: Mom visiting, Thursday to Saturday (all day, shown as free).", single: true)),
        q("X12", "holdout", "What's overdue?",
          .answer("Buy train tickets (due Monday).", single: true)),
        q("X13", "holdout", "When's my piano lesson?",
          .answer("Today, 6 to 6:45 PM.", single: true)),
        q("X14", "holdout", "Any time Friday afternoon?",
          .free(FreeQuery(days: days("2027-03-05"), part: .afternoon))),
        q("X15", "holdout", "What am I doing on March 10?",
          .answer("March 10 is outside this week; say it isn't in the calendar shown (no invented events).",
                  single: true)),
        q("X16", "holdout", "Do any of my meetings clash this week?",
          .answer("Client workshop and 1:1 with Jordan overlap Thursday, 11:30 AM to 12 PM.", single: true)),
        Question(id: "X17", week: "holdout", text: "When am I free for an hour?", followUp: false,
                 chip: .freeFor(minutes: 60), expected: nil),
        q("X18", "holdout", "just mornings", followUp: true,
          .free(FreeQuery(minMinutes: 60, part: .morning))),
        q("X19", "holdout", "and what's due Friday?", followUp: true,
          .answer("File taxes, due Friday.", single: true)),
    ]
}

import Foundation

/// A suggestion chip (spec "Chips"). Every chip is answered in Swift, with
/// no model call: instant, exact, and working with no model installed. Each
/// stores the record a typed question would, so a typed follow-up builds on
/// it. A chip is known by what was clicked, never by matching typed text.
public enum WeekChip: Sendable, Hashable {
    /// "When am I free for 2 hours?"
    case freeFor(minutes: Int)
    /// "Am I free today?" / "Am I free on Friday?"
    case freeOn(Date)
    /// "What's on today?" / "What's on Friday?"
    case whatsOn(Date)
    /// "What's tomorrow like?"
    case tomorrow
    /// "What's my busiest day?"
    case busiest
    /// "What's due this week?"
    case due
    /// "Anything overlapping?"
    case overlaps

    /// The chip's words, which are also the question shown and replayed.
    public func title(_ text: WeekText) -> String {
        switch self {
        case .freeFor(let minutes):
            "When am I free for \(minutes % 60 == 0 ? WeekText.count(minutes / 60, "hour") : "\(minutes) minutes")?"
        case .freeOn(let day): "Am I free \(Self.when(day, text))?"
        case .whatsOn(let day): "What's on \(text.dayWord(day))?"
        case .tomorrow: "What's tomorrow like?"
        case .busiest: "What's my busiest day?"
        case .due: "What's due this week?"
        case .overlaps: "Anything overlapping?"
        }
    }

    /// "today", "tomorrow", "on Friday".
    static func when(_ day: Date, _ text: WeekText) -> String {
        let word = text.dayWord(day)
        return word == "today" || word == "tomorrow" ? word : "on \(word)"
    }

    /// The chips while nothing is asked: about the selected day in Day view,
    /// about the week in Week and Month.
    public static func suggestions(dayView: Bool, selected: Date, text: WeekText) -> [WeekChip] {
        guard dayView else { return [.freeFor(minutes: 120), .busiest, .due, .overlaps] }
        if text.dayWord(selected) == "today" {
            return [.whatsOn(selected), .freeOn(selected), .tomorrow, .due, .busiest]
        }
        return [.whatsOn(selected), .freeOn(selected), .due, .busiest]
    }
}

extension WeekAnswerer {
    /// A chip's answer, worked out from `context` alone.
    public static func answer(_ chip: WeekChip, context: WeekContext) -> WeekAnswer {
        let week = context.week
        let question = chip.title(week.text)
        switch chip {
        case .freeFor(let minutes):
            return freeTime(FreeQuery(minMinutes: minutes), question: question, context: context, askedBy: nil)
        case .freeOn(let day):
            return freeTime(FreeQuery(days: [WeekContext.isoDate(day, in: week.calendar)]), question: question,
                            context: context, askedBy: nil)
        case .whatsOn(let day):
            return dayAnswer(day, question: question, context: context)
        case .tomorrow:
            let tomorrow = week.calendar.date(byAdding: .day, value: 1, to: week.today) ?? week.today
            return dayAnswer(tomorrow, question: question, context: context)
        case .busiest:
            return busiestAnswer(question: question, context: context)
        case .due:
            return dueAnswer(question: question, context: context)
        case .overlaps:
            return overlapAnswer(question: question, context: context)
        }
    }

    private static func lookup(_ question: String, _ sentence: String, refs: [WeekFact],
                               extra: [WeekFact] = []) -> WeekAnswer {
        WeekAnswer(question: question, sentence: sentence, rows: refs + extra, source: .calendar(askedBy: nil),
                   record: .lookup(answer: sentence, refs: refs))
    }

    /// A day's events, booked time and open time: "Today: 3 events, 2 h 30
    /// min booked, 4 h 30 min open."
    static func dayAnswer(_ date: Date, question: String, context: WeekContext) -> WeekAnswer {
        let text = context.week.text
        guard let day = context.day(date) else {
            let note = "I'm looking at \(text.weekRange(context.week.interval)). Go to that week to check "
                + "\(text.dayWord(date) == "tomorrow" ? "tomorrow" : text.monthDay(date))."
            return lookup(question, note, refs: [])
        }
        let isPast = day.date < context.week.today
        var parts: [String] = []
        let allDay = day.entries.compactMap(\.event).filter { $0.isAllDay && !$0.isFree && $0.isUserCalendar }
        if day.entries.isEmpty {
            parts.append(isPast ? "nothing was booked" : "nothing booked")
        } else {
            parts.append(WeekText.count(day.entries.count, "event"))
        }
        if let taken = allDay.first, !isPast {
            parts.append("busy all day with \(WeekContext.clean(taken.title))")
        } else if day.load.bookedMinutes > 0 {
            parts.append("\(text.duration(minutes: day.load.bookedMinutes)) booked")
        }
        if !isPast, allDay.isEmpty {
            parts.append(day.open.isEmpty ? "no open time left" : "\(text.duration(minutes: day.openMinutes)) open")
        }
        let sentence = "\(text.dayWordCapitalized(day.date)): " + parts.joined(separator: ", ") + "."
        return lookup(question, sentence, refs: day.events, extra: rows(day.open, in: context))
    }

    /// The person's own all-day days first, else the most booked time.
    static func busiestAnswer(question: String, context: WeekContext) -> WeekAnswer {
        let text = context.week.text
        guard let busiest = context.busiest else {
            return lookup(question, context.days.isEmpty ? "This week is over."
                              : "Nothing is booked for the rest of this week.", refs: [])
        }
        let names = busiest.days.map { text.dayWordCapitalized($0.date) }
        let events = busiest.days.flatMap(\.events)
        let sentence: String
        if busiest.allDay {
            let titles = busiest.days.compactMap { day in
                day.entries.compactMap(\.event).first { $0.isAllDay && !$0.isFree && $0.isUserCalendar }
                    .map { WeekContext.clean($0.title) }
            }
            sentence = names.count == 1
                ? "\(names[0]) is your busiest day left: \(titles.first ?? "an all-day event") takes the whole day."
                : "\(WeekText.list(names)) are your busiest days left: each is taken all day."
        } else {
            let booked = text.duration(minutes: busiest.days[0].load.bookedMinutes)
            sentence = names.count == 1 ? "\(names[0]) is your busiest day left, with \(booked) booked."
                : "\(WeekText.list(names)) are your busiest days left, with \(booked) booked each."
        }
        return lookup(question, sentence, refs: events)
    }

    /// Due and overdue to-dos.
    static func dueAnswer(question: String, context: WeekContext) -> WeekAnswer {
        let text = context.week.text
        let today = context.week.today
        var due: [String] = [], overdue: [String] = []
        for fact in context.todos {
            guard case .todo(let todo) = fact.kind, let date = todo.due else { continue }
            let title = WeekContext.clean(todo.title)
            if date < today { overdue.append(title) } else { due.append("\(title) (\(text.dayWord(date)))") }
        }
        var sentences: [String] = []
        if due.isEmpty {
            sentences.append("Nothing is due this week.")
        } else {
            sentences.append("\(due.count == 1 ? "1 to-do is" : "\(due.count) to-dos are") due this week: "
                + WeekText.list(due) + ".")
        }
        if !overdue.isEmpty {
            sentences.append("Overdue: " + WeekText.list(overdue) + ".")
        }
        return lookup(question, sentences.joined(separator: " "), refs: context.todos)
    }

    /// Overlapping timed events on the listed days.
    static func overlapAnswer(question: String, context: WeekContext) -> WeekAnswer {
        let text = context.week.text
        guard !context.overlaps.isEmpty else {
            return lookup(question, "Nothing overlaps for the rest of this week.", refs: [])
        }
        let lines = context.overlaps.map { overlap -> String in
            let titles = [overlap.first, overlap.second].map { fact -> String in
                if case .event(let event) = fact.kind { WeekContext.clean(event.title) } else { fact.id }
            }
            return "\(titles[0]) and \(titles[1]) overlap \(text.dayWord(overlap.interval.start)), "
                + "\(text.range(overlap.interval.start, overlap.interval.end))"
        }
        var refs: [WeekFact] = []
        for overlap in context.overlaps {
            for fact in [overlap.first, overlap.second] where !refs.contains(fact) { refs.append(fact) }
        }
        return lookup(question, lines.joined(separator: ". ") + ".", refs: refs)
    }
}

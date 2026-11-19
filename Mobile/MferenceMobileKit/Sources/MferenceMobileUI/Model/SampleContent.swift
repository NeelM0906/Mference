import Foundation

/// The sample conversations, events, to-dos and scan used by this first
/// slice, matching the Mac mockups' examples.
enum SampleContent {
    static let deskTable = AnswerTable(
        headers: ["Desk", "Price", "Warranty"],
        rows: [["Stride E2", "$589", "15 yr frame, 5 yr motor [1]"],
               ["Oakline Duo", "$499", "7 yr frame and motor [2]"],
               ["Fernway Lift", "$549", "10 yr frame, 3 yr motor [3]"]])

    static func conversations() -> [Conversation] {
        [
            Conversation(
                title: "Standing desk warranties", group: .today,
                messages: [
                    ChatMessage(role: .user,
                                text: "Which standing desk under $600 has the best warranty right now? Check current prices."),
                    ChatMessage(role: .assistant,
                                text: "Three desks under $600 have warranties worth comparing. Prices are as of this morning.",
                                webSteps: "Searched via paired Mac · 3 searches · 7 pages",
                                table: deskTable,
                                closing: "For the longest cover, the Stride E2. For the best price, the Oakline Duo, on sale through Sunday [2].",
                                sources: ["stride-desks.example", "oakline.example/sale", "deskreviews.example"],
                                actions: ["Track the price", "Buy one, with my approval"],
                                engineLabel: PhoneModel.quick.engineLabel),
                ],
                updatedLabel: "9:41"),
            Conversation(
                title: "Board deck revenue", group: .today,
                messages: [
                    ChatMessage(role: .user,
                                text: "Go through the 40 board PDFs and pull every revenue figure into one table."),
                    ChatMessage(role: .assistant, text: "",
                                handoff: HandoffOffer(reason: "40 long documents is more than this iPhone can read well. Your Mac can do it with Deep and send the table back here.",
                                                      state: .sentToMac)),
                ],
                isRunningOnMac: true, updatedLabel: "9:12"),
            Conversation(
                title: "Lisbon trip, 4 days", group: .today,
                messages: [
                    ChatMessage(role: .user, text: "Plan four days in Lisbon in October, not too rushed."),
                    ChatMessage(role: .assistant,
                                text: "Day one: Alfama in the morning, then the castle before the crowds. Day two: Belém and the riverside. Day three: a train to Sintra. Day four: Príncipe Real and a slow lunch.",
                                engineLabel: PhoneModel.quick.engineLabel),
                ],
                updatedLabel: "8:02"),
            Conversation(title: "Receipt to spreadsheet row", group: .skills),
            Conversation(title: "Weekly review", group: .skills),
            Conversation(title: "Lab results questions", group: .privateChats),
        ]
    }

    struct Planner {
        var events: [PhoneEvent]
        var todos: [PhoneTodo]
        var suggestion: CalendarSuggestion?
    }

    static func planner(today now: Date, calendar: Calendar) -> Planner {
        let today = calendar.startOfDay(for: now)
        func at(_ dayOffset: Int, _ hour: Int, _ minute: Int = 0) -> Date {
            let day = calendar.date(byAdding: .day, value: dayOffset, to: today) ?? today
            return calendar.date(bySettingHour: hour, minute: minute, second: 0, of: day) ?? day
        }
        let review = PhoneEvent(title: "Design review", start: at(1, 10), end: at(1, 10, 45), tint: .blue)
        let events = [
            PhoneEvent(title: "Coffee with Tom", start: at(0, 9, 30), end: at(0, 10), tint: .orange),
            PhoneEvent(title: "Focus time", start: at(0, 10), end: at(0, 11), tint: .grey),
            PhoneEvent(title: "Lunch", start: at(0, 12), end: at(0, 13), tint: .green),
            PhoneEvent(title: "Weekly planning", start: at(0, 13), end: at(0, 14), tint: .blue),
            review,
            PhoneEvent(title: "Launch check-in", start: at(2, 11), end: at(2, 12), tint: .purple),
            PhoneEvent(title: "Team sync", start: at(-1, 10), end: at(-1, 10, 30), tint: .blue),
            PhoneEvent(title: "Contract call", start: at(3, 13), end: at(3, 13, 30), tint: .orange),
        ]
        let tomorrow = calendar.date(byAdding: .day, value: 1, to: today) ?? today
        let weekday = tomorrow.formatted(.dateTime.weekday(.abbreviated))
        var moved = review
        moved.start = at(1, 11)
        moved.end = at(1, 11, 45)
        let suggestion = CalendarSuggestion(
            question: "Move the design review to \(weekday) 11:00?",
            detail: "From Priya's email this morning. Priya gets an update.",
            event: moved)
        let todos = [
            PhoneTodo(title: "Send copy decisions to Priya", detail: "Mail · due \(weekday)"),
            PhoneTodo(title: "Review contract section 4", detail: "Mail · due Fri"),
            PhoneTodo(title: "Pick a dentist appointment", detail: "Mail · Northside Dental"),
            PhoneTodo(title: "Share the launch checklist", detail: "Meeting · Weekly product sync"),
            PhoneTodo(title: "Expense the team lunch", detail: "Scan", isDone: true),
        ]
        return Planner(events: events, todos: todos, suggestion: suggestion)
    }

    static func scanFields(today: Date) -> [ScanField] {
        [
            ScanField(id: "date", label: "Date", value: today.formatted(date: .abbreviated, time: .omitted), isSure: true),
            ScanField(id: "total", label: "Total", value: "$105.47", isSure: true),
            ScanField(id: "tip", label: "Tip", value: "$16.40", isSure: true),
            ScanField(id: "category", label: "Category", value: "Team meals", isSure: false),
        ]
    }
}

/// Canned replies for the sample build, chosen by keyword. They stand in for
/// the on-device engine until it is wired in; the labels name the model the
/// person picked.
struct ScriptedReplies {
    func reply(to request: String, webSearch: Bool, model: PhoneModel) -> ChatMessage {
        let lowered = request.lowercased()
        let label = model.engineLabel
        if lowered.contains("revenue") || lowered.contains("pdf") {
            return ChatMessage(
                role: .assistant,
                text: "Done on your Mac. Revenue rose in every quarter but one; the table has all 40 decks.",
                table: AnswerTable(headers: ["Quarter", "Revenue", "Change"],
                                   rows: [["Q1", "$4.2M", "+6%"], ["Q2", "$4.6M", "+9%"],
                                          ["Q3", "$4.4M", "−4%"], ["Q4", "$5.1M", "+16%"]]),
                closing: "Q3 dipped after the pricing change; the Q3 deck explains it on page 7.",
                actions: ["Export as CSV"],
                engineLabel: label)
        }
        if lowered.contains("desk") || lowered.contains("warranty") {
            return ChatMessage(
                role: .assistant,
                text: "Three desks under $600 have warranties worth comparing. Prices are as of this morning.",
                webSteps: webSearch ? "Searched via paired Mac · 3 searches · 7 pages" : nil,
                table: SampleContent.deskTable,
                closing: "For the longest cover, the Stride E2. For the best price, the Oakline Duo [2].",
                sources: webSearch ? ["stride-desks.example", "oakline.example/sale", "deskreviews.example"] : [],
                engineLabel: label)
        }
        if lowered.contains("free") || lowered.contains("week") || lowered.contains("calendar") {
            return ChatMessage(
                role: .assistant,
                text: "You're free tomorrow from 2 to 5 PM and Friday morning until 1. Tomorrow afternoon is the longer stretch.",
                actions: ["Hold tomorrow 2 to 4 PM"],
                engineLabel: label)
        }
        if lowered.contains("priya") || lowered.contains("email") {
            return ChatMessage(
                role: .assistant,
                text: "She wants to move the design review to 11:00 tomorrow and have the copy decisions settled first.",
                actions: ["Move the review", "Draft a reply"],
                engineLabel: label)
        }
        if webSearch {
            return ChatMessage(
                role: .assistant,
                text: "Here's what I found. The three most recent sources agree on the main points [1], and the oldest one is out of date [3].",
                webSteps: "Searched via paired Mac · 2 searches · 5 pages",
                sources: ["example.org/guide", "news.example/today", "archive.example/2024"],
                engineLabel: label)
        }
        return ChatMessage(
            role: .assistant,
            text: "Sample reply. In this build answers come from a scripted stand-in; the on-device model replaces it when the engine is wired in.",
            engineLabel: label)
    }
}

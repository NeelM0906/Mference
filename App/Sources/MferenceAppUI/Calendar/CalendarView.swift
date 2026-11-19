import AppKit
import EventKit
import MferenceAppCore
import MferenceDesign
import SwiftUI

/// Calendar and to-dos (mockup 04): the work week from Apple Calendar,
/// proposals drawn dashed with one-click Add, and the to-do panel.
public struct CalendarView: View {
    @Environment(CalendarWeekModel.self) private var environmentModel: CalendarWeekModel?
    private let injected: CalendarWeekModel?
    private let loadsOnAppear: Bool

    /// The shell's slot: the model comes from the environment.
    public init() {
        injected = nil
        loadsOnAppear = true
    }

    /// Snapshots and previews; `loadsOnAppear` false keeps fixture state.
    public init(model: CalendarWeekModel, loadsOnAppear: Bool = true) {
        injected = model
        self.loadsOnAppear = loadsOnAppear
    }

    public var body: some View {
        if let model = injected ?? environmentModel {
            CalendarContent(model: model)
                .task { if loadsOnAppear { await model.load() } }
        } else {
            AppleEmptyState(systemImage: "calendar", title: "Calendar",
                            message: "Calendar is starting. Try again in a moment.")
        }
    }
}

struct CalendarContent: View {
    @Bindable var model: CalendarWeekModel

    var body: some View {
        HStack(spacing: 0) {
            VStack(alignment: .leading, spacing: MFSpacing.m) {
                header
                switch model.access {
                case .granted:
                    WeekGrid(model: model)
                    if let proposal = model.proposalsThisWeek.first {
                        ProposalCard(proposal: proposal, model: model)
                    }
                case .notAsked:
                    AppleEmptyState(systemImage: "calendar", title: "See your week here",
                                    message: AppPermission.calendar.reason,
                                    actionTitle: "Show my calendar") { Task { await model.requestAccess() } }
                case .blocked:
                    AppleEmptyState(systemImage: "lock", title: "Calendar is turned off for Mference",
                                    message: "Turn on Mference under System Settings > Privacy & Security > Calendars.",
                                    actionTitle: "Open System Settings") { Task { await model.openCalendarSettings() } }
                case .promptNotShown:
                    AppleEmptyState(systemImage: "exclamationmark.lock", title: "macOS didn't show the Calendar prompt",
                                    message: "Turn on Mference under Privacy & Security > Calendars, then come back here.",
                                    actionTitle: "Open System Settings > Calendars") {
                        Task { await model.openCalendarSettings() }
                    }
                }
                if let message = model.errorMessage {
                    Text(message)
                        .font(MFFont.callout)
                        .foregroundStyle(MFColor.danger)
                }
            }
            .padding(.horizontal, MFSpacing.xl)
            .padding(.vertical, MFSpacing.l)
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
            Rectangle().fill(MFColor.hairline).frame(width: 1)
            TodoList(model: model)
                .frame(width: 315)
                .background(MFColor.canvas)
        }
        .background(MFColor.surface)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        // Back from System Settings: look again while access is missing.
        .onReceive(NotificationCenter.default.publisher(for: NSApplication.didBecomeActiveNotification)) { _ in
            Task { await model.recheckAccess() }
        }
        // Access granted in the prompt or System Settings, or events edited
        // in Calendar: show them now, not after a relaunch.
        .onReceive(NotificationCenter.default.publisher(for: .EKEventStoreChanged)) { _ in
            Task { await model.calendarChanged() }
        }
        .shellToolbar(id: "calendar.new-event") { NewEventToolbarButton(model: model) }
    }

    private var header: some View {
        HStack(spacing: MFSpacing.m) {
            Text(model.weekTitle)
                .font(MFFont.title)
                .foregroundStyle(MFColor.ink)
                .accessibilityAddTraits(.isHeader)
            Button("Today") { Task { await model.showToday() } }
                .buttonStyle(.appleSecondary)
                .keyboardShortcut("t", modifiers: .command)
            HStack(spacing: MFSpacing.xs) {
                Button { Task { await model.showWeek(offset: -1) } } label: { Image(systemName: "chevron.left") }
                    .accessibilityLabel("Previous week")
                    .keyboardShortcut(.leftArrow, modifiers: .command)
                Button { Task { await model.showWeek(offset: 1) } } label: { Image(systemName: "chevron.right") }
                    .accessibilityLabel("Next week")
                    .keyboardShortcut(.rightArrow, modifiers: .command)
            }
            .buttonStyle(.appleQuiet)
            Spacer()
            HStack(spacing: 0) {
                Text("Week")
                    .font(MFFont.bodyEmphasis)
                    .foregroundStyle(MFColor.onInk)
                    .padding(.horizontal, MFSpacing.l)
                    .padding(.vertical, MFSpacing.s)
                    .background(Capsule().fill(MFColor.ink))
            }
            .padding(MFSpacing.xs)
            .background(Capsule().fill(MFColor.fill))
            .accessibilityLabel("Week view")
        }
    }
}

/// The hour grid with the week's blocks.
struct WeekGrid: View {
    var model: CalendarWeekModel
    /// Fixed hour rows (mockup 04); the grid scrolls through the day.
    private let hourHeight: CGFloat = 72
    private let gutter: CGFloat = 70

    var body: some View {
        let layout = model.layout
        let hours = Array(layout.firstHour..<layout.lastHour)
        VStack(spacing: 0) {
            HStack(spacing: 0) {
                Color.clear.frame(width: gutter, height: 1)
                ForEach(Array(model.days.enumerated()), id: \.offset) { _, day in
                    dayLabel(day).frame(maxWidth: .infinity)
                }
            }
            .padding(.bottom, MFSpacing.s)
            let outside = model.outsideGrid
            if !outside.isEmpty {
                HStack(alignment: .top, spacing: 0) {
                    Text("All day")
                        .font(MFFont.caption)
                        .foregroundStyle(MFColor.tertiary)
                        .frame(width: gutter, alignment: .leading)
                    ForEach(model.days.indices, id: \.self) { day in
                        VStack(alignment: .leading, spacing: MFSpacing.xxs) {
                            ForEach(outside[day] ?? []) { event in
                                Text(event.isAllDay ? event.title : "\(EventBlockView.time(event.start)) \(event.title)")
                                    .font(MFFont.caption)
                                    .foregroundStyle(MFColor.ink)
                                    .lineLimit(1)
                                    .padding(.horizontal, MFSpacing.s)
                                    .padding(.vertical, MFSpacing.xxs)
                                    .frame(maxWidth: .infinity, alignment: .leading)
                                    .background(RoundedRectangle(cornerRadius: MFRadius.xs).fill(MFColor.fill))
                                    .accessibilityLabel(event.isAllDay ? "\(event.title), all day"
                                                        : "\(event.title), \(EventBlockView.range(event.start, event.end))")
                            }
                        }
                        .padding(.horizontal, MFSpacing.xs)
                        .frame(maxWidth: .infinity, alignment: .topLeading)
                    }
                }
                .padding(.bottom, MFSpacing.s)
            }
            Rectangle().fill(MFColor.hairline).frame(height: 1)
            GeometryReader { outer in
                MFScroll {
                ZStack(alignment: .topLeading) {
                    VStack(spacing: 0) {
                        ForEach(hours, id: \.self) { hour in
                            HStack(alignment: .top, spacing: 0) {
                                Text(hourLabel(hour))
                                    .font(MFFont.callout)
                                    .foregroundStyle(MFColor.tertiary)
                                    .frame(width: gutter, alignment: .leading)
                                    .padding(.top, MFSpacing.xs)
                                Spacer()
                            }
                            .frame(height: hourHeight, alignment: .top)
                            .overlay(alignment: .bottom) { Rectangle().fill(MFColor.hairlineFaint).frame(height: 1) }
                        }
                    }
                    let columnWidth = (outer.size.width - gutter) / CGFloat(max(model.days.count, 1))
                    ForEach(model.blocks) { block in
                        let laneWidth = (columnWidth - MFSpacing.s) / CGFloat(block.lanes)
                        EventBlockButton(block: block, model: model)
                            .frame(width: laneWidth - MFSpacing.xxs,
                                   height: CGFloat(block.height) * hourHeight - MFSpacing.xs)
                            .offset(x: gutter + CGFloat(block.day) * columnWidth + MFSpacing.xs
                                        + CGFloat(block.lane) * laneWidth,
                                    y: CGFloat(block.top) * hourHeight + MFSpacing.xxs)
                    }
                }
                .frame(width: outer.size.width, height: CGFloat(hours.count) * hourHeight, alignment: .topLeading)
                }
            }
        }
    }

    private func dayLabel(_ day: Date) -> some View {
        let calendar = model.layout.calendar
        let weekday = day.formatted(Date.FormatStyle(timeZone: calendar.timeZone).weekday(.abbreviated))
        let number = calendar.component(.day, from: day)
        let today = model.isToday(day)
        return HStack(spacing: MFSpacing.xs) {
            Text(weekday).font(MFFont.caption)
            Text("\(number)").font(MFFont.caption.weight(.bold))
                .foregroundStyle(today ? MFColor.onInk : MFColor.ink)
        }
        .foregroundStyle(today ? MFColor.onInk : MFColor.secondary)
        .padding(.horizontal, today ? MFSpacing.m : 0)
        .padding(.vertical, today ? MFSpacing.xxs : 0)
        .background(Capsule().fill(today ? MFColor.ink : .clear))
        .accessibilityElement(children: .combine)
        .accessibilityAddTraits(today ? .isSelected : [])
    }

    private func hourLabel(_ hour: Int) -> String {
        switch hour {
        case 0: "12 AM"
        case 12: "12 PM"
        case 13...: "\(hour - 12) PM"
        default: "\(hour) AM"
        }
    }
}

/// One block: solid event, dashed proposal, struck-through moving event.
struct EventBlockView: View {
    var block: WeekBlock

    var body: some View {
        VStack(alignment: .leading, spacing: MFSpacing.xxs) {
            Text(block.title)
                .font(MFFont.caption.weight(.semibold))
                .strikethrough(isMoving)
            if let detail {
                Text(detail)
                    .font(MFFont.caption)
                    .strikethrough(isMoving)
                    .lineLimit(isProposal ? 2 : 1)
                    .fixedSize(horizontal: false, vertical: isProposal)
            }
        }
        .foregroundStyle(foreground)
        .padding(.horizontal, MFSpacing.s + MFSpacing.xxs)
        .padding(.vertical, MFSpacing.xs + MFSpacing.xxs)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .background(RoundedRectangle(cornerRadius: MFRadius.s).fill(fill))
        .overlay(RoundedRectangle(cornerRadius: MFRadius.s)
            .strokeBorder(isProposal ? MFColor.accent : .clear,
                          style: StrokeStyle(lineWidth: 1.5, dash: isProposal ? [5, 4] : [])))
        .accessibilityElement(children: .combine)
        .accessibilityLabel(accessibilityText)
    }

    private var isProposal: Bool { if case .proposal = block.kind { true } else { false } }
    private var isMoving: Bool { if case .moving = block.kind { true } else { false } }

    private var detail: String? {
        switch block.kind {
        case .proposal(let proposal):
            "\(CalendarTimeFormat.time(proposal.start)) · proposed from \(proposal.source.kindLabel)"
        case .moving(let event):
            "\(CalendarTimeFormat.time(event.start)), moving"
        case .event(let event):
            block.height >= 1 ? Self.range(event.start, event.end) : nil
        }
    }

    private var accessibilityText: String {
        switch block.kind {
        case .proposal(let proposal): "Proposed: \(proposal.title), \(Self.range(proposal.start, proposal.end))"
        case .moving(let event): "\(event.title), moving"
        case .event(let event): "\(event.title), \(Self.range(event.start, event.end))"
        }
    }

    private var fill: Color {
        switch block.kind {
        case .proposal: MFColor.accentFillFaint
        case .moving: MFColor.fill
        case .event(let event): Self.palette(for: event).fill
        }
    }

    private var foreground: Color {
        switch block.kind {
        case .proposal: MFColor.accentStrong
        case .moving: MFColor.tertiary
        case .event(let event): Self.palette(for: event).text
        }
    }

    /// Calendars get the mockup's tints, by calendar name so they stay put.
    static func palette(for event: CalendarEvent) -> (fill: Color, text: Color) {
        let tints: [(Color, Color)] = [(MFColor.accentFill, MFColor.accentStrong),
                                       (MFColor.successFill, MFColor.success),
                                       (MFColor.previewFill, MFColor.preview),
                                       (MFColor.fill, MFColor.ink)]
        let index = event.calendarName.unicodeScalars.reduce(0) { ($0 &* 31 &+ Int($1.value)) & 0xFFFF } % tints.count
        return tints[index]
    }

    static func time(_ date: Date) -> String {
        date.formatted(date: .omitted, time: .shortened)
    }

    static func range(_ start: Date, _ end: Date) -> String {
        CalendarTimeFormat.range(start, end)
    }
}

/// "Move the design review to Wed 11:00?" with Dismiss and Add.
struct ProposalCard: View {
    var proposal: ProposedEvent
    var model: CalendarWeekModel

    var body: some View {
        AppleActionCard {
            HStack(spacing: MFSpacing.m) {
                let note = Text(proposal.note.map { " \($0)" } ?? "")
                    .font(MFFont.body).foregroundStyle(MFColor.secondary)
                Text("\(Text(question).font(MFFont.bodyEmphasis).foregroundStyle(MFColor.ink))\(note)")
                    .fixedSize(horizontal: false, vertical: true)
                Spacer(minLength: MFSpacing.m)
                Button("Dismiss") { model.dismiss(proposal) }
                    .buttonStyle(.appleSecondary)
                Button("Add") { Task { await model.add(proposal) } }
                    .buttonStyle(.applePrimary)
                    .disabled(model.addingProposals.contains(proposal.id))
                    .keyboardShortcut(.return, modifiers: .command)
            }
        }
    }

    private var question: String {
        let when = CalendarTimeFormat.dayTime(proposal.start)
        return proposal.replacesEventID == nil
            ? "Add \(proposal.title) on \(when)?"
            : "Move the \(proposal.title.lowercased()) to \(when)?"
    }
}

/// New event (⌘N) in the shell toolbar: its own view with its own state,
/// since the shell keeps a toolbar item by id.
struct NewEventToolbarButton: View {
    var model: CalendarWeekModel
    @State private var showsForm = false

    var body: some View {
        Button("New event") { showsForm = true }
            .buttonStyle(.applePrimary)
            .keyboardShortcut("n", modifiers: .command)
            .disabled(model.access == .blocked || model.access == .promptNotShown)
            .popover(isPresented: $showsForm, arrowEdge: .bottom) {
                NewEventForm(model: model) { showsForm = false }
            }
    }
}

/// "New event": title, start and length; added to Calendar on Add.
struct NewEventForm: View {
    var model: CalendarWeekModel
    var close: () -> Void
    @State private var title = ""
    @State private var start = NewEventForm.nextHalfHour()
    @State private var minutes = 30
    @State private var isAdding = false

    var body: some View {
        VStack(alignment: .leading, spacing: MFSpacing.m) {
            Text("New event")
                .font(MFFont.headline)
                .foregroundStyle(MFColor.ink)
            TextField("Title", text: $title)
                .textFieldStyle(.roundedBorder)
                .onSubmit(add)
            DatePicker("Starts", selection: $start)
            Picker("Length", selection: $minutes) {
                ForEach([15, 30, 45, 60, 90, 120], id: \.self) { value in
                    Text(Self.lengthLabel(value)).tag(value)
                }
            }
            if let message = model.errorMessage {
                Text(message).font(MFFont.callout).foregroundStyle(MFColor.danger)
            }
            HStack {
                Spacer()
                Button("Cancel", action: close)
                    .buttonStyle(.appleSecondary)
                    .keyboardShortcut(.cancelAction)
                Button("Add", action: add)
                    .buttonStyle(.applePrimary)
                    .keyboardShortcut(.defaultAction)
                    .disabled(title.trimmingCharacters(in: .whitespaces).isEmpty || isAdding)
            }
        }
        .padding(MFSpacing.l)
        .frame(width: 320)
    }

    private func add() {
        isAdding = true
        Task {
            let added = await model.createEvent(title: title, start: start,
                                                end: start.addingTimeInterval(Double(minutes) * 60))
            isAdding = false
            if added { close() }
        }
    }

    static func lengthLabel(_ minutes: Int) -> String {
        switch minutes {
        case ..<60: "\(minutes) min"
        case 60: "1 hour"
        default: String(format: "%g hours", Double(minutes) / 60)
        }
    }

    static func nextHalfHour(after date: Date = .now) -> Date {
        let calendar = Calendar.current
        let minute = calendar.component(.minute, from: date)
        let rounded = calendar.date(byAdding: .minute, value: minute < 30 ? 30 - minute : 60 - minute, to: date) ?? date
        return calendar.date(bySetting: .second, value: 0, of: rounded) ?? rounded
    }
}

/// An event block that opens its editor on click; proposals and moving
/// events are plain (their actions are on the proposal card).
struct EventBlockButton: View {
    var block: WeekBlock
    var model: CalendarWeekModel
    @State private var editing = false

    var body: some View {
        if case .event(let event) = block.kind {
            Button { editing = true } label: { EventBlockView(block: block) }
                .buttonStyle(.plain)
                .accessibilityHint("Edit or delete this event")
                .popover(isPresented: $editing, arrowEdge: .trailing) {
                    EventEditForm(event: event, model: model) { editing = false }
                }
        } else {
            EventBlockView(block: block)
        }
    }
}

/// Edit or delete one event (this occurrence only, for repeating events).
struct EventEditForm: View {
    var event: CalendarEvent
    var model: CalendarWeekModel
    var close: () -> Void
    @State private var title: String
    @State private var start: Date
    @State private var minutes: Int
    /// The length stays exactly the event's own until the person changes it.
    @State private var lengthEdited = false
    @State private var isWorking = false

    init(event: CalendarEvent, model: CalendarWeekModel, close: @escaping () -> Void) {
        self.event = event
        self.model = model
        self.close = close
        _title = State(initialValue: event.title)
        _start = State(initialValue: event.start)
        _minutes = State(initialValue: max(5, Int(event.end.timeIntervalSince(event.start) / 60)))
    }

    /// The new end: the original duration (to the second) unless the length
    /// was edited.
    static func end(for event: CalendarEvent, start: Date, minutes: Int?) -> Date {
        if let minutes { return start.addingTimeInterval(Double(minutes) * 60) }
        return start.addingTimeInterval(event.end.timeIntervalSince(event.start))
    }

    var body: some View {
        VStack(alignment: .leading, spacing: MFSpacing.m) {
            Text(event.occurrenceDate == nil ? "Edit event" : "Edit this occurrence")
                .font(MFFont.headline)
                .foregroundStyle(MFColor.ink)
            TextField("Title", text: $title)
                .textFieldStyle(.roundedBorder)
            DatePicker("Starts", selection: $start)
            Stepper("Length: \(minutes) min", value: Binding(
                get: { minutes },
                set: { minutes = $0; lengthEdited = true }), in: 5...(24 * 60), step: 15)
            if let message = model.errorMessage {
                Text(message).font(MFFont.callout).foregroundStyle(MFColor.danger)
            }
            HStack {
                Button("Delete", role: .destructive) {
                    isWorking = true
                    Task {
                        let done = await model.deleteEvent(event)
                        isWorking = false
                        if done { close() }
                    }
                }
                .buttonStyle(.appleSecondary)
                .foregroundStyle(MFColor.danger)
                Spacer()
                Button("Cancel", action: close)
                    .buttonStyle(.appleSecondary)
                    .keyboardShortcut(.cancelAction)
                Button("Save") {
                    isWorking = true
                    Task {
                        let done = await model.updateEvent(event, title: title, start: start,
                                                           end: EventEditForm.end(for: event, start: start,
                                                                                  minutes: lengthEdited ? minutes : nil))
                        isWorking = false
                        if done { close() }
                    }
                }
                .buttonStyle(.applePrimary)
                .keyboardShortcut(.defaultAction)
                .disabled(isWorking || title.trimmingCharacters(in: .whitespaces).isEmpty)
            }
            .disabled(isWorking)
        }
        .padding(MFSpacing.l)
        .frame(width: 340)
    }
}

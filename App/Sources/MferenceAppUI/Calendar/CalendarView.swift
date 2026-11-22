import AppKit
import EventKit
import MferenceAppCore
import MferenceDesign
import SwiftUI

/// Calendar and to-dos (mockup 04): the work week from Apple Calendar,
/// proposals drawn dashed with one-click Add, and the to-do panel.
public struct CalendarView: View {
    @Environment(CalendarWeekModel.self) private var environmentModel: CalendarWeekModel?
    @Environment(WeekAskModel.self) private var environmentWeekAsk: WeekAskModel?
    private let injected: CalendarWeekModel?
    private let injectedWeekAsk: WeekAskModel?
    private let loadsOnAppear: Bool

    /// The shell's slot: the models come from the environment.
    public init() {
        injected = nil
        injectedWeekAsk = nil
        loadsOnAppear = true
    }

    /// Snapshots and previews; `loadsOnAppear` false keeps fixture state.
    public init(model: CalendarWeekModel, weekAsk: WeekAskModel? = nil, loadsOnAppear: Bool = true) {
        injected = model
        injectedWeekAsk = weekAsk
        self.loadsOnAppear = loadsOnAppear
    }

    public var body: some View {
        if let model = injected ?? environmentModel {
            CalendarContent(model: model, weekAsk: injectedWeekAsk ?? environmentWeekAsk)
                .task { if loadsOnAppear { await model.load() } }
        } else {
            AppleEmptyState(systemImage: "calendar", title: "Calendar",
                            message: "Calendar is starting. Try again in a moment.")
        }
    }
}

struct CalendarContent: View {
    @Bindable var model: CalendarWeekModel
    /// Ask about your week; nil before Apple services exist.
    var weekAsk: WeekAskModel?
    /// Opens a Private chat seeded with the answers; nil hides the button.
    var continueInChat: (() -> Void)?
    /// The to-do panel per view, once the person toggled it; by default it
    /// shows in Day and Week and hides in Month and Year.
    @State private var todoPanel: [CalendarViewMode: Bool] = [:]

    private var showsTodos: Bool { todoPanel[model.mode] ?? (model.mode == .day || model.mode == .week) }

    var body: some View {
        HStack(spacing: 0) {
            VStack(alignment: .leading, spacing: MFSpacing.m) {
                header
                switch model.access {
                case .granted:
                    period
                    if model.mode == .month, !showsTodos, let weekAsk {
                        WeekAskPanel(ask: weekAsk, calendar: model, compact: true, continueInChat: continueInChat)
                            .frame(maxWidth: 560, alignment: .leading)
                    }
                    if model.mode == .day || model.mode == .week, let proposal = model.proposalsThisWeek.first {
                        ProposalCard(proposal: proposal, model: model)
                    }
                case .notAsked:
                    AppleEmptyState(systemImage: "calendar", title: "See your calendar here",
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
            .padding(.top, MFSpacing.l)
            .padding(.bottom, model.mode == .day || model.mode == .week ? 0 : MFSpacing.l)
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
            if showsTodos {
                Rectangle().fill(MFColor.hairline).frame(width: 1)
                TodoList(model: model, weekAsk: model.mode == .year ? nil : weekAsk, continueInChat: continueInChat)
                    .frame(width: 315)
                    .background(MFColor.canvas)
            }
        }
        .background(MFColor.surface)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        // Back from System Settings: look again while access is missing,
        // and pick up Reminders changes made while away.
        .onReceive(NotificationCenter.default.publisher(for: NSApplication.didBecomeActiveNotification)) { _ in
            Task { await model.recheckAccess() }
        }
        // Access granted, or events or reminders edited in Calendar or
        // Reminders: show them now, not after a relaunch.
        .onReceive(NotificationCenter.default.publisher(for: .EKEventStoreChanged)) { _ in
            Task {
                await model.calendarChanged()
                await weekAsk?.calendarChanged()
            }
        }
        .onChange(of: model.todos) { Task { await weekAsk?.calendarChanged() } }
        .shellToolbar(id: "calendar.new-event") { NewEventToolbarButton(model: model) }
    }

    @ViewBuilder
    private var period: some View {
        switch model.mode {
        case .day, .week: TimeGrid(model: model, ask: weekAsk)
        case .month: MonthGrid(model: model)
        case .year: YearGrid(model: model)
        }
    }

    private var header: some View {
        HStack(spacing: MFSpacing.m) {
            Text(model.title)
                .font(MFFont.title)
                .foregroundStyle(MFColor.ink)
                .lineLimit(1)
                .accessibilityAddTraits(.isHeader)
            Spacer(minLength: MFSpacing.m)
            ModeSwitch(model: model)
            Spacer(minLength: MFSpacing.m)
            HStack(spacing: MFSpacing.xs) {
                Button { Task { await model.step(-1) } } label: { Image(systemName: "chevron.left") }
                    .accessibilityLabel("Previous \(model.mode.title.lowercased())")
                    .keyboardShortcut(.leftArrow, modifiers: .command)
                    .buttonStyle(.appleQuiet)
                Button("Today") { Task { await model.showToday() } }
                    .buttonStyle(.appleSecondary)
                    .keyboardShortcut("t", modifiers: .command)
                Button { Task { await model.step(1) } } label: { Image(systemName: "chevron.right") }
                    .accessibilityLabel("Next \(model.mode.title.lowercased())")
                    .keyboardShortcut(.rightArrow, modifiers: .command)
                    .buttonStyle(.appleQuiet)
            }
            Button {
                todoPanel[model.mode] = !showsTodos
            } label: {
                Image(systemName: "checklist")
                    .foregroundStyle(showsTodos ? MFColor.accent : MFColor.secondary)
            }
            .buttonStyle(.appleQuiet)
            .accessibilityLabel(showsTodos ? "Hide to-dos" : "Show to-dos")
            .keyboardShortcut("t", modifiers: [.command, .option])
        }
    }
}

/// Day, Week, Month, Year (⌘1 to ⌘4).
struct ModeSwitch: View {
    var model: CalendarWeekModel

    var body: some View {
        HStack(spacing: 0) {
            ForEach(Array(CalendarViewMode.allCases.enumerated()), id: \.element) { index, mode in
                let selected = model.mode == mode
                Button { Task { await model.show(mode) } } label: {
                    Text(mode.title)
                        .font(selected ? MFFont.bodyEmphasis : MFFont.body)
                        .foregroundStyle(selected ? MFColor.onInk : MFColor.secondary)
                        .padding(.horizontal, MFSpacing.l)
                        .padding(.vertical, MFSpacing.xs + MFSpacing.xxs)
                        .background(Capsule().fill(selected ? MFColor.ink : .clear))
                        .contentShape(Capsule())
                }
                .buttonStyle(.plain)
                .keyboardShortcut(KeyEquivalent(Character("\(index + 1)")), modifiers: .command)
                .accessibilityAddTraits(selected ? .isSelected : [])
                .help("\(mode.title) (⌘\(index + 1))")
            }
        }
        .padding(MFSpacing.xxs)
        .background(Capsule().fill(MFColor.fill))
        .accessibilityElement(children: .contain)
        .accessibilityLabel("Calendar view")
    }
}

/// Day and week: a column per day over the whole 24 hours, scrolled to the
/// morning (or an earlier event) when the period changes.
struct TimeGrid: View {
    var model: CalendarWeekModel
    /// Draws the latest answer's open blocks and found events.
    var ask: WeekAskModel?
    @Environment(\.mfRendersStatically) private var rendersStatically
    @State private var position = ScrollPosition(edge: .top)
    /// A new event being made by double-clicking an empty slot.
    @State private var draftStart: Date?
    private let hourHeight: CGFloat = 56
    private let gutter: CGFloat = 64
    /// Room above midnight's line so its label is not cut.
    private let topInset: CGFloat = 8

    var body: some View {
        VStack(spacing: 0) {
            // The day view's title already names its day.
            if model.mode == .week {
                HStack(spacing: 0) {
                    Color.clear.frame(width: gutter, height: 1)
                    ForEach(model.days, id: \.self) { day in
                        dayLabel(day).frame(maxWidth: .infinity)
                    }
                }
                .padding(.bottom, MFSpacing.s)
            }
            allDayRow
            Rectangle().fill(MFColor.hairline).frame(height: 1)
            GeometryReader { outer in
                let grid = gridContent(width: outer.size.width)
                if rendersStatically {
                    Color.clear
                        .overlay(alignment: .top) { grid.offset(y: -scrollTarget) }
                        .clipped()
                } else {
                    ScrollView { grid }
                        .scrollPosition($position)
                        .onAppear { position.scrollTo(y: scrollTarget) }
                        .onChange(of: scrollKey) { position.scrollTo(y: scrollTarget) }
                }
            }
        }
    }

    /// Scroll again when the period or view changes.
    private var scrollKey: String { "\(model.mode.rawValue)-\(model.weekStart.timeIntervalSince1970)" }

    /// 7 AM, or an hour that shows the earliest event of the days shown.
    private var scrollTarget: CGFloat {
        let earliest = model.blocks.map(\.top).min() ?? 24
        let hour = max(0, min(7, earliest - 0.25))
        return CGFloat(hour) * hourHeight
    }

    @ViewBuilder
    private var allDayRow: some View {
        let allDay = model.outsideGrid
        if !allDay.isEmpty {
            HStack(alignment: .top, spacing: 0) {
                Text("All day")
                    .font(MFFont.caption)
                    .foregroundStyle(MFColor.tertiary)
                    .frame(width: gutter, alignment: .leading)
                ForEach(model.days.indices, id: \.self) { day in
                    VStack(alignment: .leading, spacing: MFSpacing.xxs) {
                        ForEach(allDay[day] ?? []) { event in
                            AllDayChip(event: event, model: model)
                        }
                    }
                    .padding(.horizontal, MFSpacing.xxs)
                    .frame(maxWidth: .infinity, alignment: .topLeading)
                }
            }
            .padding(.vertical, MFSpacing.xs)
            .overlay(alignment: .top) { Rectangle().fill(MFColor.hairline).frame(height: 1) }
        }
    }

    private func gridContent(width: CGFloat) -> some View {
        let columnWidth = (width - gutter) / CGFloat(max(model.days.count, 1))
        return ZStack(alignment: .topLeading) {
            VStack(spacing: 0) {
                ForEach(0..<24, id: \.self) { hour in
                    HStack(alignment: .top, spacing: 0) {
                        Text(hourLabel(hour))
                            .font(MFFont.caption)
                            .foregroundStyle(MFColor.tertiary)
                            .frame(width: gutter - MFSpacing.s, alignment: .trailing)
                            .offset(y: -7)
                        Spacer()
                    }
                    .frame(height: hourHeight, alignment: .top)
                    .overlay(alignment: .top) {
                        Rectangle().fill(MFColor.hairlineFaint).frame(height: 1).padding(.leading, gutter)
                    }
                }
            }
            ForEach(1..<max(model.days.count, 1), id: \.self) { column in
                Rectangle().fill(MFColor.hairlineFaint)
                    .frame(width: 1, height: 24 * hourHeight)
                    .offset(x: gutter + CGFloat(column) * columnWidth)
            }
            // Double-click an empty slot: a new event there.
            Color.clear
                .contentShape(Rectangle())
                .frame(width: max(width - gutter, 0), height: 24 * hourHeight)
                .offset(x: gutter)
                .onTapGesture(count: 2) { location in
                    newDraft(at: location, columnWidth: columnWidth)
                }
            ForEach(model.blocks) { block in
                let laneWidth = (columnWidth - MFSpacing.s) / CGFloat(block.lanes)
                EventBlockButton(block: block, model: model)
                    .frame(width: max(laneWidth - MFSpacing.xxs, 4),
                           height: max(CGFloat(block.height) * hourHeight - 2, 14))
                    .overlay {
                        if ringed(block) {
                            RoundedRectangle(cornerRadius: MFRadius.xs).strokeBorder(MFColor.ink, lineWidth: 2)
                        }
                    }
                    .opacity(faded(block) ? 0.45 : 1)
                    .offset(x: gutter + CGFloat(block.day) * columnWidth + MFSpacing.xs
                                + CGFloat(block.lane) * laneWidth,
                            y: CGFloat(block.top) * hourHeight + 1)
            }
            ForEach(ask?.highlightedOpenTimes ?? [], id: \.self) { interval in
                if let column = model.days.firstIndex(where: { model.calendar.isDate($0, inSameDayAs: interval.start) }) {
                    OpenTimeBlock(label: "Open · " + (ask?.text.duration(minutes: Int(interval.duration / 60)) ?? ""))
                        .frame(width: max(columnWidth - MFSpacing.s, 4),
                               height: max(CGFloat(interval.duration / 3_600) * hourHeight - 2, 14))
                        .offset(x: gutter + CGFloat(column) * columnWidth + MFSpacing.xs,
                                y: hours(interval.start) * hourHeight + 1)
                        .allowsHitTesting(false)
                }
            }
            nowLine(width: width, columnWidth: columnWidth)
            if let draftStart, let column = model.days.firstIndex(where: { model.layout.calendar.isDate($0, inSameDayAs: draftStart) }) {
                DraftBlock(start: draftStart, model: model) { self.draftStart = nil }
                    .frame(width: columnWidth - MFSpacing.s, height: hourHeight - 2)
                    .offset(x: gutter + CGFloat(column) * columnWidth + MFSpacing.xs,
                            y: hours(draftStart) * hourHeight + 1)
            }
        }
        .frame(width: width, height: 24 * hourHeight, alignment: .topLeading)
        .padding(.top, topInset)
    }

    @ViewBuilder
    private func nowLine(width: CGFloat, columnWidth: CGFloat) -> some View {
        if let column = model.days.firstIndex(where: model.isToday) {
            TimelineView(.periodic(from: .now, by: 60)) { _ in
                let y = hours(model.currentTime) * hourHeight
                ZStack(alignment: .topLeading) {
                    Rectangle().fill(MFColor.danger.opacity(0.35))
                        .frame(width: max(width - gutter, 0), height: 1)
                        .offset(x: gutter, y: y)
                    Rectangle().fill(MFColor.danger)
                        .frame(width: columnWidth, height: 2)
                        .offset(x: gutter + CGFloat(column) * columnWidth, y: y - 0.5)
                    Text(model.currentTime.formatted(date: .omitted, time: .shortened))
                        .font(MFFont.micro)
                        .foregroundStyle(MFColor.onInk)
                        .padding(.horizontal, MFSpacing.xs)
                        .background(Capsule().fill(MFColor.danger))
                        .frame(width: gutter - MFSpacing.xxs, alignment: .trailing)
                        .offset(y: y - 8)
                }
                .allowsHitTesting(false)
                .accessibilityHidden(true)
            }
        }
    }

    private func hours(_ date: Date) -> CGFloat {
        let calendar = model.layout.calendar
        return CGFloat(date.timeIntervalSince(calendar.startOfDay(for: date)) / 3_600)
    }

    private func newDraft(at location: CGPoint, columnWidth: CGFloat) {
        guard columnWidth > 0 else { return }
        let column = min(max(Int(location.x / columnWidth), 0), model.days.count - 1)
        let halfHours = min(max(Int((location.y / hourHeight * 2).rounded(.down)), 0), 46)
        guard model.days.indices.contains(column) else { return }
        draftStart = model.layout.calendar.date(byAdding: .minute, value: halfHours * 30, to: model.days[column])
    }

    /// An answer is pointing at some events or open time: the rest fade.
    private var highlighting: Bool {
        guard let ask else { return false }
        return !ask.highlightedOpenTimes.isEmpty || !ask.highlightedEventIDs.isEmpty
    }

    private func ringed(_ block: WeekBlock) -> Bool {
        guard let ask, case .event(let event) = block.kind else { return false }
        return ask.highlightedEventIDs.contains(event.id)
    }

    private func faded(_ block: WeekBlock) -> Bool { highlighting && !ringed(block) }

    private func dayLabel(_ day: Date) -> some View {
        let calendar = model.layout.calendar
        let today = model.isToday(day)
        let busiest = ask?.highlightedDays.contains { calendar.isDate($0, inSameDayAs: day) } ?? false
        return Button { Task { await model.open(day, in: .day) } } label: {
            HStack(spacing: MFSpacing.xs) {
                Text(CalendarText.weekday(day, calendar: calendar)).font(MFFont.caption)
                Text("\(calendar.component(.day, from: day))").font(MFFont.caption.weight(.bold))
                    .foregroundStyle(today ? MFColor.onInk : MFColor.ink)
                if busiest {
                    Text("Busiest left")
                        .font(MFFont.micro)
                        .lineLimit(1)
                        .fixedSize()
                        .foregroundStyle(MFColor.accentStrong)
                        .padding(.horizontal, MFSpacing.xs)
                        .background(Capsule().fill(MFColor.accentFill))
                }
            }
            .foregroundStyle(today ? MFColor.onInk : MFColor.secondary)
            .padding(.horizontal, today ? MFSpacing.m : MFSpacing.xs)
            .padding(.vertical, MFSpacing.xxs)
            .background(Capsule().fill(today ? MFColor.ink : .clear))
            .contentShape(Capsule())
        }
        .buttonStyle(.plain)
        .accessibilityAddTraits(today ? .isSelected : [])
        .accessibilityHint("Opens this day")
    }

    private func hourLabel(_ hour: Int) -> String {
        let calendar = model.layout.calendar
        guard let date = calendar.date(byAdding: .hour, value: hour, to: calendar.startOfDay(for: model.weekStart))
        else { return "" }
        return CalendarText.hour(date, calendar: calendar)
    }
}

/// The dashed block a double-click leaves, with the new-event form beside it.
struct DraftBlock: View {
    var start: Date
    var model: CalendarWeekModel
    var close: () -> Void
    @State private var showsForm = true

    var body: some View {
        VStack(alignment: .leading, spacing: MFSpacing.xxs) {
            Text("New event").font(MFFont.caption.weight(.semibold))
            Text(start.formatted(date: .omitted, time: .shortened)).font(MFFont.caption)
        }
        .foregroundStyle(MFColor.accentStrong)
        .padding(.horizontal, MFSpacing.s)
        .padding(.vertical, MFSpacing.xs)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .background(RoundedRectangle(cornerRadius: MFRadius.s).fill(MFColor.accentFillFaint))
        .overlay(RoundedRectangle(cornerRadius: MFRadius.s)
            .strokeBorder(MFColor.accent, style: StrokeStyle(lineWidth: 1.5, dash: [5, 4])))
        .popover(isPresented: $showsForm, arrowEdge: .trailing) {
            NewEventForm(model: model, start: start, close: close)
        }
        .onChange(of: showsForm) { if !showsForm { close() } }
    }
}

/// An all-day event above the grid; click to edit.
struct AllDayChip: View {
    var event: CalendarEvent
    var model: CalendarWeekModel
    @State private var editing = false

    var body: some View {
        let tint = EventBlockView.tint(for: event)
        Button { editing = true } label: {
            Text(event.title)
                .font(MFFont.caption.weight(.semibold))
                .foregroundStyle(MFColor.ink)
                .lineLimit(1)
                .padding(.horizontal, MFSpacing.s)
                .padding(.vertical, MFSpacing.xxs)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(RoundedRectangle(cornerRadius: MFRadius.xs).fill(tint.opacity(0.28)))
        }
        .buttonStyle(.plain)
        .accessibilityLabel("\(event.title), all day")
        .popover(isPresented: $editing, arrowEdge: .bottom) {
            EventEditForm(event: event, model: model) { editing = false }
        }
    }
}

/// The month: whole weeks, up to three events a day, "+N more" opens the day.
struct MonthGrid: View {
    var model: CalendarWeekModel
    private let shown = 3

    var body: some View {
        let calendar = model.calendar
        let days = model.monthGrid
        let weeks = stride(from: 0, to: days.count, by: 7).map { Array(days[$0..<min($0 + 7, days.count)]) }
        VStack(spacing: 0) {
            HStack(spacing: 0) {
                ForEach(CalendarText.weekdaySymbols(calendar: calendar), id: \.self) { symbol in
                    Text(symbol)
                        .font(MFFont.caption.weight(.semibold))
                        .foregroundStyle(MFColor.tertiary)
                        .frame(maxWidth: .infinity, alignment: .trailing)
                        .padding(.trailing, MFSpacing.s)
                }
            }
            .padding(.bottom, MFSpacing.xs)
            VStack(spacing: 0) {
                ForEach(weeks.indices, id: \.self) { row in
                    HStack(spacing: 0) {
                        ForEach(weeks[row], id: \.self) { day in
                            cell(day)
                        }
                    }
                    .frame(maxHeight: .infinity)
                }
            }
            .overlay(alignment: .top) { Rectangle().fill(MFColor.hairline).frame(height: 1) }
            .overlay(alignment: .leading) { Rectangle().fill(MFColor.hairlineFaint).frame(width: 1) }
        }
    }

    private func cell(_ day: Date) -> some View {
        let calendar = model.calendar
        let inMonth = calendar.isDate(day, equalTo: model.selectedDate, toGranularity: .month)
        let today = model.isToday(day)
        let events = model.events(on: day)
        let weekend = calendar.isDateInWeekend(day)
        return VStack(alignment: .leading, spacing: MFSpacing.xxs) {
            HStack {
                Spacer(minLength: 0)
                Button { Task { await model.open(day, in: .day) } } label: {
                    Text(CalendarText.monthDay(day, calendar: calendar))
                        .font(MFFont.callout.weight(.semibold))
                        .foregroundStyle(today ? MFColor.onInk : inMonth ? MFColor.ink : MFColor.tertiary)
                        .padding(.horizontal, MFSpacing.s)
                        .frame(minWidth: 24, minHeight: 24)
                        .background(Capsule().fill(today ? MFColor.ink : .clear))
                        .contentShape(Capsule())
                }
                .buttonStyle(.plain)
                .accessibilityLabel(day.formatted(date: .complete, time: .omitted))
                .accessibilityHint("Opens this day")
            }
            ForEach(events.prefix(shown)) { event in
                MonthEventChip(event: event, model: model)
            }
            if events.count > shown {
                Button("+\(events.count - shown) more") { Task { await model.open(day, in: .day) } }
                    .buttonStyle(.plain)
                    .font(MFFont.caption.weight(.semibold))
                    .foregroundStyle(MFColor.secondary)
                    .padding(.leading, MFSpacing.xs)
            }
            Spacer(minLength: 0)
        }
        .padding(MFSpacing.xs)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .background(weekend ? MFColor.fillSubtle : .clear)
        .overlay(alignment: .trailing) { Rectangle().fill(MFColor.hairlineFaint).frame(width: 1) }
        .overlay(alignment: .bottom) { Rectangle().fill(MFColor.hairlineFaint).frame(height: 1) }
        .contentShape(Rectangle())
        .onTapGesture(count: 2) { Task { await model.open(day, in: .day) } }
        .clipped()
    }
}

/// One event in a month cell: a dot, the time, the title; click to edit.
struct MonthEventChip: View {
    var event: CalendarEvent
    var model: CalendarWeekModel
    @State private var editing = false

    var body: some View {
        let tint = EventBlockView.tint(for: event)
        Button { editing = true } label: {
            HStack(spacing: MFSpacing.xs) {
                Circle().fill(tint).frame(width: 7, height: 7)
                if !event.isAllDay {
                    Text(event.start.formatted(date: .omitted, time: .shortened))
                        .foregroundStyle(MFColor.tertiary)
                        .monospacedDigit()
                }
                Text(event.title)
                    .foregroundStyle(MFColor.ink)
                    .fontWeight(event.isAllDay ? .semibold : .regular)
            }
            .font(MFFont.caption)
            .lineLimit(1)
            .padding(.horizontal, MFSpacing.xs)
            .padding(.vertical, 1)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(RoundedRectangle(cornerRadius: MFRadius.xs).fill(event.isAllDay ? tint.opacity(0.25) : .clear))
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityLabel(event.isAllDay ? "\(event.title), all day"
                            : "\(event.title), \(EventBlockView.range(event.start, event.end))")
        .popover(isPresented: $editing, arrowEdge: .trailing) {
            EventEditForm(event: event, model: model) { editing = false }
        }
    }
}

/// The year: twelve small months; a dot marks a day with events. A month's
/// name opens it, a date opens that day.
struct YearGrid: View {
    var model: CalendarWeekModel

    var body: some View {
        let calendar = model.calendar
        let year = calendar.dateInterval(of: .year, for: model.selectedDate)?.start ?? model.selectedDate
        let months = (0..<12).compactMap { calendar.date(byAdding: .month, value: $0, to: year) }
        let busy = model.daysWithEvents
        MFScroll {
            LazyVGrid(columns: Array(repeating: GridItem(.flexible(), spacing: MFSpacing.xl, alignment: .top), count: 4),
                      alignment: .leading, spacing: MFSpacing.l) {
                ForEach(months, id: \.self) { month in
                    MiniMonth(model: model, month: month, busyDays: busy, showsTitle: true,
                              highlightsSelection: false) { day in
                        Task { await model.open(day, in: .day) }
                    } openMonth: {
                        Task { await model.open(month, in: .month) }
                    }
                }
            }
            .padding(.top, MFSpacing.s)
        }
    }
}

/// A small month: weekday initials, dates, today filled, the selected day
/// ringed, a dot under days with events. Used by the year view and the
/// to-do panel.
struct MiniMonth: View {
    var model: CalendarWeekModel
    var month: Date
    var busyDays: Set<Date>
    var showsTitle: Bool
    var highlightsSelection: Bool
    var pick: (Date) -> Void
    var openMonth: (() -> Void)?

    var body: some View {
        let calendar = model.calendar
        let days = model.monthDays(containing: month)
        let isCurrentMonth = calendar.isDate(month, equalTo: model.currentTime, toGranularity: .month)
        VStack(alignment: .leading, spacing: MFSpacing.xxs) {
            if showsTitle {
                Button { openMonth?() } label: {
                    Text(CalendarText.monthName(month, calendar: calendar))
                        .font(MFFont.headline)
                        .foregroundStyle(isCurrentMonth ? MFColor.danger : MFColor.ink)
                }
                .buttonStyle(.plain)
                .accessibilityHint("Opens this month")
            }
            Grid(horizontalSpacing: 0, verticalSpacing: 0) {
                GridRow {
                    ForEach(Array(CalendarText.weekdayInitials(calendar: calendar).enumerated()), id: \.offset) { _, initial in
                        Text(initial)
                            .font(MFFont.micro)
                            .foregroundStyle(MFColor.tertiary)
                            .frame(maxWidth: .infinity)
                            .padding(.bottom, MFSpacing.xxs)
                    }
                }
                ForEach(Array(stride(from: 0, to: days.count, by: 7)), id: \.self) { row in
                    GridRow {
                        ForEach(days[row..<min(row + 7, days.count)], id: \.self) { day in
                            dayButton(day, inMonth: calendar.isDate(day, equalTo: month, toGranularity: .month))
                        }
                    }
                }
            }
        }
    }

    @ViewBuilder
    private func dayButton(_ day: Date, inMonth: Bool) -> some View {
        let calendar = model.calendar
        if !inMonth && showsTitle {
            Color.clear.frame(maxWidth: .infinity, minHeight: 25, maxHeight: 25)
        } else {
            let today = model.isToday(day)
            let selected = highlightsSelection && model.isSelected(day)
            Button { pick(day) } label: {
                ZStack(alignment: .bottom) {
                    Text("\(calendar.component(.day, from: day))")
                        .font(MFFont.caption.weight(today ? .bold : .regular))
                        .monospacedDigit()
                        .foregroundStyle(today ? MFColor.onInk : inMonth ? MFColor.ink : MFColor.tertiary)
                        .frame(width: 22, height: 22)
                        .background(Circle().fill(today ? MFColor.ink : .clear))
                        .overlay(Circle().strokeBorder(selected && !today ? MFColor.accent : .clear, lineWidth: 1.5))
                    if busyDays.contains(calendar.startOfDay(for: day)) {
                        Circle().fill(today ? MFColor.accent : MFColor.tertiary).frame(width: 3, height: 3).offset(y: 2)
                    }
                }
                .frame(maxWidth: .infinity, minHeight: 25, maxHeight: 25)
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .accessibilityLabel(day.formatted(date: .complete, time: .omitted))
            .accessibilityAddTraits(selected ? .isSelected : [])
        }
    }
}

/// Dates and hours in this Mac's language and region.
enum CalendarText {
    static func formatter(_ template: String, calendar: Calendar) -> DateFormatter {
        let formatter = DateFormatter()
        formatter.calendar = calendar
        formatter.timeZone = calendar.timeZone
        formatter.locale = calendar.locale ?? .autoupdatingCurrent
        formatter.setLocalizedDateFormatFromTemplate(template)
        return formatter
    }

    /// "Wed".
    static func weekday(_ date: Date, calendar: Calendar) -> String { formatter("EEE", calendar: calendar).string(from: date) }

    /// "9 AM", or "09" on a 24-hour Mac.
    static func hour(_ date: Date, calendar: Calendar) -> String { formatter("j", calendar: calendar).string(from: date) }

    /// "November".
    static func monthName(_ date: Date, calendar: Calendar) -> String { formatter("MMMM", calendar: calendar).string(from: date) }

    /// "18", or "Dec 1" on the first of a month.
    static func monthDay(_ date: Date, calendar: Calendar) -> String {
        calendar.component(.day, from: date) == 1
            ? formatter("MMMd", calendar: calendar).string(from: date)
            : "\(calendar.component(.day, from: date))"
    }

    /// Short weekday names from the week's first day ("Sun", "Mon", ...).
    static func weekdaySymbols(calendar: Calendar) -> [String] {
        rotate(formatter("EEE", calendar: calendar).shortWeekdaySymbols ?? calendar.shortWeekdaySymbols, calendar: calendar)
    }

    /// One-letter weekday names from the week's first day.
    static func weekdayInitials(calendar: Calendar) -> [String] {
        rotate(formatter("EEE", calendar: calendar).veryShortWeekdaySymbols ?? calendar.veryShortWeekdaySymbols,
               calendar: calendar)
    }

    private static func rotate(_ symbols: [String], calendar: Calendar) -> [String] {
        let first = max(calendar.firstWeekday - 1, 0) % max(symbols.count, 1)
        return Array(symbols[first...] + symbols[..<first])
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
        .padding(.leading, MFSpacing.s + MFSpacing.xxs)
        .padding(.trailing, MFSpacing.xs)
        .padding(.vertical, block.height < 0.5 ? 0 : MFSpacing.xxs + 1)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .background(RoundedRectangle(cornerRadius: MFRadius.s).fill(fill))
        .overlay(alignment: .leading) {
            if case .event(let event) = block.kind {
                RoundedRectangle(cornerRadius: 1.5).fill(Self.tint(for: event)).frame(width: 3).padding(.vertical, 2)
            }
        }
        .clipShape(RoundedRectangle(cornerRadius: MFRadius.s))
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
            block.height >= 0.75 ? Self.range(event.start, event.end) : nil
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
        case .event(let event): Self.tint(for: event).opacity(0.22)
        }
    }

    private var foreground: Color {
        switch block.kind {
        case .proposal: MFColor.accentStrong
        case .moving: MFColor.tertiary
        case .event: MFColor.ink
        }
    }

    /// The calendar's own color from Apple Calendar; a steady fallback
    /// picked by calendar name when it has none.
    static func tint(for event: CalendarEvent) -> Color {
        if let hex = event.calendarColor {
            return Color(red: Double((hex >> 16) & 0xFF) / 255, green: Double((hex >> 8) & 0xFF) / 255,
                         blue: Double(hex & 0xFF) / 255)
        }
        let tints = [MFColor.accent, MFColor.success, MFColor.preview, MFColor.secondary]
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
    @State private var start: Date
    @State private var minutes: Int
    @State private var isAdding = false

    /// `start` nil: the next half hour (toolbar); a double-clicked slot
    /// passes its time and gets an hour.
    init(model: CalendarWeekModel, start: Date? = nil, close: @escaping () -> Void) {
        self.model = model
        self.close = close
        _start = State(initialValue: start ?? NewEventForm.nextHalfHour())
        _minutes = State(initialValue: start == nil ? 30 : 60)
    }

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

/// An open block an answer found, drawn dashed over the empty grid.
struct OpenTimeBlock: View {
    var label: String

    var body: some View {
        RoundedRectangle(cornerRadius: MFRadius.xs)
            .fill(MFColor.successFill)
            .overlay(RoundedRectangle(cornerRadius: MFRadius.xs)
                .strokeBorder(MFColor.success, style: StrokeStyle(lineWidth: 1.5, dash: [5, 4])))
            .overlay(alignment: .topLeading) {
                Text(label)
                    .font(MFFont.caption.weight(.semibold))
                    .foregroundStyle(MFColor.success)
                    .padding(MFSpacing.xs)
            }
            .accessibilityElement(children: .combine)
    }
}

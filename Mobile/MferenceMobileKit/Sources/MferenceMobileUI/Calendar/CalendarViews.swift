import MferenceDesign
import SwiftUI

/// Calendar on iPhone and Duo closed: the month title, Today and New event,
/// then Day / Month / To-dos.
struct CalendarScreen: View {
    @Bindable var model: MobileAppModel
    @State private var isAddingEvent = false

    var body: some View {
        VStack(spacing: 0) {
            CalendarHeader(model: model, title: monthTitle, isAddingEvent: $isAddingEvent)
                .padding(.horizontal, MFPhoneMetrics.screenMargin)
            SegmentedPills(options: CalendarMode.allCases, selection: $model.calendarMode) { mode in
                mode == .todos ? "To-dos · \(model.openTodoCount)" : mode.rawValue
            }
            .padding(.horizontal, MFPhoneMetrics.screenMargin)
            .padding(.vertical, MFSpacing.m)
            switch model.calendarMode {
            case .day:
                WeekStrip(model: model)
                    .padding(.horizontal, MFSpacing.m)
                Divider().overlay(MFColor.chromeDivider)
                DayTimeline(model: model, margin: MFPhoneMetrics.screenMargin)
                    .safeAreaInset(edge: .bottom) {
                        if model.suggestion != nil {
                            SuggestionBanner(model: model)
                                .padding(.horizontal, MFSpacing.m)
                                .padding(.bottom, MFSpacing.s)
                        }
                    }
            case .month:
                ScrollView {
                    VStack(alignment: .leading, spacing: MFSpacing.l) {
                        MonthGridView(model: model, switchesToDay: true)
                        DayAgenda(model: model)
                    }
                    .padding(.horizontal, MFPhoneMetrics.screenMargin)
                    .padding(.bottom, MFSpacing.xl)
                }
            case .todos:
                ScrollView {
                    TodoListView(model: model)
                        .padding(.horizontal, MFPhoneMetrics.screenMargin)
                        .padding(.bottom, MFSpacing.xl)
                }
            }
        }
        .background(MFColor.canvas)
        .sheet(isPresented: $isAddingEvent) {
            NewEventSheet(model: model)
                .presentationDetents([.medium])
        }
    }

    private var monthTitle: String {
        model.selectedDay.formatted(.dateTime.month(.wide))
    }
}

/// Title, Today and the ink + button.
struct CalendarHeader: View {
    @Bindable var model: MobileAppModel
    let title: String
    @Binding var isAddingEvent: Bool
    var titleFont: Font = MFPhoneFont.largeTitle

    var body: some View {
        HStack(spacing: MFSpacing.s) {
            Text(title)
                .font(titleFont)
                .foregroundStyle(MFColor.ink)
                .accessibilityAddTraits(.isHeader)
            Spacer()
            Button("Today") { model.goToToday() }
                .buttonStyle(PhoneButtonStyle(kind: .secondary, size: .compact))
            Button {
                isAddingEvent = true
            } label: {
                Image(systemName: "plus")
                    .font(.system(size: 17, weight: .semibold))
                    .foregroundStyle(MFColor.onInk)
                    .frame(width: MFPhoneMetrics.touchTarget, height: MFPhoneMetrics.touchTarget)
                    .background(Circle().fill(MFColor.ink))
            }
            .buttonStyle(.plain)
            .accessibilityLabel("New event")
        }
        .frame(minHeight: MFPhoneMetrics.touchTarget)
        .padding(.top, MFSpacing.xs)
    }
}

/// The seven days around the selected day; swipe for the next or last week.
struct WeekStrip: View {
    @Bindable var model: MobileAppModel

    var body: some View {
        let days = MonthGrid.week(containing: model.selectedDay, calendar: model.calendar)
        HStack(spacing: 0) {
            ForEach(days, id: \.self) { day in
                let isSelected = model.calendar.isDate(day, inSameDayAs: model.selectedDay)
                let isToday = model.calendar.isDateInToday(day)
                Button {
                    model.select(day: day)
                } label: {
                    VStack(spacing: MFSpacing.xs) {
                        Text(day.formatted(.dateTime.weekday(.abbreviated)))
                            .font(MFPhoneFont.micro.weight(isSelected ? .semibold : .regular))
                            .foregroundStyle(isSelected ? MFColor.ink : MFColor.tertiary)
                        Text(day.formatted(.dateTime.day()))
                            .font(MFFont.geist(17, isSelected ? .semibold : .regular, relativeTo: .body))
                            .foregroundStyle(isSelected ? MFColor.onInk : (isToday ? MFColor.accent : MFColor.ink))
                            .frame(width: 36, height: 36)
                            .background(Circle().fill(isSelected ? MFColor.ink : .clear))
                        Circle().fill(model.hasEvents(on: day) && !isSelected ? MFColor.tertiary : .clear)
                            .frame(width: 4, height: 4)
                    }
                    .frame(maxWidth: .infinity, minHeight: 64)
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .accessibilityLabel(day.formatted(date: .complete, time: .omitted))
                .accessibilityAddTraits(isSelected ? .isSelected : [])
            }
        }
        .gesture(DragGesture(minimumDistance: 30).onEnded { value in
            let weeks = value.translation.width < 0 ? 1 : -1
            if let day = model.calendar.date(byAdding: .day, value: 7 * weeks, to: model.selectedDay) {
                withAnimation(.snappy(duration: 0.2)) { model.select(day: day) }
            }
        })
        .sensoryFeedback(.selection, trigger: model.selectedDay)
    }
}

/// The full 24-hour day, scrolled to the morning, with the now line on today.
struct DayTimeline: View {
    @Bindable var model: MobileAppModel
    var margin: CGFloat

    private let gutter: CGFloat = 54
    private var hourHeight: CGFloat { MFPhoneMetrics.hourHeight }

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                ZStack(alignment: .topLeading) {
                    VStack(spacing: 0) {
                        ForEach(0..<24, id: \.self) { hour in
                            HStack(alignment: .top, spacing: MFSpacing.s) {
                                Text(hourLabel(hour))
                                    .font(MFPhoneFont.micro)
                                    .foregroundStyle(MFColor.tertiary)
                                    .frame(width: gutter - MFSpacing.s, alignment: .leading)
                                    .offset(y: -7)
                                Rectangle().fill(MFColor.hairlineFaint).frame(height: MFLayout.hairline)
                            }
                            .frame(height: hourHeight, alignment: .top)
                            .id(hour)
                        }
                    }
                    GeometryReader { geometry in
                        let width = geometry.size.width - gutter
                        ForEach(model.events(on: model.selectedDay)) { event in
                            EventBlock(event: event)
                                .frame(width: width, height: max(28, height(of: event) - 2))
                                .offset(x: gutter, y: y(of: event.start) + 1)
                        }
                        if model.calendar.isDateInToday(model.selectedDay) {
                            TimelineView(.everyMinute) { context in
                                let nowY = y(of: context.date)
                                ZStack(alignment: .leading) {
                                    Rectangle().fill(MFColor.accent).frame(height: 2)
                                        .padding(.leading, gutter - 6)
                                    Circle().fill(MFColor.accent).frame(width: 12, height: 12)
                                        .padding(.leading, gutter - 12)
                                }
                                .offset(y: nowY - 6)
                                .accessibilityHidden(true)
                            }
                        }
                    }
                }
                .frame(height: hourHeight * 24)
                .padding(.horizontal, margin)
                .padding(.top, MFSpacing.m)
            }
            .onAppear { proxy.scrollTo(8, anchor: .top) }
            .onChange(of: model.selectedDay) { proxy.scrollTo(8, anchor: .top) }
        }
    }

    private func minutes(_ date: Date) -> CGFloat {
        let parts = model.calendar.dateComponents([.hour, .minute], from: date)
        return CGFloat((parts.hour ?? 0) * 60 + (parts.minute ?? 0))
    }

    private func y(of date: Date) -> CGFloat { minutes(date) / 60 * hourHeight }

    private func height(of event: PhoneEvent) -> CGFloat {
        CGFloat(event.end.timeIntervalSince(event.start) / 3600) * hourHeight
    }

    private func hourLabel(_ hour: Int) -> String {
        switch hour {
        case 0: "12 AM"
        case 12: "Noon"
        case 1..<12: "\(hour) AM"
        default: "\(hour - 12) PM"
        }
    }
}

/// One event in the day grid.
struct EventBlock: View {
    let event: PhoneEvent

    var body: some View {
        let short = event.end.timeIntervalSince(event.start) < 45 * 60
        VStack(alignment: .leading, spacing: 1) {
            if short {
                Text("\(Text(event.title).fontWeight(.semibold)) \(event.start.formatted(date: .omitted, time: .shortened))")
            } else {
                Text(event.title).fontWeight(.semibold)
                Text(TimeRangeText.text(from: event.start, to: event.end))
            }
        }
        .font(MFFont.geist(14, .regular, relativeTo: .subheadline))
        .foregroundStyle(event.tint.text)
        .lineLimit(1)
        .padding(.horizontal, MFSpacing.compact)
        .padding(.vertical, short ? 4 : 6)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .background(RoundedRectangle(cornerRadius: MFRadius.s).fill(event.tint.fill))
        .accessibilityElement(children: .combine)
    }
}

/// "Move the design review to Thu 11:00?" with Dismiss and Add.
struct SuggestionBanner: View {
    @Bindable var model: MobileAppModel

    var body: some View {
        if let suggestion = model.suggestion {
            VStack(alignment: .leading, spacing: MFSpacing.compact) {
                Text("\(Text(suggestion.question).fontWeight(.semibold).foregroundStyle(MFColor.ink)) \(suggestion.detail)")
                    .font(MFPhoneFont.preview)
                    .foregroundStyle(MFColor.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                HStack(spacing: MFSpacing.s) {
                    Spacer()
                    Button("Dismiss") { withAnimation { model.dismissSuggestion() } }
                        .buttonStyle(PhoneButtonStyle(kind: .secondary, size: .compact))
                    Button("Add") { withAnimation { model.acceptSuggestion() } }
                        .buttonStyle(PhoneButtonStyle(kind: .primary, size: .compact))
                }
            }
            .padding(.horizontal, 14)
            .padding(.vertical, MFSpacing.m)
            .background(RoundedRectangle(cornerRadius: MFRadius.l).fill(MFColor.accentFillFaint))
            .overlay(RoundedRectangle(cornerRadius: MFRadius.l).strokeBorder(MFColor.accentFill, lineWidth: MFLayout.hairline))
            .transition(.move(edge: .bottom).combined(with: .opacity))
        }
    }
}

/// The month grid. Week start follows the device setting.
struct MonthGridView: View {
    @Bindable var model: MobileAppModel
    var switchesToDay = false

    var body: some View {
        let month = model.calendar.date(byAdding: .month, value: model.monthOffset, to: model.selectedDay) ?? model.selectedDay
        let grid = MonthGrid(month: month, calendar: model.calendar)
        VStack(spacing: MFSpacing.s) {
            HStack {
                Text(month.formatted(.dateTime.month(.wide).year()))
                    .font(MFPhoneFont.headline)
                    .foregroundStyle(MFColor.ink)
                Spacer()
                Button { model.monthOffset -= 1 } label: { Image(systemName: "chevron.left") }
                    .accessibilityLabel("Previous month")
                    .frame(width: MFPhoneMetrics.touchTarget, height: MFPhoneMetrics.touchTarget)
                Button { model.monthOffset += 1 } label: { Image(systemName: "chevron.right") }
                    .accessibilityLabel("Next month")
                    .frame(width: MFPhoneMetrics.touchTarget, height: MFPhoneMetrics.touchTarget)
            }
            .foregroundStyle(MFColor.ink)
            LazyVGrid(columns: Array(repeating: GridItem(.flexible(), spacing: 0), count: 7), spacing: 2) {
                ForEach(Array(grid.weekdaySymbols.enumerated()), id: \.offset) { _, symbol in
                    Text(symbol).font(MFPhoneFont.micro).foregroundStyle(MFColor.tertiary)
                        .frame(height: 20)
                }
                ForEach(Array(grid.cells.enumerated()), id: \.offset) { _, day in
                    if let day {
                        dayCell(day)
                    } else {
                        Color.clear.frame(height: 44)
                    }
                }
            }
        }
    }

    private func dayCell(_ day: Date) -> some View {
        let isSelected = model.calendar.isDate(day, inSameDayAs: model.selectedDay)
        let isToday = model.calendar.isDateInToday(day)
        return Button {
            model.select(day: day, switchToDay: switchesToDay)
        } label: {
            VStack(spacing: 2) {
                Text(day.formatted(.dateTime.day()))
                    .font(MFFont.geist(16, isSelected ? .semibold : .regular, relativeTo: .callout))
                    .foregroundStyle(isSelected ? MFColor.onInk : (isToday ? MFColor.accent : MFColor.ink))
                    .frame(width: 34, height: 34)
                    .background(Circle().fill(isSelected ? MFColor.ink : .clear))
                Circle().fill(model.hasEvents(on: day) ? MFColor.tertiary : .clear).frame(width: 4, height: 4)
            }
            .frame(maxWidth: .infinity, minHeight: 44)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityLabel(day.formatted(date: .complete, time: .omitted))
    }
}

/// The selected day's events as a list (under the month grid).
struct DayAgenda: View {
    @Bindable var model: MobileAppModel

    var body: some View {
        let events = model.events(on: model.selectedDay)
        VStack(alignment: .leading, spacing: MFSpacing.s) {
            PhoneSectionHeader(model.selectedDay.formatted(.dateTime.weekday(.wide).month(.abbreviated).day()))
            if events.isEmpty {
                Text("Nothing scheduled.").font(MFPhoneFont.preview).foregroundStyle(MFColor.secondary)
                    .padding(.horizontal, MFSpacing.xs)
            } else {
                PhoneCard {
                    ForEach(events) { event in
                        HStack(spacing: MFSpacing.m) {
                            RoundedRectangle(cornerRadius: 2).fill(event.tint.text).frame(width: 4, height: 28)
                            Text(event.title).font(MFPhoneFont.rowTitle).foregroundStyle(MFColor.ink)
                            Spacer()
                            Text(event.start.formatted(date: .omitted, time: .shortened))
                                .font(MFPhoneFont.preview).foregroundStyle(MFColor.secondary)
                        }
                        .padding(.horizontal, MFSpacing.l)
                        .frame(minHeight: MFPhoneMetrics.rowHeight)
                    }
                }
            }
        }
    }
}

/// To-dos from mail and meetings, plus a field to add one.
struct TodoListView: View {
    @Bindable var model: MobileAppModel
    @State private var newTitle = ""

    var body: some View {
        VStack(alignment: .leading, spacing: MFSpacing.s) {
            HStack(alignment: .firstTextBaseline, spacing: MFSpacing.s) {
                Text("To-dos").font(MFPhoneFont.headline).foregroundStyle(MFColor.ink)
                Text("from mail and meetings").font(MFPhoneFont.caption).foregroundStyle(MFColor.tertiary)
            }
            .padding(.horizontal, MFSpacing.xs)
            HStack(spacing: MFSpacing.s) {
                Image(systemName: "plus.circle").foregroundStyle(MFColor.tertiary)
                TextField("New to-do", text: $newTitle)
                    .font(MFPhoneFont.callout)
                    .submitLabel(.done)
                    .onSubmit {
                        model.addTodo(newTitle)
                        newTitle = ""
                    }
            }
            .padding(.horizontal, 14)
            .frame(height: MFPhoneMetrics.buttonHeight)
            .background(RoundedRectangle(cornerRadius: MFRadius.m).fill(MFColor.surface))
            .overlay(RoundedRectangle(cornerRadius: MFRadius.m).strokeBorder(MFColor.hairline, lineWidth: MFLayout.hairline))
            PhoneCard {
                ForEach(model.todos) { todo in
                    Button {
                        withAnimation(.snappy(duration: 0.2)) { model.toggle(todo) }
                    } label: {
                        HStack(alignment: .top, spacing: MFSpacing.m) {
                            ZStack {
                                Circle().strokeBorder(todo.isDone ? MFColor.ink : MFColor.controlBorder, lineWidth: 1.5)
                                if todo.isDone {
                                    Circle().fill(MFColor.ink)
                                    Image(systemName: "checkmark").font(.system(size: 11, weight: .bold))
                                        .foregroundStyle(MFColor.onInk)
                                }
                            }
                            .frame(width: 22, height: 22)
                            VStack(alignment: .leading, spacing: MFSpacing.xs) {
                                Text(todo.title)
                                    .font(MFPhoneFont.callout)
                                    .strikethrough(todo.isDone)
                                    .foregroundStyle(todo.isDone ? MFColor.tertiary : MFColor.ink)
                                if !todo.isDone {
                                    PhoneChip(title: todo.detail, tone: todo.detail.hasPrefix("Meeting") ? .accent : .neutral)
                                }
                            }
                            Spacer(minLength: 0)
                        }
                        .padding(.horizontal, MFSpacing.l)
                        .padding(.vertical, MFSpacing.m)
                        .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                    .accessibilityAddTraits(todo.isDone ? .isSelected : [])
                }
            }
        }
        .sensoryFeedback(.success, trigger: model.todos.filter(\.isDone).count)
    }
}

/// A small form for a new event on the selected day.
struct NewEventSheet: View {
    @Bindable var model: MobileAppModel
    @Environment(\.dismiss) private var dismiss
    @State private var title = ""
    @State private var start = Date()
    @State private var minutes = 30

    var body: some View {
        NavigationStack {
            Form {
                TextField("Title", text: $title)
                DatePicker("Starts", selection: $start)
                Picker("Length", selection: $minutes) {
                    ForEach([15, 30, 45, 60, 90, 120], id: \.self) { Text("\($0) min").tag($0) }
                }
            }
            .navigationTitle("New event")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) { Button("Cancel") { dismiss() } }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Add") {
                        let end = start.addingTimeInterval(TimeInterval(minutes * 60))
                        model.events.append(PhoneEvent(title: title.isEmpty ? "New event" : title,
                                                       start: start, end: end, tint: .blue))
                        model.select(day: start, switchToDay: true)
                        dismiss()
                    }
                }
            }
        }
        .tint(MFColor.ink)
        .onAppear {
            let hour = model.calendar.component(.hour, from: .now) + 1
            start = model.calendar.date(bySettingHour: min(hour, 23), minute: 0, second: 0, of: model.selectedDay)
                ?? model.selectedDay
        }
    }
}

/// "10 to 11 AM", "9:30 to 10 AM", "11:30 AM to 1 PM" on a 12-hour clock, as
/// the Mac calendar writes them; the system's short times on a 24-hour clock.
enum TimeRangeText {
    static func text(from start: Date, to end: Date, calendar: Calendar = .current, locale: Locale = .current) -> String {
        let template = DateFormatter.dateFormat(fromTemplate: "j", options: 0, locale: locale) ?? ""
        guard template.contains("a") else {
            return "\(start.formatted(date: .omitted, time: .shortened)) to \(end.formatted(date: .omitted, time: .shortened))"
        }
        func parts(_ date: Date) -> (time: String, period: String) {
            let components = calendar.dateComponents([.hour, .minute], from: date)
            let hour = components.hour ?? 0
            let minute = components.minute ?? 0
            let shown = hour % 12 == 0 ? 12 : hour % 12
            let time = minute == 0 ? "\(shown)" : "\(shown):" + (minute < 10 ? "0\(minute)" : "\(minute)")
            return (time, hour < 12 ? "AM" : "PM")
        }
        let first = parts(start)
        let second = parts(end)
        if first.period == second.period {
            return "\(first.time) to \(second.time) \(second.period)"
        }
        return "\(first.time) \(first.period) to \(second.time) \(second.period)"
    }
}

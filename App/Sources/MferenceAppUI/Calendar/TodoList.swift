import MferenceAppCore
import MferenceDesign
import SwiftUI

/// The to-do panel: a mini month to pick the day, that day's date on top,
/// the to-dos grouped by date, the Reminders sync state, and the box that
/// adds a to-do dated to the selected day.
struct TodoList: View {
    var model: CalendarWeekModel
    @State private var newTitle = ""
    /// The date chip once the person changed it; nil follows the selected day.
    @State private var dueChoice: DueChoice?
    @State private var picksDate = false
    @State private var pickedDate = Date.now
    @State private var showsCompleted = false
    /// The mini month's page, in months from the selected day's month.
    @State private var monthOffset = 0
    @FocusState private var addFieldFocused: Bool

    enum DueChoice: Equatable {
        case date(Date)
        case none
    }

    private var calendar: Calendar { model.calendar }

    /// The new to-do's date: the chip's choice, else the selected day.
    private var due: Date? {
        switch dueChoice {
        case .date(let date)?: date
        case .none?: nil
        case nil: model.selectedDate
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: MFSpacing.s) {
            miniMonth
            Rectangle().fill(MFColor.hairline).frame(height: 1).padding(.vertical, MFSpacing.xs)
            HStack(alignment: .center, spacing: MFSpacing.s) {
                Text("To-dos")
                    .font(MFFont.headline)
                    .foregroundStyle(MFColor.ink)
                    .accessibilityAddTraits(.isHeader)
                DateTag(text: TodoDateFormat.short(model.selectedDate, calendar: calendar))
                    .accessibilityLabel("Selected day \(model.selectedDate.formatted(date: .long, time: .omitted))")
                Spacer(minLength: 0)
                if model.remindersSyncEnabled {
                    Button { Task { await model.syncReminders() } } label: {
                        Image(systemName: "arrow.triangle.2.circlepath")
                            .symbolEffect(.rotate, isActive: model.isSyncingReminders)
                    }
                    .buttonStyle(.appleQuiet)
                    .disabled(model.isSyncingReminders)
                    .accessibilityLabel("Sync with Reminders now")
                }
            }
            syncLine
            MFScroll {
                LazyVStack(alignment: .leading, spacing: 0) {
                    ForEach(model.todoSections) { section in
                        sectionView(section)
                    }
                }
            }
            addBox
        }
        .padding(MFSpacing.l)
    }

    // MARK: Mini month

    private var miniMonth: some View {
        let month = calendar.date(byAdding: .month, value: monthOffset, to: model.selectedDate) ?? model.selectedDate
        return VStack(alignment: .leading, spacing: MFSpacing.xs) {
            HStack {
                Text(month.formatted(.dateTime.month(.wide).year()))
                    .font(MFFont.bodyEmphasis)
                    .foregroundStyle(MFColor.ink)
                Spacer()
                Button { monthOffset -= 1 } label: { Image(systemName: "chevron.left") }
                    .accessibilityLabel("Previous month")
                Button { monthOffset += 1 } label: { Image(systemName: "chevron.right") }
                    .accessibilityLabel("Next month")
            }
            .buttonStyle(.appleQuiet)
            MiniMonth(model: model, month: month, busyDays: model.daysWithEvents, showsTitle: false,
                      highlightsSelection: true) { day in
                monthOffset = 0
                dueChoice = nil
                Task { await model.select(day) }
            }
        }
        .onChange(of: model.selectedDate) { monthOffset = 0 }
    }

    // MARK: Sync line

    @ViewBuilder
    private var syncLine: some View {
        HStack(spacing: MFSpacing.xs) {
            if model.remindersSyncEnabled {
                Image(systemName: model.isSyncingReminders ? "arrow.triangle.2.circlepath" : "checkmark")
                    .font(MFFont.caption.weight(.bold))
                    .foregroundStyle(MFColor.success)
                Text(syncText)
            } else {
                Text("Apple Reminders sync is off.")
                Button("Turn it on") { model.openSettings?() }
                    .buttonStyle(.plain)
                    .font(MFFont.caption.weight(.semibold))
                    .foregroundStyle(MFColor.accent)
                    .accessibilityHint("Opens Settings > Connections, where Reminders sync is turned on")
            }
        }
        .font(MFFont.caption)
        .foregroundStyle(MFColor.tertiary)
    }

    private var syncText: String {
        if model.isSyncingReminders { return "Syncing with Apple Reminders…" }
        guard let last = model.lastReminderSync else { return "Syncs with Apple Reminders" }
        return "Synced with Apple Reminders · \(last.formatted(date: .omitted, time: .shortened))"
    }

    // MARK: Sections

    @ViewBuilder
    private func sectionView(_ section: TodoSection) -> some View {
        switch section.kind {
        case .overdue:
            header("Overdue · \(section.todos.count)", tone: MFColor.danger)
            rows(section.todos, overdue: true)
        case .day(let day):
            let selected = model.isSelected(day)
            dayHeader(day, selected: selected)
            if section.todos.isEmpty {
                Text(selected ? "Nothing for this day yet." : "Nothing for today.")
                    .font(MFFont.caption)
                    .foregroundStyle(MFColor.tertiary)
                    .padding(.vertical, MFSpacing.s)
                    .padding(.leading, MFSpacing.xs)
            }
            rows(section.todos, overdue: false)
        case .undated:
            header("No date", tone: MFColor.tertiary)
            rows(section.todos, overdue: false)
        case .completed:
            Button { showsCompleted.toggle() } label: {
                HStack(spacing: MFSpacing.xs) {
                    Text("Completed · \(section.todos.count)")
                    Image(systemName: showsCompleted ? "chevron.down" : "chevron.right")
                }
                .font(MFFont.micro.weight(.bold))
                .foregroundStyle(MFColor.tertiary)
                .padding(.top, MFSpacing.m)
                .padding(.bottom, MFSpacing.xxs)
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .accessibilityLabel(showsCompleted ? "Hide completed to-dos" : "Show completed to-dos")
            if showsCompleted { rows(section.todos, overdue: false) }
        }
    }

    private func header(_ text: String, tone: Color) -> some View {
        Text(text.uppercased())
            .font(MFFont.micro.weight(.bold))
            .tracking(0.4)
            .foregroundStyle(tone)
            .padding(.top, MFSpacing.m)
            .padding(.bottom, MFSpacing.xxs)
            .accessibilityAddTraits(.isHeader)
    }

    private func dayHeader(_ day: Date, selected: Bool) -> some View {
        HStack(spacing: MFSpacing.xs) {
            Text(TodoDateFormat.dayName(day, today: model.today, calendar: calendar).uppercased())
                .font(MFFont.micro.weight(.bold))
                .tracking(0.4)
            Text(TodoDateFormat.short(day, calendar: calendar))
                .font(MFFont.micro)
                .monospacedDigit()
        }
        .foregroundStyle(selected ? MFColor.accentStrong : MFColor.tertiary)
        .padding(.horizontal, selected ? MFSpacing.xs : 0)
        .padding(.vertical, MFSpacing.xxs)
        .background(RoundedRectangle(cornerRadius: MFRadius.xs).fill(selected ? MFColor.accentFill : .clear))
        .padding(.top, MFSpacing.m)
        .padding(.bottom, MFSpacing.xxs)
        .accessibilityAddTraits(.isHeader)
    }

    private func rows(_ todos: [TodoItem], overdue: Bool) -> some View {
        ForEach(todos) { todo in
            TodoRow(todo: todo, overdue: overdue, calendar: calendar) { model.setDone($0, for: todo.id) }
                .contextMenu {
                    Button("Move to Today") { model.setDue(model.today, for: todo.id) }
                    Button("Move to Tomorrow") {
                        model.setDue(calendar.date(byAdding: .day, value: 1, to: model.today), for: todo.id)
                    }
                    if !model.isSelected(model.today) {
                        Button("Move to \(TodoDateFormat.short(model.selectedDate, calendar: calendar))") {
                            model.setDue(model.selectedDate, for: todo.id)
                        }
                    }
                    Button("Remove the date") { model.setDue(nil, for: todo.id) }
                    Divider()
                    Button("Delete", role: .destructive) { model.removeTodo(todo.id) }
                }
            Rectangle().fill(MFColor.hairlineFaint).frame(height: 1)
        }
    }

    // MARK: Add

    private var addBox: some View {
        HStack(spacing: MFSpacing.s) {
            AppleTextField(title: "Add a to-do", text: $newTitle)
                .font(MFFont.body)
                .focused($addFieldFocused)
                .onSubmit(add)
            dueMenu
            Button(action: add) {
                Image(systemName: "arrow.up")
                    .font(MFFont.calloutEmphasis)
                    .foregroundStyle(MFColor.onInk)
                    .padding(MFSpacing.s)
                    .background(Circle().fill(MFColor.ink))
            }
            .buttonStyle(.plain)
            .accessibilityLabel("Add to-do")
            .disabled(newTitle.trimmingCharacters(in: .whitespaces).isEmpty)
        }
        .padding(.leading, MFSpacing.l)
        .padding(.trailing, MFSpacing.xs)
        .padding(.vertical, MFSpacing.xs)
        .background(Capsule().fill(MFColor.surface))
        .overlay(Capsule().strokeBorder(addFieldFocused ? MFColor.accent : MFColor.hairlineStrong))
    }

    private var dueMenu: some View {
        let tomorrow = calendar.date(byAdding: .day, value: 1, to: model.today) ?? model.today
        return Menu {
            Button("Today · \(TodoDateFormat.short(model.today, calendar: calendar))") { dueChoice = .date(model.today) }
            Button("Tomorrow · \(TodoDateFormat.short(tomorrow, calendar: calendar))") { dueChoice = .date(tomorrow) }
            if !model.isSelected(model.today), !model.isSelected(tomorrow) {
                Button("Selected day · \(TodoDateFormat.short(model.selectedDate, calendar: calendar))") { dueChoice = nil }
            }
            Divider()
            Button("Pick a date…") {
                pickedDate = due ?? model.selectedDate
                picksDate = true
            }
            Button("No date") { dueChoice = DueChoice.none }
        } label: {
            DateTag(text: due.map { TodoDateFormat.short($0, calendar: calendar) } ?? "No date",
                    highlighted: dueChoice != nil)
        }
        .menuStyle(.button)
        .buttonStyle(.plain)
        .menuIndicator(.hidden)
        .fixedSize()
        .accessibilityLabel("Due date, \(due.map { $0.formatted(date: .long, time: .omitted) } ?? "none")")
        .popover(isPresented: $picksDate, arrowEdge: .top) {
            VStack(alignment: .trailing, spacing: MFSpacing.m) {
                DatePicker("Due date", selection: $pickedDate, displayedComponents: .date)
                    .datePickerStyle(.graphical)
                    .labelsHidden()
                Button("Use this date") {
                    dueChoice = .date(calendar.startOfDay(for: pickedDate))
                    picksDate = false
                }
                .buttonStyle(.applePrimary)
                .keyboardShortcut(.defaultAction)
            }
            .padding(MFSpacing.l)
        }
    }

    private func add() {
        guard !newTitle.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
        model.addTodo(text: newTitle, due: due, dueChosen: dueChoice != nil)
        newTitle = ""
        dueChoice = nil
    }
}

/// A date in a small rounded tag ("11/18/26").
struct DateTag: View {
    var text: String
    var highlighted = false

    var body: some View {
        Text(text)
            .font(MFFont.caption)
            .monospacedDigit()
            .foregroundStyle(highlighted ? MFColor.accentStrong : MFColor.ink)
            .padding(.horizontal, MFSpacing.s)
            .padding(.vertical, MFSpacing.xxs)
            .background(RoundedRectangle(cornerRadius: MFRadius.xs).fill(highlighted ? MFColor.accentFill : MFColor.fill))
    }
}

struct TodoRow: View {
    var todo: TodoItem
    var overdue: Bool
    var calendar: Calendar
    var setDone: (Bool) -> Void

    var body: some View {
        HStack(alignment: .top, spacing: MFSpacing.m) {
            Button { setDone(!todo.isDone) } label: {
                Image(systemName: todo.isDone ? "checkmark.circle.fill" : "circle")
                    .font(MFFont.bodyEmphasis)
                    .foregroundStyle(todo.isDone ? MFColor.success : MFColor.tertiary)
                    .frame(width: MFSpacing.l + MFSpacing.xs, height: MFSpacing.l + MFSpacing.xs)
            }
            .buttonStyle(.plain)
            .accessibilityLabel(todo.isDone ? "Mark \(todo.title) not done" : "Mark \(todo.title) done")
            VStack(alignment: .leading, spacing: MFSpacing.xxs) {
                Text(todo.title)
                    .font(MFFont.body)
                    .foregroundStyle(todo.isDone ? MFColor.tertiary : MFColor.ink)
                    .strikethrough(todo.isDone)
                if let meta {
                    HStack(spacing: MFSpacing.xs) {
                        if overdue, let due = todo.due {
                            Text(TodoDateFormat.short(due, calendar: calendar))
                                .monospacedDigit()
                                .foregroundStyle(MFColor.danger)
                        }
                        Text(meta)
                    }
                    .font(MFFont.caption)
                    .foregroundStyle(MFColor.tertiary)
                }
            }
            Spacer(minLength: 0)
        }
        .padding(.vertical, MFSpacing.s)
        .accessibilityElement(children: .contain)
    }

    /// "Reminders · Personal", "From mail · Priya Raman", "Added here".
    private var meta: String? {
        switch todo.source.kind {
        case .reminders: "Reminders · \(todo.source.label)"
        case .mail: "From mail · \(todo.source.label)"
        case .meeting: "From a meeting · \(todo.source.label)"
        case .scan: "From a scan · \(todo.source.label)"
        case .manual: todo.reminderID == nil ? "Added here" : "Added here · in Reminders"
        }
    }
}

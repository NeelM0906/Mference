import Foundation
import Observation

/// The Calendar screen's state: the day, week, month or year shown, its
/// events, proposals waiting for one click, and the dated to-do list kept in
/// step with Apple Reminders.
@MainActor @Observable
public final class CalendarWeekModel {
    public enum Access: Sendable, Hashable {
        /// Not asked yet: the screen shows the one-line reason and a button.
        case notAsked
        case granted
        /// Denied: the screen points to System Settings.
        case blocked
        /// Asked, but macOS showed no prompt: the screen points to System
        /// Settings and never shows the reason again on its own.
        case promptNotShown
    }

    public private(set) var access: Access = .notAsked
    /// Day, week, month or year; remembered across launches.
    public private(set) var mode: CalendarViewMode
    /// The day in focus: the period shown contains it, the to-do panel's
    /// date is it, and a typed to-do is dated to it.
    public private(set) var selectedDate: Date
    public private(set) var events: [CalendarEvent] = []
    public private(set) var todos: [TodoItem] = []
    public private(set) var proposals: [ProposedEvent] = []
    /// One user-facing sentence after a failed action.
    public private(set) var errorMessage: String?
    public private(set) var isSyncingReminders = false
    /// When the last Reminders sync finished.
    public private(set) var lastReminderSync: Date?
    /// The Reminders lists, read when Settings shows them.
    public private(set) var reminderLists: [ReminderList] = []
    /// Opens Settings > Connections (the "Settings" link in the to-do panel);
    /// set by the app.
    @ObservationIgnored public var openSettings: (@MainActor () -> Void)?
    @ObservationIgnored private var loadGeneration = 0
    @ObservationIgnored private var syncAgain = false

    private let baseLayout: WeekLayout
    private let service: any CalendarServicing
    private let store: PlannerStore
    private let defaults: UserDefaults
    private let now: @Sendable () -> Date
    /// Local to-do changes start a Reminders sync on their own (only when
    /// sync is on and access was already given; never a prompt).
    private let syncsAutomatically: Bool

    static let modeKey = "calendarViewMode"

    public init(service: any CalendarServicing, store: PlannerStore, layout: WeekLayout = WeekLayout(),
                defaults: UserDefaults = .standard, now: @escaping @Sendable () -> Date = { .now },
                syncsAutomatically: Bool = true) {
        self.service = service
        self.store = store
        self.baseLayout = layout
        self.defaults = defaults
        self.now = now
        self.syncsAutomatically = syncsAutomatically
        self.mode = defaults.string(forKey: Self.modeKey).flatMap(CalendarViewMode.init(rawValue:)) ?? .week
        self.selectedDate = layout.calendar.startOfDay(for: now())
        let contents = store.snapshot
        todos = contents.todos
        proposals = contents.proposals
        lastReminderSync = contents.lastReminderSync
    }

    public var calendar: Calendar { baseLayout.calendar }

    /// The hour grid's layout: one column in the day view, seven otherwise.
    public var layout: WeekLayout {
        var layout = baseLayout
        layout.days = mode == .day ? 1 : 7
        return layout
    }

    // MARK: Period

    /// The first day of the period shown: the day, its week, its month or
    /// its year.
    public var periodStart: Date {
        switch mode {
        case .day: calendar.startOfDay(for: selectedDate)
        case .week: layout.weekStart(containing: selectedDate)
        case .month: calendar.dateInterval(of: .month, for: selectedDate)?.start ?? selectedDate
        case .year: calendar.dateInterval(of: .year, for: selectedDate)?.start ?? selectedDate
        }
    }

    /// Kept for the hour grid: the first column's day.
    public var weekStart: Date { mode == .day || mode == .week ? periodStart : layout.weekStart(containing: selectedDate) }

    /// The hour grid's columns (day and week views).
    public var days: [Date] { layout.dayDates(from: weekStart) }

    /// The month view's cells: whole weeks from the one holding the 1st to
    /// the one holding the last day.
    public var monthGrid: [Date] { monthDays(containing: selectedDate) }

    /// Whole weeks covering the month that contains `date` (month view,
    /// mini months).
    public func monthDays(containing date: Date) -> [Date] {
        guard let month = calendar.dateInterval(of: .month, for: date) else { return [] }
        let first = baseLayout.weekStart(containing: month.start)
        let lastDay = calendar.date(byAdding: .day, value: -1, to: month.end) ?? month.start
        let lastWeek = baseLayout.weekStart(containing: lastDay)
        let weeks = (calendar.dateComponents([.day], from: first, to: lastWeek).day ?? 0) / 7 + 1
        return (0..<(weeks * 7)).compactMap { calendar.date(byAdding: .day, value: $0, to: first) }
    }

    /// The window loaded from Calendar for the current view.
    public var fetchInterval: DateInterval {
        switch mode {
        case .day, .week: return layout.interval(from: weekStart)
        case .month:
            let grid = monthGrid
            guard let first = grid.first, let last = grid.last,
                  let end = calendar.date(byAdding: .day, value: 1, to: last) else { return DateInterval(start: periodStart, duration: 0) }
            return DateInterval(start: first, end: end)
        case .year:
            return calendar.dateInterval(of: .year, for: selectedDate) ?? DateInterval(start: periodStart, duration: 0)
        }
    }

    public var blocks: [WeekBlock] {
        layout.blocks(weekStart: weekStart, events: events, proposals: proposalsThisWeek)
    }

    /// All-day events by column (0 = first day shown).
    public var outsideGrid: [Int: [CalendarEvent]] {
        layout.outsideGrid(weekStart: weekStart, events: events)
    }

    /// One day's events: all-day first, then by start (month view).
    public func events(on day: Date) -> [CalendarEvent] {
        let start = calendar.startOfDay(for: day)
        guard let end = calendar.date(byAdding: .day, value: 1, to: start) else { return [] }
        return events.filter { event in
            event.start == event.end ? (event.start >= start && event.start < end) : (event.start < end && event.end > start)
        }.sorted { ($0.isAllDay ? 0 : 1, $0.start, $0.title) < ($1.isAllDay ? 0 : 1, $1.start, $1.title) }
    }

    /// Days of the loaded period that have any event (year view's dots).
    public var daysWithEvents: Set<Date> {
        var days: Set<Date> = []
        for event in events {
            var day = calendar.startOfDay(for: event.start)
            let last = event.end > event.start ? event.end.addingTimeInterval(-1) : event.start
            while day <= last {
                days.insert(day)
                guard let next = calendar.date(byAdding: .day, value: 1, to: day) else { break }
                day = next
            }
        }
        return days
    }

    /// Proposals being added right now: a second click does nothing.
    public private(set) var addingProposals: Set<UUID> = []

    public var proposalsThisWeek: [ProposedEvent] {
        let interval = mode == .day || mode == .week ? layout.interval(from: weekStart) : fetchInterval
        return proposals.filter { interval.contains($0.start) }.sorted { $0.start < $1.start }
    }

    /// The header, in this Mac's language and region: "Wednesday, November
    /// 18, 2026", "November 2026" (or "November – December 2026"),
    /// "November 2026", "2026".
    public var title: String {
        switch mode {
        case .day: return format(selectedDate, template: "EEEEMMMMdy")
        case .week:
            guard let first = days.first, let last = days.last else { return "" }
            if calendar.isDate(first, equalTo: last, toGranularity: .month) { return format(first, template: "MMMMy") }
            let formatter = DateIntervalFormatter()
            formatter.calendar = calendar
            formatter.timeZone = calendar.timeZone
            formatter.dateTemplate = "MMMMy"
            return formatter.string(from: first, to: last)
        case .month: return format(selectedDate, template: "MMMMy")
        case .year: return format(selectedDate, template: "y")
        }
    }

    private func format(_ date: Date, template: String) -> String {
        let formatter = DateFormatter()
        formatter.calendar = calendar
        formatter.timeZone = calendar.timeZone
        formatter.locale = calendar.locale ?? .autoupdatingCurrent
        formatter.setLocalizedDateFormatFromTemplate(template)
        return formatter.string(from: date)
    }

    public func isToday(_ date: Date) -> Bool {
        calendar.isDate(date, inSameDayAs: now())
    }

    public func isSelected(_ date: Date) -> Bool {
        calendar.isDate(date, inSameDayAs: selectedDate)
    }

    /// Today's start, for grouping to-dos.
    public var today: Date { calendar.startOfDay(for: now()) }

    /// The current time, for the grid's now line.
    public var currentTime: Date { now() }

    // MARK: Navigation

    /// Day, Week, Month or Year (⌘1 to ⌘4); the selected day stays.
    public func show(_ mode: CalendarViewMode) async {
        guard mode != self.mode else { return }
        self.mode = mode
        defaults.set(mode.rawValue, forKey: Self.modeKey)
        await reload()
    }

    /// ‹ and ›: one day, week, month or year back or forward.
    public func step(_ offset: Int) async {
        let component: Calendar.Component = switch mode {
        case .day: .day
        case .week: .weekOfYear
        case .month: .month
        case .year: .year
        }
        selectedDate = calendar.date(byAdding: component, value: offset, to: selectedDate) ?? selectedDate
        await reload()
    }

    /// Today (⌘T), in the view shown.
    public func showToday() async {
        selectedDate = today
        await reload()
    }

    /// A day clicked in the mini month, a column header or a month cell:
    /// it becomes the selected day; the period follows if needed.
    public func select(_ date: Date) async {
        let before = fetchInterval
        selectedDate = calendar.startOfDay(for: date)
        if fetchInterval != before { await reload() }
    }

    /// Open a day (or month) in another view: a month cell's "+2 more", a
    /// date in the year view.
    public func open(_ date: Date, in mode: CalendarViewMode) async {
        selectedDate = calendar.startOfDay(for: date)
        self.mode = mode
        defaults.set(mode.rawValue, forKey: Self.modeKey)
        await reload()
    }

    /// Checks access without asking, then loads the week if allowed.
    public func load() async {
        switch await service.calendarAccess() {
        case .granted: access = .granted
        case .notDetermined: access = access == .promptNotShown ? .promptNotShown : .notAsked
        case .denied, .needsSystemSettings: access = .blocked
        }
        await reload()
    }

    /// EventKit changed (`EKEventStoreChanged`: access just granted, or
    /// events edited in Calendar): re-reads access and the week, without
    /// asking.
    public func calendarChanged() async {
        await load()
        await syncRemindersQuietly()
    }

    /// The app became active again (back from System Settings, say):
    /// re-reads access while it is missing.
    public func recheckAccess() async {
        if access != .granted { await load() }
        await syncRemindersQuietly()
    }

    /// The person clicked "Show my calendar": the broker shows the reason,
    /// then the system prompt.
    public func requestAccess() async {
        switch await service.requestCalendarAccess() {
        case .granted: access = .granted
        case .notDetermined: access = .notAsked
        case .denied, .needsSystemSettings:
            // Still undecided after asking: macOS never showed its prompt.
            access = await service.calendarAccess() == .notDetermined ? .promptNotShown : .blocked
        }
        await reload()
    }

    /// "Open System Settings" when Calendar access was denied.
    public func openCalendarSettings() async {
        await service.openSystemSettings(for: .calendar)
    }

    public func reload() async {
        guard access == .granted else { events = []; return }
        // A slower load for a week the person already left must not
        // overwrite the newer one.
        loadGeneration += 1
        let generation = loadGeneration
        let interval = fetchInterval
        do {
            let loaded = try await service.events(from: interval.start, to: interval.end)
            guard generation == loadGeneration else { return }
            events = loaded
            errorMessage = nil
        } catch let error as CalendarServiceError {
            errorMessage = error.userMessage
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    // MARK: Proposals

    /// A new suggestion (from Mail, say). Ignored if this source was already
    /// handled, so a dismissed proposal does not come back.
    public func propose(_ proposal: ProposedEvent) {
        let key = proposal.source.reference.map { "\(proposal.source.kind.rawValue):\($0)" }
        guard key.map({ !store.snapshot.handledSources.contains($0) }) ?? true else { return }
        persist {
            $0.proposals.append(proposal)
            if let key { $0.handledSources.insert(key) }
        }
    }

    /// One click: the event goes into Calendar.
    public func add(_ proposal: ProposedEvent) async {
        guard !addingProposals.contains(proposal.id), proposals.contains(where: { $0.id == proposal.id }) else { return }
        addingProposals.insert(proposal.id)
        defer { addingProposals.remove(proposal.id) }
        if access == .notAsked { await requestAccess() }
        guard access == .granted else { return }
        do {
            _ = try await service.add(proposal)
            persist { $0.proposals.removeAll { $0.id == proposal.id } }
            await reload()
        } catch let error as CalendarServiceError {
            errorMessage = error.userMessage
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    /// "New event" in the toolbar: the person typed it, so it is added at
    /// once (Calendar access asked first if needed). Returns whether it was
    /// added; `errorMessage` says why not.
    @discardableResult
    public func createEvent(title: String, start: Date, end: Date) async -> Bool {
        let trimmed = title.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, end > start else {
            errorMessage = "An event needs a title and an end after its start."
            return false
        }
        if access == .notAsked { await requestAccess() }
        guard access == .granted else { return false }
        do {
            _ = try await service.add(ProposedEvent(title: trimmed, start: start, end: end,
                                                    source: ItemSource(kind: .manual, label: "Added by you")))
            await reload()
            return true
        } catch let error as CalendarServiceError {
            errorMessage = error.userMessage
        } catch {
            errorMessage = error.localizedDescription
        }
        return false
    }

    /// Edit from the event's popover: this occurrence only.
    @discardableResult
    public func updateEvent(_ event: CalendarEvent, title: String, start: Date, end: Date) async -> Bool {
        let trimmed = title.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, end > start else {
            errorMessage = "An event needs a title and an end after its start."
            return false
        }
        do {
            _ = try await service.update(event.reference, title: trimmed, start: start, end: end)
            await reload()
            return true
        } catch let error as CalendarServiceError {
            errorMessage = error.userMessage
        } catch {
            errorMessage = error.localizedDescription
        }
        return false
    }

    /// Delete from the event's popover: this occurrence only.
    @discardableResult
    public func deleteEvent(_ event: CalendarEvent) async -> Bool {
        do {
            try await service.remove(event.reference)
            await reload()
            return true
        } catch let error as CalendarServiceError {
            errorMessage = error.userMessage
        } catch {
            errorMessage = error.localizedDescription
        }
        return false
    }

    public func dismiss(_ proposal: ProposedEvent) {
        persist { $0.proposals.removeAll { $0.id == proposal.id } }
    }

    // MARK: To-dos

    public var remindersSyncEnabled: Bool { RemindersSyncSetting.isEnabled(defaults) }

    /// A to-do typed in the panel: dated to `due` (the selected day unless
    /// the person changed the chip), or to a date written in the text
    /// ("tomorrow", "fri", "11/24"), which is then taken out of the title.
    @discardableResult
    public func addTodo(text: String, due: Date?, dueChosen: Bool = false) -> TodoItem? {
        var parsed = TodoInput.parse(text, calendar: calendar)
        if dueChosen || parsed.due == nil { parsed.due = due.map(calendar.startOfDay) }
        guard !parsed.title.isEmpty else { return nil }
        let todo = TodoItem(title: parsed.title, due: parsed.due, createdAt: now())
        addTodo(todo)
        return todo
    }

    public func addTodo(_ todo: TodoItem) {
        let key = todo.source.reference.map { "todo-\(todo.source.kind.rawValue):\($0)" }
        guard key.map({ !store.snapshot.handledSources.contains($0) }) ?? true else { return }
        persist {
            $0.todos.append(todo)
            if let key { $0.handledSources.insert(key) }
        }
        changedLocally()
    }

    public func setDone(_ done: Bool, for id: UUID) {
        persist {
            guard let index = $0.todos.firstIndex(where: { $0.id == id }) else { return }
            $0.todos[index].isDone = done
        }
        changedLocally()
    }

    /// Moves a to-do to another day (nil: no date).
    public func setDue(_ due: Date?, for id: UUID) {
        persist {
            guard let index = $0.todos.firstIndex(where: { $0.id == id }) else { return }
            $0.todos[index].due = due.map(calendar.startOfDay)
        }
        changedLocally()
    }

    /// Deletes the to-do; its reminder goes too at the next sync (a delete
    /// in one app deletes in both).
    public func removeTodo(_ id: UUID) {
        let deletesReminder = remindersSyncEnabled
        persist {
            guard let index = $0.todos.firstIndex(where: { $0.id == id }) else { return }
            if deletesReminder, let reminderID = $0.todos[index].reminderID { $0.deletedReminderIDs.insert(reminderID) }
            $0.todos.remove(at: index)
        }
        changedLocally()
    }

    /// Open first, then done ones.
    public var sortedTodos: [TodoItem] {
        todos.sorted { lhs, rhs in
            if lhs.isDone != rhs.isDone { return !lhs.isDone }
            switch (lhs.due, rhs.due) {
            case let (l?, r?) where l != r: return l < r
            case (.some, nil): return true
            case (nil, .some): return false
            default: return lhs.createdAt < rhs.createdAt
            }
        }
    }

    /// The panel's groups: Overdue, Today, each later day, No date, then
    /// Completed. A day group is shown for the selected day even when empty.
    public var todoSections: [TodoSection] {
        TodoSection.sections(todos, today: today, selected: calendar.startOfDay(for: selectedDate), calendar: calendar)
    }

    /// "Sync now": asks for Reminders access if needed (each step asks
    /// first, L9). Does nothing unless the Settings switch is on.
    public func syncReminders() async {
        await runSync(mayAsk: true)
    }

    /// Automatic sync (a local change, Reminders changed, the app came back):
    /// only when the switch is on and access was already given.
    public func syncRemindersQuietly() async {
        guard remindersSyncEnabled, await service.remindersAccess() == .granted else { return }
        await runSync(mayAsk: false)
    }

    private func changedLocally() {
        guard syncsAutomatically, remindersSyncEnabled else { return }
        Task { await syncRemindersQuietly() }
    }

    private func runSync(mayAsk: Bool) async {
        guard remindersSyncEnabled else { return }
        // One sync at a time; a request during one runs once more after it.
        guard !isSyncingReminders else { syncAgain = true; return }
        isSyncingReminders = true
        defer { isSyncingReminders = false }
        var ask = mayAsk
        repeat {
            syncAgain = false
            await syncOnce(mayAsk: ask)
            ask = false
        } while syncAgain
    }

    private func syncOnce(mayAsk: Bool) async {
        var state = await service.remindersAccess()
        if state != .granted, mayAsk { state = await service.requestRemindersAccess() }
        guard state == .granted else {
            if mayAsk {
                errorMessage = "Mference can't see Reminders. Turn it on in System Settings > Privacy & Security > Reminders."
            }
            return
        }
        do {
            // 1. To-dos deleted here: delete their reminders.
            let deletions = store.snapshot.deletedReminderIDs
            if !deletions.isEmpty {
                try await service.removeReminders(deletions.sorted())
                persist { $0.deletedReminderIDs.subtract(deletions) }
            }
            // 2. Push changes made here; read back what Reminders has.
            let pushed = Dictionary(uniqueKeysWithValues: todos.map { ($0.id, $0.reminderFields) })
            let target = RemindersSyncSetting.targetListID(defaults)
            let outcomes = try await service.syncReminders(todos, to: target)
            // 3. Open reminders in the lists the person chose. A to-do added
            // here stays whatever list its reminder went to.
            let open = try await service.openReminders(in: RemindersSyncSetting.listIDs(defaults))
            let openIDs = Set(open.map(\.id))
            let stamp = now()
            var orphaned: Set<String> = []
            persist {
                // A to-do deleted here while this sync ran: the reminder the
                // sync just made for it is deleted next, never imported back.
                let present = Set($0.todos.map(\.id))
                for (todoID, outcome) in outcomes where !present.contains(todoID) {
                    if case .synced(let reminderID, _) = outcome { orphaned.insert(reminderID) }
                }
                $0.deletedReminderIDs.formUnion(orphaned)
                var deletedThere: Set<UUID> = []
                for index in $0.todos.indices {
                    switch outcomes[$0.todos[index].id] {
                    case .synced(let reminderID, let fields)?:
                        $0.todos[index].reminderID = reminderID
                        $0.todos[index].lastSynced = fields
                        // Edits made in Reminders become the to-do's own, so
                        // the next sync does not mistake the old local values
                        // for changes and revert them. A to-do edited here
                        // during the sync keeps its edit (pushed next time).
                        if $0.todos[index].reminderFields == pushed[$0.todos[index].id] {
                            $0.todos[index].title = fields.title
                            $0.todos[index].isDone = fields.isDone
                            $0.todos[index].due = fields.due
                        }
                    case .deletedInReminders?:
                        deletedThere.insert($0.todos[index].id)
                    case nil:
                        break
                    }
                }
                // Deleted in Reminders: deleted here too. A reminder that
                // came from a list no longer shown leaves the panel (it stays
                // in Reminders).
                $0.todos.removeAll { todo in
                    deletedThere.contains(todo.id)
                        || (todo.source.kind == .reminders && !todo.isDone
                            && todo.reminderID.map { !openIDs.contains($0) } == true)
                }
                // New in Reminders: a new to-do here.
                let skipped = Set($0.todos.compactMap(\.reminderID)).union($0.deletedReminderIDs)
                for reminder in open where !skipped.contains(reminder.id) {
                    var todo = TodoItem(title: reminder.title, due: reminder.due,
                                        source: ItemSource(kind: .reminders, label: reminder.listTitle),
                                        reminderID: reminder.id, createdAt: reminder.created ?? stamp)
                    todo.lastSynced = reminder.fields
                    $0.todos.append(todo)
                }
                $0.lastReminderSync = stamp
            }
            if !orphaned.isEmpty { syncAgain = true }
            errorMessage = nil
        } catch let error as CalendarServiceError {
            errorMessage = error.userMessage
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    // MARK: Reminders lists (Settings)

    /// Reads the lists for Settings (only once access was given).
    public func loadReminderLists() async {
        guard await service.remindersAccess() == .granted else { reminderLists = []; return }
        reminderLists = (try? await service.reminderLists()) ?? []
    }

    /// The lists shown as to-dos; nil means every list.
    public var shownReminderListIDs: Set<String>? { RemindersSyncSetting.listIDs(defaults) }

    public func setShown(_ shown: Bool, list id: String) {
        var ids = shownReminderListIDs ?? Set(reminderLists.map(\.id))
        if shown { ids.insert(id) } else { ids.remove(id) }
        // Every list ticked: store "every list", so new lists show too.
        RemindersSyncSetting.setListIDs(ids == Set(reminderLists.map(\.id)) ? nil : ids, defaults)
        Task { await syncRemindersQuietly() }
    }

    /// The list new to-dos go to; nil means the default list.
    public var targetReminderListID: String? { RemindersSyncSetting.targetListID(defaults) }

    public func setTargetReminderList(_ id: String?) {
        RemindersSyncSetting.setTargetListID(id, defaults)
    }

    public func clearError() { errorMessage = nil }

    private func persist(_ change: (inout PlannerStore.Contents) -> Void) {
        do {
            try store.update(change)
        } catch {
            errorMessage = "Mference could not save your to-dos: \(error.localizedDescription)"
        }
        let contents = store.snapshot
        todos = contents.todos
        proposals = contents.proposals
        lastReminderSync = contents.lastReminderSync
    }
}

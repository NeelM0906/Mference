import Foundation
import Observation

/// What the Calendar screens read and do (Day, Month, To-dos, the proposed
/// change and the new-event sheet). The screens depend on this protocol only.
///
/// Owned by mobile-apple: the live model reads and writes Apple Calendar and
/// Reminders through EventKit. `SampleCalendarModel` (MferenceMobileSamples)
/// stands in for previews, tests and the skeleton build.
public protocol CalendarScreenModel: AnyObject, Observable {
    /// The calendar days are laid out in (the device's).
    var calendar: Calendar { get }
    var calendarMode: CalendarMode { get set }
    /// The day shown, at its start. The shell reads it for the Duo pane title.
    var selectedDay: Date { get }
    /// Months away from `selectedDay` that the month grid shows (the arrows).
    var monthOffset: Int { get set }
    var events: [PhoneEvent] { get set }
    var todos: [PhoneTodo] { get }
    /// A change proposed from mail, waiting for the person's approval.
    var suggestion: CalendarSuggestion? { get }
    var openTodoCount: Int { get }

    func events(on day: Date) -> [PhoneEvent]
    func hasEvents(on day: Date) -> Bool
    /// Selects `day` and brings the month grid back to its month;
    /// `switchToDay` also shows the Day view.
    func select(day: Date, switchToDay: Bool)
    /// Selects today and brings the month grid back to its month, even when
    /// today was already selected.
    func goToToday()
    func acceptSuggestion()
    func dismissSuggestion()
    func toggle(_ todo: PhoneTodo)
    /// Adds a to-do; a blank title adds nothing.
    func addTodo(_ title: String)
}

extension CalendarScreenModel {
    func select(day: Date) { select(day: day, switchToDay: false) }
}

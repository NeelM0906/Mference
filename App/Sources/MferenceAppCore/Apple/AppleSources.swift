import Foundation

/// Where Mail and Calendar read from and write to: this Mac's Apple Mail, EventKit and
/// planner, or (with `MFERENCE_DEMO_DATA`) the demo directory's in-memory data.
struct AppleSources {
    let calendar: any CalendarServicing
    let mail: any MailSource
    let sender: any MailSending
    let mailApp: any MailAppControlling
    let planner: PlannerStore
    /// Where the calendar model keeps its Reminders sync choices.
    let settings: UserDefaults

    /// `livePlanner` builds the person's planner; demo mode never calls it.
    static func make(demo: URL?, broker: any PermissionBroker,
                     livePlanner: () -> PlannerStore) -> AppleSources {
        guard let demo else {
            return AppleSources(calendar: EventKitService(broker: broker), mail: AppleMailSource(),
                                sender: AppleMailSender(broker: broker), mailApp: SystemMailApp(),
                                planner: livePlanner(), settings: .standard)
        }
        return AppleSources(calendar: DemoData.calendarService(in: demo), mail: DemoData.mailSource(in: demo),
                            sender: RecordingMailSender(), mailApp: FixedMailApp(running: true),
                            planner: DemoData.plannerStore(in: demo), settings: DemoData.settings)
    }
}

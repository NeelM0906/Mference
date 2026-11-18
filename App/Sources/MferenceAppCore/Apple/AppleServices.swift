import Foundation
import SwiftData

/// Every Apple-surface model the app needs, built once at launch: the Scan,
/// Calendar and Mail screens read theirs from the SwiftUI environment, and
/// Settings > Connections reads `connections`. U2's `AppServices` holds one.
@MainActor
public final class AppleServices {
    /// Titles, summaries, mail sorting, scan fields (L12).
    public let assistant: AppleModelService
    public let scan: ScanSession
    public let calendar: CalendarWeekModel
    public let mail: MailInboxModel
    public let connections: AppleConnections

    public init(assistant: AppleModelService, scan: ScanSession, calendar: CalendarWeekModel, mail: MailInboxModel,
                connections: AppleConnections) {
        self.assistant = assistant
        self.scan = scan
        self.calendar = calendar
        self.mail = mail
        self.connections = connections
    }

    /// The app's: Vision, EventKit and this Mac's Apple Mail, one calendar
    /// service and one mail source shared by the screens and Connections.
    /// Nothing is read or asked until a screen or Settings needs it.
    public static func live(engine: any EngineServicing, broker: any PermissionBroker, container: ModelContainer,
                            plannerStore: PlannerStore = .live()) -> AppleServices {
        let assistant = AppleModelService.live(engine: engine)
        let calendarService = EventKitService(broker: broker)
        let mailSource = AppleMailSource()
        let calendar = CalendarWeekModel(service: calendarService, store: plannerStore)
        let mail = MailInboxModel(source: mailSource, context: container.mainContext, assistant: assistant,
                                  drafter: MailReplyDrafter(engine: engine), sender: AppleMailSender(broker: broker),
                                  broker: broker,
                                  onEventSuggested: { [weak calendar] proposal in calendar?.propose(proposal) })
        return AppleServices(assistant: assistant,
                             scan: ScanSession(scanner: VisionScanner(), assistant: assistant,
                                               context: container.mainContext),
                             calendar: calendar, mail: mail,
                             connections: AppleConnections(calendar: calendarService, mail: mailSource, broker: broker))
    }
}

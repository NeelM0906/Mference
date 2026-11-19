import CoreGraphics
import Foundation
import SwiftData

/// Every Apple-surface model the app needs, built once at launch: the Scan,
/// Calendar and Mail screens read theirs from the SwiftUI environment, and
/// Settings > Connections reads `connections`. U2's `AppServices` holds one.
@MainActor
public final class AppleServices {
    /// Titles, summaries, event suggestions, scan fields (L12).
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
        let mail = MailInboxModel(source: mailSource, assistant: assistant,
                                  drafter: MailReplyDrafter(engine: engine), sender: AppleMailSender(broker: broker),
                                  broker: broker,
                                  onEventSuggested: { [weak calendar] proposal in calendar?.propose(proposal) })
        return AppleServices(assistant: assistant,
                             scan: ScanSession(scanner: VisionScanner(), assistant: assistant,
                                               context: container.mainContext),
                             calendar: calendar, mail: mail,
                             connections: AppleConnections(calendar: calendarService, mail: mailSource, broker: broker))
    }

    /// Previews, snapshots and tests: in-memory calendar, mail and planner, a
    /// scanner that reads nothing, Apple's model reported off (Quick through
    /// `engine`, a fake one in tests), no prompts, no Mail store, no EventKit,
    /// no network.
    public static func preview(engine: any EngineServicing, container: ModelContainer,
                               calendar calendarService: any CalendarServicing = InMemoryCalendarService(
                                   calendarAccess: .notDetermined),
                               mail mailSource: any MailSource = InMemoryMailSource()) -> AppleServices {
        let broker = PreviewPermissionBroker()
        let assistant = AppleModelService(
            primary: ScriptedAssistantTextModel(kind: .appleOnDevice,
                                                availability: .unavailable(reason: "Apple's on-device model is off in previews."),
                                                answer: ""),
            fallback: QuickTextModel(engine: engine))
        let calendar = CalendarWeekModel(service: calendarService, store: PlannerStore(url: nil))
        let mail = MailInboxModel(source: mailSource, assistant: assistant,
                                  drafter: MailReplyDrafter(engine: engine), sender: RecordingMailSender(),
                                  broker: broker, mailApp: FixedMailApp(),
                                  onEventSuggested: { [weak calendar] proposal in calendar?.propose(proposal) })
        return AppleServices(assistant: assistant,
                             scan: ScanSession(scanner: PreviewScanner(), assistant: assistant,
                                               context: container.mainContext,
                                               imageDirectory: URL.temporaryDirectory.appending(path: "mference-preview-scans")),
                             calendar: calendar, mail: mail,
                             connections: AppleConnections(calendar: calendarService, mail: mailSource, broker: broker))
    }
}

/// Reads nothing: previews show Scan's empty and failed states only.
public struct PreviewScanner: DocumentRecognizing {
    public init() {}
    public func recognize(imageAt url: URL) async throws -> ScanReading { throw ScanError.noTextFound }
    public func recognize(image: CGImage) async throws -> ScanReading { throw ScanError.noTextFound }
}

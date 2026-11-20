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
    #if os(macOS)
    /// Apple Mail is read on the Mac only.
    public let mail: MailInboxModel
    #endif
    public let connections: AppleConnections

    #if os(macOS)
    public init(assistant: AppleModelService, scan: ScanSession, calendar: CalendarWeekModel, mail: MailInboxModel,
                connections: AppleConnections) {
        self.assistant = assistant
        self.scan = scan
        self.calendar = calendar
        self.mail = mail
        self.connections = connections
    }
    #else
    public init(assistant: AppleModelService, scan: ScanSession, calendar: CalendarWeekModel,
                connections: AppleConnections) {
        self.assistant = assistant
        self.scan = scan
        self.calendar = calendar
        self.connections = connections
    }
    #endif

    /// The app's: Vision, EventKit and this Mac's Apple Mail, one calendar
    /// service and one mail source shared by the screens and Connections.
    /// Nothing is read or asked until a screen or Settings needs it.
    /// `useAppleModel` reads Settings > Models > Apple Intelligence model.
    public static func live(engine: any EngineServicing, broker: any PermissionBroker, container: ModelContainer,
                            plannerStore: PlannerStore = .live(),
                            useAppleModel: @escaping @Sendable () async -> Bool) -> AppleServices {
        let assistant = AppleModelService.live(engine: engine, useAppleModel: useAppleModel)
        let calendarService = EventKitService(broker: broker)
        let calendar = CalendarWeekModel(service: calendarService, store: plannerStore)
        let scan = ScanSession(scanner: VisionScanner(), assistant: assistant, context: container.mainContext)
        #if os(macOS)
        let mailSource = AppleMailSource()
        let mail = MailInboxModel(source: mailSource, assistant: assistant,
                                  drafter: MailReplyDrafter(engine: engine), sender: AppleMailSender(broker: broker),
                                  broker: broker,
                                  onEventSuggested: { [weak calendar] proposal in calendar?.propose(proposal) })
        return AppleServices(assistant: assistant, scan: scan, calendar: calendar, mail: mail,
                             connections: AppleConnections(calendar: calendarService, mail: mailSource, broker: broker))
        #else
        return AppleServices(assistant: assistant, scan: scan, calendar: calendar,
                             connections: AppleConnections(calendar: calendarService, broker: broker))
        #endif
    }

    /// Previews, snapshots and tests: in-memory calendar, mail and planner, a
    /// scanner that reads nothing, Apple's model reported off (Quick through
    /// `engine`, a fake one in tests), no prompts, no Mail store, no EventKit,
    /// no network.
    #if os(macOS)
    public static func preview(engine: any EngineServicing, container: ModelContainer,
                               calendar calendarService: any CalendarServicing = InMemoryCalendarService(
                                   calendarAccess: .notDetermined),
                               mail mailSource: any MailSource = InMemoryMailSource()) -> AppleServices {
        let broker = PreviewPermissionBroker()
        let (assistant, scan, calendar) = previewParts(engine: engine, container: container, calendar: calendarService)
        let mail = MailInboxModel(source: mailSource, assistant: assistant,
                                  drafter: MailReplyDrafter(engine: engine), sender: RecordingMailSender(),
                                  broker: broker, mailApp: FixedMailApp(),
                                  onEventSuggested: { [weak calendar] proposal in calendar?.propose(proposal) })
        return AppleServices(assistant: assistant, scan: scan, calendar: calendar, mail: mail,
                             connections: AppleConnections(calendar: calendarService, mail: mailSource, broker: broker))
    }
    #else
    public static func preview(engine: any EngineServicing, container: ModelContainer,
                               calendar calendarService: any CalendarServicing = InMemoryCalendarService(
                                   calendarAccess: .notDetermined)) -> AppleServices {
        let (assistant, scan, calendar) = previewParts(engine: engine, container: container, calendar: calendarService)
        return AppleServices(assistant: assistant, scan: scan, calendar: calendar,
                             connections: AppleConnections(calendar: calendarService, broker: PreviewPermissionBroker()))
    }
    #endif

    private static func previewParts(engine: any EngineServicing, container: ModelContainer,
                                     calendar calendarService: any CalendarServicing)
        -> (AppleModelService, ScanSession, CalendarWeekModel) {
        let assistant = AppleModelService(
            primary: ScriptedAssistantTextModel(kind: .appleOnDevice,
                                                availability: .unavailable(reason: "Apple's on-device model is off in previews."),
                                                answer: ""),
            fallback: QuickTextModel(engine: engine))
        let scan = ScanSession(scanner: PreviewScanner(), assistant: assistant, context: container.mainContext,
                               imageDirectory: URL.temporaryDirectory.appending(path: "mference-preview-scans"))
        let calendar = CalendarWeekModel(service: calendarService, store: PlannerStore(url: nil))
        return (assistant, scan, calendar)
    }
}

/// Reads nothing: previews show Scan's empty and failed states only.
public struct PreviewScanner: DocumentRecognizing {
    public init() {}
    public func recognize(imageAt url: URL) async throws -> ScanReading { throw ScanError.noTextFound }
    public func recognize(image: CGImage) async throws -> ScanReading { throw ScanError.noTextFound }
}

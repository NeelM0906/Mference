import Foundation

/// Diagnostic only, for README and site captures: with `MFERENCE_DEMO_DATA=<dir>` set, Mail
/// reads `<dir>/mail/*.eml` (and `<dir>/mail/sent/*.eml`) and Calendar reads
/// `<dir>/calendar/*.ics`, both in memory, instead of this Mac's Apple Mail and EventKit; the
/// planner lives in `<dir>/Planner.json` and starts with the to-dos in `<dir>/calendar/todos.tsv`. Nothing is written back
/// to Mail or Calendar and nothing is sent. The mail runs through the same parser, sorter and
/// reply drafter as real mail. A file named `*-seen.eml` is read; others are unread. Not a
/// supported control.
public enum DemoData {
    public static let variable = "MFERENCE_DEMO_DATA"

    /// The demo directory whenever the variable is set. A set but missing path fails closed:
    /// demo mode stays on with no demo data, and never falls back to this Mac's Mail and Calendar.
    public static var directory: URL? { directory(environment: ProcessInfo.processInfo.environment) }

    static func directory(environment: [String: String]) -> URL? {
        guard let path = environment[variable], !path.isEmpty else { return nil }
        var isDirectory: ObjCBool = false
        if !FileManager.default.fileExists(atPath: path, isDirectory: &isDirectory) || !isDirectory.boolValue {
            FileHandle.standardError.write(Data("\(variable): \(path) is not a directory; demo mode shows no mail or events\n".utf8))
        }
        return URL(fileURLWithPath: path, isDirectory: true)
    }

    /// Where the demo planner keeps its to-dos and proposals: inside the demo directory, never
    /// the person's own Planner.json.
    static func plannerURL(in directory: URL) -> URL {
        directory.appending(path: "Planner.json")
    }

    static let accountID = "demo"

    // MARK: Mail

    public static func mailSource(in directory: URL) -> InMemoryMailSource {
        let mailDirectory = directory.appending(path: "mail", directoryHint: .isDirectory)
        var summaries: [MailMessageSummary] = []
        var bodies: [String: MailBody] = [:]
        var owner: String?
        let folders: [(URL, String, Bool)] = [
            (mailDirectory, "INBOX", false),
            (mailDirectory.appending(path: "sent", directoryHint: .isDirectory), "Sent Messages", true),
        ]
        for (folder, mailboxName, outgoing) in folders {
            let mailbox = MailboxRef(url: "imap://\(accountID)/\(mailboxName.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? mailboxName)")
            for file in emlFiles(in: folder) {
                guard let data = try? Data(contentsOf: file), let body = MailMessageParser.parseMessage(data),
                      let message = summary(of: body, file: file, rowID: Int64(summaries.count + 1),
                                            mailbox: mailbox, outgoing: outgoing) else { continue }
                if !outgoing, owner == nil { owner = parseAddress(body.headers["to"] ?? "").address }
                if outgoing, owner == nil { owner = parseAddress(body.headers["from"] ?? "").address }
                summaries.append(message)
                bodies[message.messageID] = body
            }
        }
        let address = owner ?? "me@demo.example"
        return InMemoryMailSource(messages: summaries, bodies: bodies,
                                  accounts: [MailAccount(id: accountID, name: address, address: address)])
    }

    static func emlFiles(in folder: URL) -> [URL] {
        ((try? FileManager.default.contentsOfDirectory(at: folder, includingPropertiesForKeys: nil)) ?? [])
            .filter { $0.pathExtension.lowercased() == "eml" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
    }

    static func summary(of body: MailBody, file: URL, rowID: Int64, mailbox: MailboxRef,
                        outgoing: Bool) -> MailMessageSummary? {
        let headers = body.headers
        guard let dateText = headers["date"], let date = parseDate(dateText) else { return nil }
        let sender = parseAddress(MailMessageParser.decodeHeader(headers["from"] ?? ""))
        let recipient = parseAddress(MailMessageParser.decodeHeader(headers["to"] ?? ""))
        let snippet = String(body.text.split(whereSeparator: \.isNewline).joined(separator: " ").prefix(200))
        return MailMessageSummary(
            rowID: rowID,
            messageID: headers["message-id"] ?? "demo-\(file.lastPathComponent)",
            senderName: sender.name, senderAddress: sender.address,
            subject: MailMessageParser.decodeHeader(headers["subject"] ?? ""),
            snippet: snippet, date: date,
            isRead: outgoing || file.deletingPathExtension().lastPathComponent.hasSuffix("-seen"),
            mailbox: mailbox,
            recipients: outgoing ? [MailParticipant(name: recipient.name, address: recipient.address)] : [],
            isInSentOrDrafts: outgoing)
    }

    /// `"Priya Raman" <priya@x.example>`, `Priya Raman <priya@x.example>` or a bare address.
    static func parseAddress(_ value: String) -> (name: String, address: String) {
        let trimmed = value.trimmingCharacters(in: .whitespaces)
        guard let open = trimmed.lastIndex(of: "<"), let close = trimmed.lastIndex(of: ">"), open < close else {
            return ("", trimmed)
        }
        let address = String(trimmed[trimmed.index(after: open)..<close])
        let name = trimmed[..<open].trimmingCharacters(in: .whitespaces)
            .trimmingCharacters(in: CharacterSet(charactersIn: "\""))
        return (name, address)
    }

    static func parseDate(_ text: String) -> Date? {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        for format in ["EEE, dd MMM yyyy HH:mm:ss Z", "dd MMM yyyy HH:mm:ss Z", "EEE, d MMM yyyy HH:mm:ss Z"] {
            formatter.dateFormat = format
            if let date = formatter.date(from: text.trimmingCharacters(in: .whitespaces)) { return date }
        }
        return nil
    }

    // MARK: Calendar

    public static func calendarService(in directory: URL) -> InMemoryCalendarService {
        let folder = directory.appending(path: "calendar", directoryHint: .isDirectory)
        let files = ((try? FileManager.default.contentsOfDirectory(at: folder, includingPropertiesForKeys: nil)) ?? [])
            .filter { $0.pathExtension.lowercased() == "ics" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        let events = files.flatMap { parseICS((try? String(contentsOf: $0, encoding: .utf8)) ?? "") }
        return InMemoryCalendarService(events: events, calendarAccess: .granted, remindersAccess: .granted)
    }

    /// The VEVENTs of a calendar file: SUMMARY, LOCATION, and floating or UTC DTSTART/DTEND.
    static func parseICS(_ text: String, timeZone: TimeZone = .current) -> [CalendarEvent] {
        var events: [CalendarEvent] = []
        var fields: [String: String]?
        for rawLine in text.replacingOccurrences(of: "\r\n", with: "\n").split(separator: "\n") {
            let line = String(rawLine)
            if line == "BEGIN:VEVENT" { fields = [:]; continue }
            if line == "END:VEVENT" {
                if let f = fields, let title = f["SUMMARY"],
                   let start = f["DTSTART"].flatMap({ parseICSDate($0, timeZone: timeZone) }),
                   let end = f["DTEND"].flatMap({ parseICSDate($0, timeZone: timeZone) }) {
                    events.append(CalendarEvent(id: f["UID"] ?? "demo-\(events.count)", title: title,
                                                start: start, end: end, calendarName: "Demo",
                                                location: f["LOCATION"]))
                }
                fields = nil
                continue
            }
            guard fields != nil, let colon = line.firstIndex(of: ":") else { continue }
            let key = line[..<colon].split(separator: ";").first.map(String.init) ?? ""
            fields?[key] = String(line[line.index(after: colon)...])
        }
        return events
    }

    static func parseICSDate(_ value: String, timeZone: TimeZone) -> Date? {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        if value.hasSuffix("Z") {
            formatter.timeZone = TimeZone(identifier: "UTC")
            formatter.dateFormat = "yyyyMMdd'T'HHmmss'Z'"
        } else {
            formatter.timeZone = timeZone
            formatter.dateFormat = "yyyyMMdd'T'HHmmss"
        }
        return formatter.date(from: value)
    }

    // MARK: To-dos

    /// The demo planner (`<dir>/Planner.json`); when it has no file yet it starts with the demo to-dos
    /// (`yyyy-MM-dd<TAB>title` lines).
    public static func plannerStore(in directory: URL, at storeURL: URL? = nil) -> PlannerStore {
        let url = directory.appending(path: "calendar/todos.tsv")
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "yyyy-MM-dd HH:mm"
        let todos: [TodoItem] = ((try? String(contentsOf: url, encoding: .utf8)) ?? "")
            .split(separator: "\n").compactMap { line in
                let parts = line.split(separator: "\t", maxSplits: 1).map(String.init)
                guard parts.count == 2, let due = formatter.date(from: parts[0] + " 09:00") else { return nil }
                return TodoItem(title: parts[1], due: due, createdAt: due.addingTimeInterval(-3 * 86_400))
            }
        return PlannerStore(url: storeURL ?? plannerURL(in: directory), initial: PlannerStore.Contents(todos: todos))
    }
}

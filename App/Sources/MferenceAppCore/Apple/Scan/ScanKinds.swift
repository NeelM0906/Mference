import Foundation

/// What a scan is, which decides what Scan offers after the full text.
public enum ScanKind: String, Sendable, Hashable, CaseIterable, Identifiable, Codable {
    /// Any other text: copy it, save it to Notes or export it.
    case document
    /// Today's receipt fields, Add row and CSV.
    case receipt
    /// A flyer or invitation: an event card for Calendar.
    case event
    /// A business card: a contact card for Contacts.
    case contact
    /// A table: exported as CSV.
    case table

    public var id: String { rawValue }

    public var label: String {
        switch self {
        case .document: "Document"
        case .receipt: "Receipt"
        case .event: "Event"
        case .contact: "Contact"
        case .table: "Table"
        }
    }
}

/// A business card's fields, for the person to check before Contacts.
public struct ContactDraft: Sendable, Hashable, Codable {
    public var name: String
    public var organization: String
    public var jobTitle: String
    public var phones: [String]
    public var emails: [String]
    public var address: String
    public var website: String

    public init(name: String = "", organization: String = "", jobTitle: String = "", phones: [String] = [],
                emails: [String] = [], address: String = "", website: String = "") {
        self.name = name
        self.organization = organization
        self.jobTitle = jobTitle
        self.phones = phones
        self.emails = emails
        self.address = address
        self.website = website
    }

    public var isEmpty: Bool {
        name.isEmpty && organization.isEmpty && phones.isEmpty && emails.isEmpty
    }

    /// Reads the JSON object the contact prompt asks for (name, organization
    /// and title only; numbers and addresses come from the detectors).
    init(json object: [String: Any]) {
        let string = { (key: String) in AssistantAnswerParser.string(object[key]) ?? "" }
        self.init(name: string("name"), organization: string("organization"), jobTitle: string("title"))
    }

    /// A vCard 3.0 Contacts opens to add the person (it shows its own
    /// confirmation first). Values are escaped, so a card's text cannot add
    /// fields of its own.
    public var vCard: String {
        func escape(_ value: String) -> String {
            value.replacingOccurrences(of: "\\", with: "\\\\")
                .replacingOccurrences(of: "\r\n", with: "\\n").replacingOccurrences(of: "\n", with: "\\n")
                .replacingOccurrences(of: "\r", with: "\\n")
                .replacingOccurrences(of: ";", with: "\\;").replacingOccurrences(of: ",", with: "\\,")
        }
        let parts = name.split(separator: " ").map(String.init)
        let family = parts.count > 1 ? parts.last ?? "" : ""
        let given = parts.count > 1 ? parts.dropLast().joined(separator: " ") : (parts.first ?? "")
        var lines = ["BEGIN:VCARD", "VERSION:3.0",
                     "N:\(escape(family));\(escape(given));;;",
                     "FN:\(escape(name.isEmpty ? organization : name))"]
        if !organization.isEmpty { lines.append("ORG:\(escape(organization))") }
        if !jobTitle.isEmpty { lines.append("TITLE:\(escape(jobTitle))") }
        lines += phones.filter { !$0.isEmpty }.map { "TEL;TYPE=WORK:\(escape($0))" }
        lines += emails.filter { !$0.isEmpty }.map { "EMAIL;TYPE=INTERNET:\(escape($0))" }
        if !address.isEmpty { lines.append("ADR;TYPE=WORK:;;\(escape(address));;;;") }
        if !website.isEmpty { lines.append("URL:\(escape(website))") }
        lines.append("END:VCARD")
        return lines.joined(separator: "\r\n") + "\r\n"
    }

    /// What the detectors and the printed lines say, without a model: the
    /// numbers, addresses and links, a name-like line, a company-like line
    /// and a job-title line.
    static func read(_ reading: ScanReading, detected: [ScanDetected]) -> ContactDraft {
        var draft = ContactDraft()
        for item in detected {
            switch item {
            case .phone(let phone): if !draft.phones.contains(phone) { draft.phones.append(phone) }
            case .email(let email): if !draft.emails.contains(email) { draft.emails.append(email) }
            case .address(let address): if draft.address.isEmpty { draft.address = address }
            case .link(let link): if draft.website.isEmpty { draft.website = link }
            case .date: break
            }
        }
        let lines = reading.rows.map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }
        let used = Set(draft.phones + draft.emails + [draft.address, draft.website])
        let plain = lines.filter { line in !used.contains { !$0.isEmpty && line.contains($0) } }
        draft.jobTitle = plain.first { line in jobWords.contains { containsWord($0, in: line) } } ?? ""
        draft.organization = plain.first { line in
            line != draft.jobTitle && companyWords.contains { containsWord($0, in: line) }
        } ?? ""
        draft.name = plain.first { line in
            guard line != draft.jobTitle, line != draft.organization, !line.contains(where: \.isNumber),
                  !line.contains("@") else { return false }
            let words = line.split(separator: " ")
            return (2...4).contains(words.count) && words.allSatisfy { $0.first?.isUppercase == true }
        } ?? ""
        return draft
    }

    /// `word` stands alone in `line` (any case; "inc." and "co-founder" too).
    static func containsWord(_ word: String, in line: String) -> Bool {
        let pattern = "(?<![a-z])" + NSRegularExpression.escapedPattern(for: word) + "(?![a-z])"
        return line.lowercased().range(of: pattern, options: .regularExpression) != nil
    }

    static let jobWords = ["ceo", "cto", "cfo", "coo", "founder", "co-founder", "president", "director", "manager",
                           "engineer", "designer", "developer", "consultant", "partner", "officer", "head",
                           "lead", "vp", "vice president", "associate", "analyst", "architect", "attorney",
                           "specialist", "coordinator", "executive", "owner", "principal", "producer", "editor"]
    static let companyWords = ["inc", "inc.", "llc", "ltd", "ltd.", "gmbh", "co.", "corp", "corporation", "company",
                               "studio", "studios", "group", "labs", "partners", "associates", "agency", "bank",
                               "university", "clinic", "llp", "plc", "s.a.", "ag"]
}

/// An event from a flyer or invitation, for the person to check before it
/// goes to Calendar's proposals.
public struct ScanEventDraft: Sendable, Hashable {
    public var title: String
    public var start: Date
    public var end: Date

    public init(title: String, start: Date, end: Date) {
        self.title = title
        self.start = start
        self.end = max(end, start)
    }

    /// The first date with a time of day the text names, an hour long
    /// unless it gives an end; titled by the heading or the first line.
    static func read(_ reading: ScanReading, detected: [ScanDetected], title: String) -> ScanEventDraft? {
        for item in detected {
            guard case .date(let start, let end, let hasTime) = item, hasTime else { continue }
            let finish = end.flatMap { $0 > start ? $0 : nil } ?? start.addingTimeInterval(3_600)
            return ScanEventDraft(title: title, start: start, end: finish)
        }
        return nil
    }
}

/// Decides what a scan is from the text alone, so a clear case needs no
/// model: a printed total (receipt), a real table, a short card with a
/// phone or email (contact), a short text naming a date and time (event).
/// Anything else, or more than one match, is unsure: a model may decide,
/// and the person can always choose with "Read as".
public enum ScanClassifier {
    public struct Verdict: Sendable, Hashable {
        public var kind: ScanKind
        /// False when no rule or more than one matched.
        public var isSure: Bool
    }

    static func classify(_ reading: ScanReading, receipt: ReceiptDraft, detected: [ScanDetected]) -> Verdict {
        // A printed total is the receipt rule the receipt path always used.
        if receipt.total != nil { return Verdict(kind: .receipt, isSure: true) }
        let rows = reading.rows.filter { !$0.trimmingCharacters(in: .whitespaces).isEmpty }
        let characters = reading.fullText.count
        var matches: [ScanKind] = []
        if reading.tables.contains(where: { $0.rows.count >= 2 && $0.columnCount >= 2 }) { matches.append(.table) }
        let reachable = detected.contains { item in
            switch item {
            case .phone, .email: true
            default: false
            }
        }
        if reachable, rows.count <= 12, characters <= 400 { matches.append(.contact) }
        let timed = detected.contains { item in
            if case .date(_, _, true) = item { return true }
            return false
        }
        if timed, rows.count <= 25, characters <= 900 { matches.append(.event) }
        guard let first = matches.first else { return Verdict(kind: .document, isSure: false) }
        return Verdict(kind: first, isSure: matches.count == 1)
    }

    /// The title a document is shown under: Vision's title line, else the
    /// first line, kept short.
    static func title(of reading: ScanReading) -> String {
        let candidate = reading.heading?.isEmpty == false ? reading.heading
            : reading.rows.first { !$0.trimmingCharacters(in: .whitespaces).isEmpty }
        let line = AssistantPrompts.oneLine(candidate ?? "", limit: 60).trimmingCharacters(in: .whitespaces)
        return line.isEmpty ? "Scanned document" : line
    }
}

/// A detected table as CSV (formulas defused like the receipt export).
public enum ScanTableExporter {
    public static func csv(_ table: ScanTable) -> String {
        let width = table.columnCount
        return table.rows.map { row in
            (row + Array(repeating: "", count: max(0, width - row.count))).map(ScanCSVExporter.cell)
                .joined(separator: ",")
        }.joined(separator: "\r\n") + "\r\n"
    }

    public static func fileData(_ table: ScanTable) -> Data {
        Data([0xEF, 0xBB, 0xBF]) + Data(csv(table).utf8)
    }
}

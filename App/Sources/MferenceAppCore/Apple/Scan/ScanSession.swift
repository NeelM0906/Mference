import AppKit
import Foundation
import Observation
import SwiftData

/// The Scan screen's state: read an image or PDF, show its full text, tell
/// what it is, and offer what fits. A receipt keeps its fields, Add row and
/// CSV (the only scans saved, since only receipts have a place to be seen
/// again); an event goes to Calendar's proposals, a business card to
/// Contacts, a table to CSV, and any text can be copied, saved to Notes or
/// exported. Clear a scan at any step.
@MainActor @Observable
public final class ScanSession {
    public enum Phase: Sendable, Hashable {
        /// Nothing opened yet.
        case empty
        /// Vision is reading, then the kind's fields are filled.
        case reading
        /// The text and the kind's fields are shown for the person to check.
        case review
        /// The person confirmed a receipt; it is saved.
        case saved
        /// `message` is user-facing.
        case failed(message: String)
    }

    public private(set) var phase: Phase = .empty
    /// The image being reviewed.
    public private(set) var imageURL: URL?
    public private(set) var reading: ScanReading?
    /// What the person sees and edits.
    public var fields = ScanFields(fields: [])
    /// The id of the saved `ScanRecord` once confirmed.
    public private(set) var savedRecordID: UUID?
    /// What the scan is; "Read as" changes it.
    public private(set) var kind: ScanKind = .document
    /// A "Read as" choice is being prepared (fields filled for that kind).
    public private(set) var isPreparing = false
    /// A business card's fields, for the person to check.
    public var contact = ContactDraft()
    /// A flyer's event, for the person to check; nil when none was found.
    public var event: ScanEventDraft?
    /// The page the screen shows, from 0.
    public private(set) var page = 0
    /// What the last action did ("Saved to Notes."); user-facing.
    public private(set) var notice: String?
    /// Saves text as a new note (title, text); set by the app once Notes
    /// exists. Nil hides Save to Notes.
    @ObservationIgnored public var saveToNotes: (@MainActor (String, String) throws -> Void)?

    private let scanner: any DocumentRecognizing
    private let assistant: any AppleModelServicing
    private let context: ModelContext
    private let imageDirectory: URL
    private var readTask: Task<Void, Never>?
    /// Every read started and not yet awaited, so tests can wait for a
    /// replaced read to finish too.
    /// A read leaves the list when it finishes, so the list stays small.
    @ObservationIgnored private var pendingReads: [(id: UUID, task: Task<Void, Never>)] = []
    @ObservationIgnored private var pastedDirectory: URL?
    /// The private folder of the contact card handed to Contacts; Contacts
    /// reads it after `open` returns, so it stays until the next scan.
    @ObservationIgnored private var contactDirectory: URL?
    @ObservationIgnored private var documentTitle = "Scanned document"
    private let onEventSuggested: @MainActor (ProposedEvent) -> Void
    private let openFile: @MainActor (URL) -> Void
    private let now: @MainActor () -> Date
    private let timeZone: TimeZone

    /// `imageDirectory` keeps copies of saved receipts' images (default: the
    /// app's Application Support folder, `Scans/`). `onEventSuggested` hands
    /// an event to Calendar's proposals; `openFile` opens the contact card
    /// in Contacts.
    public init(scanner: any DocumentRecognizing, assistant: any AppleModelServicing, context: ModelContext,
                imageDirectory: URL = AppStorageLocation.applicationSupport.appending(path: "Scans",
                                                                                      directoryHint: .isDirectory),
                timeZone: TimeZone = .current, now: @escaping @MainActor () -> Date = { .now },
                onEventSuggested: @escaping @MainActor (ProposedEvent) -> Void = { _ in },
                openFile: @escaping @MainActor (URL) -> Void = { NSWorkspace.shared.open($0) }) {
        self.scanner = scanner
        self.assistant = assistant
        self.context = context
        self.imageDirectory = imageDirectory
        self.timeZone = timeZone
        self.now = now
        self.onEventSuggested = onEventSuggested
        self.openFile = openFile
    }

    /// "Receipt from Corner Hardware" ("Receipt" before the merchant is
    /// known), a card's name, an event's title, else the document's first
    /// line.
    public var title: String {
        switch kind {
        case .receipt: fields.value(.merchant).map { "Receipt from \($0)" } ?? "Receipt"
        case .contact: contact.name.isEmpty ? (contact.organization.isEmpty ? "Business card" : contact.organization)
            : contact.name
        case .event: event?.title.isEmpty == false ? event?.title ?? documentTitle : documentTitle
        case .document, .table: documentTitle
        }
    }

    /// Everything that was read, page by page, for reading, copying, Notes
    /// and .txt.
    public var fullText: String { reading?.fullText ?? "" }

    public var pageCount: Int { reading?.pageCount ?? 1 }

    public var tables: [ScanTable] { reading?.tables ?? [] }

    /// Boxes around the receipt fields on the page shown.
    public var highlights: [ScanFieldKey: CGRect] {
        guard kind == .receipt, let reading, phase == .review || phase == .saved else { return [:] }
        return ScanHighlights.boxes(fields: fields, reading: reading, page: page)
    }

    /// Shows another page of a PDF.
    public func showPage(_ index: Int) {
        page = min(max(index, 0), pageCount - 1)
    }

    /// Who filled the fields, for the line under the title.
    public var filledByLabel: String? { fields.filledBy?.label }

    /// Image types Scan accepts from Open file and drag-drop.
    public nonisolated static let acceptedExtensions: Set<String> = ["png", "jpg", "jpeg", "heic", "heif", "tif", "tiff", "pdf",
                                                         "gif", "bmp", "webp"]

    public nonisolated static func accepts(_ url: URL) -> Bool {
        acceptedExtensions.contains(url.pathExtension.lowercased())
    }

    /// Starts reading an image file (Open file or drag-drop); replaces any scan
    /// being reviewed.
    public func open(_ url: URL) {
        readTask?.cancel()
        imageURL = url
        clearResults()
        phase = .reading
        let id = UUID()
        let task = Task { [weak self] in
            await self?.read(url)
            self?.pendingReads.removeAll { $0.id == id }
        }
        readTask = task
        pendingReads.append((id, task))
    }

    /// An image pasted from the clipboard (PNG or TIFF data): kept as a file
    /// in a private temporary folder, then read like an opened file.
    public func openPasted(_ data: Data, fileExtension: String = "png") throws {
        let directory = pastedDirectory ?? URL.temporaryDirectory.appending(path: "mference-pasted-\(UUID().uuidString)",
                                                                            directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true,
                                                attributes: [.posixPermissions: 0o700])
        pastedDirectory = directory
        let url = directory.appending(path: "Pasted image \(UUID().uuidString.prefix(8)).\(fileExtension)")
        try data.write(to: url, options: .atomic)
        open(url)
    }

    /// Waits for every read started so far, replaced ones included (tests).
    public func waitForReading() async {
        while let first = pendingReads.first {
            await first.task.value
            pendingReads.removeAll { $0.id == first.id }
        }
    }

    private func read(_ url: URL) async {
        do {
            let reading = try await scanner.recognize(imageAt: url)
            try Task.checkCancellation()
            let printed = ReceiptHeuristics.read(reading)
            let detected = reading.detectedData(reference: now())
            // The text decides when it is clear; Apple's model only when it
            // is not (never Quick, so a scan never swaps out the chat model).
            let verdict = ScanClassifier.classify(reading, receipt: printed, detected: detected)
            var kind = verdict.kind
            if !verdict.isSure, let result = try? await assistant.classifyScan(reading.modelText, allowFallback: false) {
                kind = result.value
            }
            try Task.checkCancellation()
            try await prepare(kind, reading: reading, printed: printed, detected: detected)
            phase = .review
        } catch {
            // A replaced read may still fail after the new one finished:
            // only the current read may change the screen.
            guard !Task.isCancelled, !(error is CancellationError) else { return }
            let scanError = error as? ScanError ?? .recognitionFailed(error.localizedDescription)
            phase = .failed(message: scanError.userMessage)
        }
    }

    /// Fills what `kind` shows. Receipts work as they always did (the
    /// receipt model, with Quick when Apple's is off); the other kinds use
    /// what the text and detectors found, asking Apple's model only for
    /// what they miss.
    private func prepare(_ kind: ScanKind, reading: ScanReading, printed: ReceiptDraft,
                         detected: [ScanDetected]) async throws {
        documentTitle = ScanClassifier.title(of: reading)
        var fields = ScanFields(fields: [])
        var contact = ContactDraft()
        var event: ScanEventDraft?
        switch kind {
        case .receipt:
            var model: ReceiptDraft?
            var filledBy: AssistantModelKind?
            if let result = try? await assistant.fillReceipt(from: reading.modelText) {
                model = result.value
                filledBy = result.model
            }
            fields = ReceiptFieldMerger.merge(model: model, text: printed, reading: reading, filledBy: filledBy)
        case .contact:
            contact = ContactDraft.read(reading, detected: detected)
            if contact.name.isEmpty || contact.organization.isEmpty || contact.jobTitle.isEmpty,
               let result = try? await assistant.fillContact(from: reading.modelText, allowFallback: false) {
                // The printed lines win; the model only fills blanks.
                if contact.name.isEmpty { contact.name = result.value.name }
                if contact.organization.isEmpty { contact.organization = result.value.organization }
                if contact.jobTitle.isEmpty { contact.jobTitle = result.value.jobTitle }
            }
        case .event:
            event = ScanEventDraft.read(reading, detected: detected, title: documentTitle)
            if event == nil, let result = try? await assistant.suggestEvent(
                subject: documentTitle, body: reading.modelText, received: now(), timeZone: timeZone,
                allowFallback: false), let suggestion = result.value {
                event = ScanEventDraft(title: suggestion.title, start: suggestion.start, end: suggestion.end)
            }
        case .document, .table:
            break
        }
        try Task.checkCancellation()
        self.reading = reading
        self.kind = kind
        self.fields = fields
        self.contact = contact
        self.event = event
        notice = nil
    }

    /// "Read as": shows the scan as another kind, filling that kind's fields
    /// from the same text (nothing is read again).
    public func read(as kind: ScanKind) {
        guard let reading, phase == .review, kind != self.kind else { return }
        readTask?.cancel()
        isPreparing = true
        let id = UUID()
        let task = Task { [weak self] in
            guard let self else { return }
            let printed = ReceiptHeuristics.read(reading)
            try? await self.prepare(kind, reading: reading, printed: printed,
                                    detected: reading.detectedData(reference: self.now()))
            if !Task.isCancelled { self.isPreparing = false }
            self.pendingReads.removeAll { $0.id == id }
        }
        readTask = task
        pendingReads.append((id, task))
    }

    // MARK: Actions for any kind

    public var canSaveToNotes: Bool { saveToNotes != nil && !fullText.isEmpty }

    /// The full text as a new note in Notes, under the scan's title.
    public func saveTextToNotes() {
        guard let saveToNotes, !fullText.isEmpty else { return }
        do {
            try saveToNotes(title, fullText)
            notice = "Saved to Notes as “\(title)”."
        } catch {
            notice = "Notes could not save it: \(error.localizedDescription)"
        }
    }

    /// The full text as a plain-text file (UTF-8).
    public var textFile: Data { Data(fullText.utf8) }

    /// A file name for exports, from the title.
    public var exportName: String {
        let cleaned = title.components(separatedBy: CharacterSet(charactersIn: "/:\\?%*|\"<>")).joined()
            .trimmingCharacters(in: .whitespaces)
        return cleaned.isEmpty ? "Scan" : String(cleaned.prefix(60))
    }

    public func noteExported(to name: String) {
        notice = "Exported to \(name)."
    }

    // MARK: Event

    /// Hands the checked event to Calendar's proposals; nothing is added to
    /// Calendar until the person accepts it there.
    public func addEventToCalendar() {
        guard kind == .event, let event, !event.title.trimmingCharacters(in: .whitespaces).isEmpty else { return }
        onEventSuggested(ProposedEvent(title: event.title, start: event.start, end: event.end,
                                       source: ItemSource(kind: .scan, label: "Scan"),
                                       note: "From a scan. Nothing changes until you add it."))
        notice = "Sent to Calendar. Add it from Calendar's proposals."
    }

    // MARK: Contact

    /// Opens the checked card in Contacts, which asks before adding it. The
    /// card is a private temporary file, deleted with the next scan.
    public func addContact() {
        guard kind == .contact, !contact.isEmpty else { return }
        do {
            let directory = try contactFolder()
            let url = directory.appending(path: "\(exportName).vcf")
            try Data(contact.vCard.utf8).write(to: url, options: .atomic)
            openFile(url)
            notice = "Opened in Contacts. Confirm it there to add the card."
        } catch {
            notice = "The contact card could not be written: \(error.localizedDescription)"
        }
    }

    /// The card's file, while it exists (tests).
    public var contactCardURL: URL? {
        contactDirectory.map { $0.appending(path: "\(exportName).vcf") }
    }

    private func contactFolder() throws -> URL {
        if let contactDirectory { return contactDirectory }
        let directory = URL.temporaryDirectory.appending(path: "mference-contact-\(UUID().uuidString)",
                                                         directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true,
                                                attributes: [.posixPermissions: 0o700])
        contactDirectory = directory
        return directory
    }

    // MARK: Table

    /// A detected table as a CSV file (UTF-8 with BOM, formulas defused).
    public func tableFile(_ index: Int) -> Data? {
        tables.indices.contains(index) ? ScanTableExporter.fileData(tables[index]) : nil
    }

    /// The person edited a field: it is now theirs, so it no longer needs checking.
    public func setValue(_ value: String, for key: ScanFieldKey) {
        fields[key] = ScanField(key: key, value: value, confidence: .sure)
    }

    /// Saves the confirmed scan with a copy of its image. Throws, saving
    /// nothing, when the image can no longer be copied (moved or deleted).
    @discardableResult
    public func confirm() throws -> ScanRecord {
        guard phase == .review, kind == .receipt, let reading else { throw ScanSessionError.nothingToConfirm }
        let record = ScanRecord(kind: "receipt", title: title, recognizedText: reading.transcript)
        record.fieldsJSON = try fields.encoded()
        record.isConfirmed = true
        if let imageURL {
            record.imageRelativePath = try copyImage(imageURL, id: record.id)
        }
        context.insert(record)
        try context.save()
        savedRecordID = record.id
        phase = .saved
        return record
    }

    /// Clears the screen for the next scan, at any point: while reading (the
    /// read is cancelled and never reaches the screen), in review (unsaved
    /// edits are dropped), after a failure or after saving. Pasted images
    /// are deleted; a saved scan keeps its own copy.
    public func reset() {
        readTask?.cancel()
        readTask = nil
        phase = .empty
        imageURL = nil
        clearResults()
        if let pastedDirectory {
            try? FileManager.default.removeItem(at: pastedDirectory)
            self.pastedDirectory = nil
        }
    }

    /// Drops everything read from the last scan, and its contact card file.
    private func clearResults() {
        reading = nil
        savedRecordID = nil
        fields = ScanFields(fields: [])
        kind = .document
        isPreparing = false
        contact = ContactDraft()
        event = nil
        page = 0
        notice = nil
        documentTitle = "Scanned document"
        if let contactDirectory {
            try? FileManager.default.removeItem(at: contactDirectory)
            self.contactDirectory = nil
        }
    }

    /// The CSV file of every confirmed receipt (UTF-8 with BOM). A scan
    /// still in review is not included and not confirmed.
    public func exportCSVFile() throws -> Data {
        Data([0xEF, 0xBB, 0xBF]) + Data(try exportCSV().utf8)
    }

    /// CSV of every confirmed receipt, oldest first.
    public func exportCSV() throws -> String {
        let descriptor = FetchDescriptor<ScanRecord>(
            predicate: #Predicate { $0.isConfirmed && $0.kind == "receipt" },
            sortBy: [SortDescriptor(\.createdAt)])
        let rows = try context.fetch(descriptor).compactMap { record -> ScanExportRow? in
            guard let data = record.fieldsJSON, let fields = try? ScanFields.decode(data) else { return nil }
            return ScanExportRow(scannedAt: record.createdAt, fields: fields)
        }
        return ScanCSVExporter.csv(rows)
    }

    /// Path relative to Application Support/Mference.
    private func copyImage(_ url: URL, id: UUID) throws -> String {
        try FileManager.default.createDirectory(at: imageDirectory, withIntermediateDirectories: true)
        let name = "\(id.uuidString).\(url.pathExtension.lowercased())"
        let destination = imageDirectory.appending(path: name)
        try FileManager.default.copyItem(at: url, to: destination)
        return "Scans/\(name)"
    }
}

public enum ScanSessionError: Error, Sendable, Hashable {
    case nothingToConfirm
}

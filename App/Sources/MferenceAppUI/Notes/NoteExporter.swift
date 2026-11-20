import AppKit
import MferenceAppCore
import UniformTypeIdentifiers

/// Export formats (Notes spec 4.1).
public enum NoteExportFormat: String, CaseIterable, Sendable {
    case markdown, plainText, pdf

    public var label: String {
        switch self {
        case .markdown: "Markdown…"
        case .plainText: "Plain text…"
        case .pdf: "PDF…"
        }
    }

    var contentType: UTType {
        switch self {
        case .markdown: UTType(filenameExtension: "md") ?? .plainText
        case .plainText: .plainText
        case .pdf: .pdf
        }
    }

    var fileExtension: String {
        switch self {
        case .markdown: "md"
        case .plainText: "txt"
        case .pdf: "pdf"
        }
    }
}

/// Writes a note in a format, and asks where with a save panel.
@MainActor
enum NoteExporter {
    /// Writes `note` to `url` as `format`.
    static func write(_ note: Note, store: NoteStore, format: NoteExportFormat, to url: URL) throws {
        let document = store.document(of: note)
        // A body that could not be read would export as nothing.
        if document.readOnlyReason == .damaged { throw NoteExportError.unreadable }
        let times = segmentTimes(note)
        switch format {
        case .markdown:
            try NoteExport.writeMarkdown(document, noteID: note.id, attachments: store.attachments, to: url,
                                         segmentTimes: times)
        case .plainText:
            try NoteExport.writePlainText(document, to: url, segmentTimes: times)
        case .pdf:
            try pdfData(document, noteID: note.id, attachments: store.attachments, segmentTimes: times)
                .write(to: url, options: .atomic)
        }
    }

    /// Shows a save panel, then writes. Errors come back as a message.
    static func export(_ note: Note, store: NoteStore, format: NoteExportFormat) -> String? {
        let panel = NSSavePanel()
        panel.allowedContentTypes = [format.contentType]
        panel.nameFieldStringValue = NoteExport.fileName(for: store.displayTitle(of: note)) + "." + format.fileExtension
        panel.canCreateDirectories = true
        guard panel.runModal() == .OK, let url = panel.url else { return nil }
        do {
            try write(note, store: store, format: format, to: url)
            return nil
        } catch NoteExportError.unreadable {
            return "This note couldn't be read fully, so it can't be exported. Nothing in it was deleted."
        } catch {
            return "Couldn't export the note. \(error.localizedDescription)"
        }
    }

    /// The note laid out as the editor draws it, on US Letter pages.
    static func pdfData(_ document: NoteDocument, noteID: UUID, attachments: NoteAttachments,
                        segmentTimes: [UUID: TimeInterval]) throws -> Data {
        let page = NSPrintInfo.shared.paperSize
        let margin = NoteEditorMetrics.pdfMargin
        let view = NoteTextView(usingTextLayoutManager: true)
        view.configure()
        view.textContainerInset = .zero
        view.mappingContext = NoteTextMapping.Context(attachmentURL: { attachments.url(of: $0, in: noteID) },
                                                      segmentTime: { segmentTimes[$0] })
        let width = page.width - 2 * margin
        view.frame = CGRect(x: 0, y: 0, width: width, height: page.height)
        // The view grows to the whole note; pages are cut from it.
        view.isVerticallyResizable = true
        view.isHorizontallyResizable = false
        view.textContainer?.widthTracksTextView = true
        view.textContainer?.containerSize = NSSize(width: width, height: CGFloat.greatestFiniteMagnitude)
        view.appearance = NSAppearance(named: .aqua)
        view.load(document)
        if let manager = view.textLayoutManager {
            manager.ensureLayout(for: manager.documentRange)
            let used = manager.usageBoundsForTextContainer.maxY
            view.frame.size.height = max(ceil(used), page.height - 2 * margin)
        }
        let info = NSPrintInfo()
        info.paperSize = page
        info.topMargin = margin
        info.bottomMargin = margin
        info.leftMargin = margin
        info.rightMargin = margin
        info.horizontalPagination = .fit
        info.verticalPagination = .automatic
        // A normal print job saved to a file paginates (a PDF operation into
        // data makes one tall page).
        let file = FileManager.default.temporaryDirectory.appending(path: "mference-note-\(UUID().uuidString).pdf")
        defer { try? FileManager.default.removeItem(at: file) }
        info.jobDisposition = .save
        info.dictionary()[NSPrintInfo.AttributeKey.jobSavingURL] = file
        let operation = NSPrintOperation(view: view, printInfo: info)
        operation.showsPrintPanel = false
        operation.showsProgressPanel = false
        guard operation.run() else { throw CocoaError(.fileWriteUnknown) }
        return try Data(contentsOf: file)
    }

    static func segmentTimes(_ note: Note) -> [UUID: TimeInterval] {
        Dictionary(NoteSearch.transcript(note.recording?.transcriptJSON).map { ($0.id, $0.start) },
                   uniquingKeysWith: { first, _ in first })
    }
}

enum NoteExportError: Error {
    /// The stored body is damaged.
    case unreadable
}

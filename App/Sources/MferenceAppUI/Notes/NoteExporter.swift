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
        view.frame = CGRect(x: 0, y: 0, width: page.width - 2 * margin, height: page.height)
        view.appearance = NSAppearance(named: .aqua)
        view.load(document)
        if let manager = view.textLayoutManager {
            manager.ensureLayout(for: manager.documentRange)
            view.sizeToFit()
            var height: CGFloat = 0
            manager.enumerateTextLayoutFragments(from: manager.documentRange.endLocation,
                                                 options: [.reverse, .ensuresLayout]) { fragment in
                height = fragment.layoutFragmentFrame.maxY
                return false
            }
            view.frame.size.height = max(height, page.height - 2 * margin)
        }
        let info = NSPrintInfo()
        info.paperSize = page
        info.topMargin = margin
        info.bottomMargin = margin
        info.leftMargin = margin
        info.rightMargin = margin
        info.horizontalPagination = .fit
        info.verticalPagination = .automatic
        let data = NSMutableData()
        let operation = NSPrintOperation.pdfOperation(with: view, inside: view.bounds, to: data, printInfo: info)
        operation.showsPrintPanel = false
        operation.showsProgressPanel = false
        guard operation.run() else { throw CocoaError(.fileWriteUnknown) }
        return data as Data
    }

    static func segmentTimes(_ note: Note) -> [UUID: TimeInterval] {
        Dictionary(NoteSearch.transcript(note.recording?.transcriptJSON).map { ($0.id, $0.start) },
                   uniquingKeysWith: { first, _ in first })
    }
}

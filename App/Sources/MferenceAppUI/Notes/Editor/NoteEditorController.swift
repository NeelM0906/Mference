import AppKit
import MferenceAppCore
import Observation
import UniformTypeIdentifiers

/// Owns the open note's editor: loads a note into the text view, autosaves
/// (0.5 s after the last change, and on switching notes, app deactivation
/// and quit; Notes spec 4.1), and exposes the toolbar's commands and state.
/// Dictation (N3's controller) and meetings insert through it.
@MainActor
@Observable
public final class NoteEditorController {
    public enum SaveState: Equatable { case saved, editing, failed(String) }

    @ObservationIgnored let store: NoteStore
    @ObservationIgnored private let autosaveDelay: Duration
    @ObservationIgnored private(set) weak var textView: NoteTextView?
    @ObservationIgnored private var autosave: Task<Void, Never>?
    @ObservationIgnored private var observers: [NSObjectProtocol] = []
    @ObservationIgnored private var segmentTimes: [UUID: TimeInterval] = [:]

    /// The open note.
    public private(set) var note: Note?
    public private(set) var saveState: SaveState = .saved
    public private(set) var wordCount = 0
    /// Bumped when the caret or selection moves, for the toolbar's state.
    public private(set) var selectionRevision = 0

    public init(store: NoteStore, autosaveDelay: Duration = .milliseconds(500)) {
        self.store = store
        self.autosaveDelay = autosaveDelay
        let center = NotificationCenter.default
        for name in [NSApplication.didResignActiveNotification, NSApplication.willTerminateNotification] {
            observers.append(center.addObserver(forName: name, object: nil, queue: .main) { [weak self] _ in
                MainActor.assumeIsolated { self?.saveNow() }
            })
        }
    }

    isolated deinit {
        autosave?.cancel()
        for observer in observers { NotificationCenter.default.removeObserver(observer) }
    }

    // MARK: Opening

    /// Saves the open note, then shows `note` (nil shows nothing).
    public func open(_ note: Note?) {
        guard note?.id != self.note?.id || note == nil else { return }
        saveNow()
        textView?.dropAllPhrases()
        self.note = note
        loadIntoView()
    }

    /// Re-reads the open note (after a change made elsewhere, e.g. a title).
    public func reload() {
        loadIntoView()
    }

    func attach(_ textView: NoteTextView) {
        self.textView = textView
        textView.onTextChange = { [weak self] in self?.textDidChange() }
        textView.importImage = { [weak self] data, ext in
            guard let self, let id = note?.id else { return nil }
            return try? store.attachments.write(data, fileExtension: ext, into: id)
        }
        loadIntoView()
    }

    func selectionDidChange() {
        selectionRevision += 1
    }

    private func loadIntoView() {
        autosave?.cancel()
        guard let textView else { return }
        segmentTimes = [:]
        if let data = note?.recording?.transcriptJSON {
            for segment in NoteSearch.transcript(data) { segmentTimes[segment.id] = segment.start }
        }
        let noteID = note?.id
        let attachments = store.attachments
        let times = segmentTimes
        textView.mappingContext = NoteTextMapping.Context(
            attachmentURL: { id in noteID.map { attachments.url(of: id, in: $0) } },
            segmentTime: { times[$0] })
        textView.isEditable = note != nil && note?.deletedAt == nil
        textView.load(note.map { store.document(of: $0) } ?? NoteDocument(blocks: []))
        wordCount = textView.document.wordCount
        saveState = .saved
    }

    // MARK: Saving

    private func textDidChange() {
        guard note != nil else { return }
        saveState = .editing
        autosave?.cancel()
        let delay = autosaveDelay
        autosave = Task { [weak self] in
            try? await Task.sleep(for: delay)
            guard !Task.isCancelled else { return }
            self?.saveNow()
        }
    }

    /// Saves now if anything changed (switching notes, deactivation, quit).
    public func saveNow() {
        autosave?.cancel()
        autosave = nil
        guard let note, let textView, saveState != .saved else { return }
        let document = textView.document
        wordCount = document.wordCount
        do {
            try store.save(document, to: note)
            saveState = .saved
        } catch {
            saveState = .failed(String(describing: error))
        }
    }

    // MARK: Commands

    /// The caret line's kind (the Aa menu's tick).
    public var currentKind: NoteBlockKind {
        _ = selectionRevision
        return NoteBlockKind(textView?.currentBlockStyle.kind ?? .body)
    }

    public func apply(_ kind: NoteBlockKind) {
        textView?.applyBlockKind(kind.styleKind)
    }

    public func toggleChecklist() { textView?.toggleChecklist() }
    public func toggleBold() { textView?.toggle(.bold) }
    public func toggleItalic() { textView?.toggle(.italic) }
    public func toggleUnderline() { textView?.toggle(.underline) }
    public func toggleStrikethrough() { textView?.toggle(.strikethrough) }

    /// Copies an image or audio file into the note and adds it after the caret's line.
    public func attachFile(_ url: URL) throws {
        guard let note, let textView else { return }
        let id = try store.attachments.importFile(url, into: note.id)
        let type = UTType(filenameExtension: url.pathExtension)
        let block: NoteDocument.Block = (type?.conforms(to: .audiovisualContent) ?? false)
            ? .audio(attachmentID: id) : .image(attachmentID: id)
        textView.insertAttachment(block)
    }

    /// The note as plain text, for the share sheet.
    public var shareText: String {
        _ = selectionRevision
        _ = wordCount
        return textView?.document.plainText ?? note.map { store.document(of: $0).plainText } ?? ""
    }

    public func focus() {
        guard let textView else { return }
        textView.window?.makeFirstResponder(textView)
    }

    // MARK: Dictation and meetings

    /// Holds the caret's place for a dictated phrase.
    public func beginPhrase(_ id: UUID) { textView?.beginPhrase(id) }

    /// Inserts a phrase where it began; marks the note as dictated.
    public func finishPhrase(_ id: UUID, text: String) {
        guard let textView, let note else { return }
        if textView.finishPhrase(id, text: text) {
            try? store.markDictated(note)
        }
    }

    public func dropPhrase(_ id: UUID) { textView?.dropPhrase(id) }
    public func dropAllPhrases() { textView?.dropAllPhrases() }

    /// "Mark moment" in a meeting note.
    public func insertMarker(time: TimeInterval) { textView?.insertMarker(time: time) }
}

/// The text style menu's choices.
public enum NoteBlockKind: String, CaseIterable, Sendable {
    case title, heading, subheading, body, bullet, numbered, checklist

    public var label: String {
        switch self {
        case .title: "Title"
        case .heading: "Heading"
        case .subheading: "Subheading"
        case .body: "Body"
        case .bullet: "Bulleted list"
        case .numbered: "Numbered list"
        case .checklist: "Checklist"
        }
    }

    init(_ kind: NoteBlockStyle.Kind) {
        switch kind {
        case .title: self = .title
        case .heading: self = .heading
        case .subheading: self = .subheading
        case .body: self = .body
        case .bullet: self = .bullet
        case .numbered: self = .numbered
        case .check: self = .checklist
        }
    }

    var styleKind: NoteBlockStyle.Kind {
        switch self {
        case .title: .title
        case .heading: .heading
        case .subheading: .subheading
        case .body: .body
        case .bullet: .bullet
        case .numbered: .numbered
        case .checklist: .check
        }
    }
}

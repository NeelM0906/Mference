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
    /// Phrases still being transcribed when their note was switched away
    /// from: each finishes into an off-screen copy of that note (spec 4.2).
    @ObservationIgnored private var detached: [UUID: DetachedNote] = [:]

    private final class DetachedNote {
        let note: Note
        let view: NoteTextView
        init(note: Note, view: NoteTextView) {
            self.note = note
            self.view = view
        }
    }

    /// The open note.
    public private(set) var note: Note?
    public private(set) var saveState: SaveState = .saved
    /// Why the open note cannot be edited, when its body is newer or damaged.
    public private(set) var readOnlyReason: NoteDocument.ReadOnlyReason?
    public private(set) var wordCount = 0
    /// Bumped when the caret or selection moves, for the toolbar's state.
    public private(set) var selectionRevision = 0

    public init(store: NoteStore, autosaveDelay: Duration = .milliseconds(500)) {
        self.store = store
        self.autosaveDelay = autosaveDelay
        let center = NotificationCenter.default
        for name in [NSApplication.didResignActiveNotification, NSApplication.willTerminateNotification] {
            // No queue: runs at once on the posting (main) thread, before the
            // app goes inactive or quits.
            observers.append(center.addObserver(forName: name, object: nil, queue: nil) { [weak self] _ in
                MainActor.assumeIsolated { self?.saveNow(force: true) }
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
        saveNow(force: true)
        detachPendingPhrases()
        self.note = note
        loadIntoView()
    }

    /// Re-reads the open note (after a change made elsewhere, e.g. a title),
    /// saving what was typed first.
    public func reload() {
        saveNow(force: true)
        loadIntoView()
    }

    func attach(_ textView: NoteTextView) {
        if let old = self.textView, old !== textView { saveNow(force: true) }
        self.textView = textView
        textView.onTextChange = { [weak self] in self?.textDidChange() }
        textView.importImage = { [weak self] data, ext in
            guard let self, let id = note?.id else { return nil }
            return try? store.attachments.write(data, fileExtension: ext, into: id)
        }
        textView.importFile = { [weak self] url in
            guard let self, let id = note?.id else { return nil }
            return Self.importBlock(url, into: id, attachments: store.attachments)
        }
        textView.copyAttachment = { [weak self] attachmentID, source in
            guard let self, let id = note?.id else { return nil }
            let file = store.attachments.url(of: attachmentID, in: source)
            return try? store.attachments.importFile(file, into: id)
        }
        loadIntoView()
    }

    /// The view is going away (leaving Notes): save what it holds now.
    func detach(_ textView: NoteTextView) {
        guard self.textView === textView else { return }
        saveNow(force: true)
    }

    /// Moves phrases still waiting in the open note to an off-screen copy
    /// of it, so they land in the note they began in.
    private func detachPendingPhrases() {
        guard let textView, let note, !textView.pendingPhrases.isEmpty else { return }
        let holder = NoteTextView(usingTextLayoutManager: true)
        holder.configure()
        holder.mappingContext = textView.mappingContext
        holder.load(store.document(of: note))
        holder.adoptPendingPhrases(textView.pendingPhrases)
        let detachedNote = DetachedNote(note: note, view: holder)
        for phrase in textView.pendingPhrases { detached[phrase.id] = detachedNote }
        textView.dropAllPhrases()
    }

    static func importBlock(_ url: URL, into noteID: UUID, attachments: NoteAttachments) -> NoteDocument.Block? {
        guard let type = UTType(filenameExtension: url.pathExtension),
              type.conforms(to: .image) || type.conforms(to: .audiovisualContent),
              let id = try? attachments.importFile(url, into: noteID) else { return nil }
        return type.conforms(to: .image) ? .image(attachmentID: id) : .audio(attachmentID: id)
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
        let document = note.map { store.document(of: $0) } ?? NoteDocument(blocks: [])
        // Notes in Recently deleted, and bodies from a newer version or
        // damaged ones, are shown but never edited (nothing saves over them).
        textView.isEditable = note != nil && note?.deletedAt == nil && document.readOnlyReason == nil
        textView.noteID = note?.id
        textView.load(document)
        readOnlyReason = document.readOnlyReason
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
    /// While an input method is composing, an autosave waits for it;
    /// `force` (switching, leaving, quitting) saves the text so far.
    public func saveNow(force: Bool = false) {
        guard let note, let textView, saveState != .saved else {
            autosave?.cancel()
            autosave = nil
            return
        }
        if textView.hasMarkedText(), !force {
            textDidChange()
            return
        }
        autosave?.cancel()
        autosave = nil
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

    public func focus() {
        guard let textView else { return }
        textView.window?.makeFirstResponder(textView)
    }

    // MARK: Dictation and meetings

    /// Holds the caret's place for a dictated phrase.
    public func beginPhrase(_ id: UUID) { textView?.beginPhrase(id) }

    /// Inserts a phrase where it began, in the note it began in; marks that
    /// note as dictated.
    public func finishPhrase(_ id: UUID, text: String) {
        if let holder = detached.removeValue(forKey: id) {
            guard holder.view.finishPhrase(id, text: text) else { return }
            try? store.save(holder.view.document, to: holder.note)
            try? store.markDictated(holder.note)
            return
        }
        guard let textView, let note else { return }
        if textView.finishPhrase(id, text: text) {
            try? store.markDictated(note)
        }
    }

    public func dropPhrase(_ id: UUID) {
        if detached.removeValue(forKey: id) != nil { return }
        textView?.dropPhrase(id)
    }

    public func dropAllPhrases() {
        detached.removeAll()
        textView?.dropAllPhrases()
    }

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

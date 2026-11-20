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
    /// The stored body the editor shows (as loaded or last saved). When the
    /// store's copy differs, something else wrote the note (a meeting
    /// summary, a marker, a title): the editor takes that in instead of
    /// saving over it.
    @ObservationIgnored private var loadedBody: Data?
    /// An outside write arrived while an input method was composing.
    @ObservationIgnored private var absorbWhenComposed = false
    /// The note the running dictation writes into: the note shown when its
    /// first phrase began. Every later phrase goes there too, shown or not,
    /// until the dictation ends (`dropAllPhrases`).
    @ObservationIgnored private var dictationNote: Note?
    /// Where the caret was in each note when it was left (a phrase for a
    /// note not shown starts there).
    @ObservationIgnored private var lastCaret: [UUID: Int] = [:]
    /// A ⌘K transcript hit waiting for the editor to show its note.
    @ObservationIgnored private var pendingReveal: UUID?
    /// Phrases still being transcribed when their note left the editor
    /// (another note opened, the view went away or was replaced). They are
    /// handed back when the note is shown again; one that finishes before
    /// that lands in the note's stored body, never in a stale copy (spec 4.2).
    @ObservationIgnored private var detached: [UUID: DetachedNote] = [:]

    private final class DetachedNote {
        let note: Note
        /// Off-screen editor holding the phrases' places in `snapshot`.
        let view: NoteTextView
        /// The stored body the places refer to.
        var snapshot: NoteDocument
        init(note: Note, view: NoteTextView, snapshot: NoteDocument) {
            self.note = note
            self.view = view
            self.snapshot = snapshot
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
        watchStore()
    }

    /// Follows the store's revision: a write to the open note from
    /// elsewhere is taken in at once.
    private func watchStore() {
        withObservationTracking {
            _ = store.revision
        } onChange: { [weak self] in
            Task { @MainActor [weak self] in
                self?.absorbExternalChanges()
                self?.watchStore()
            }
        }
    }

    /// When the open note's stored body changed underneath the editor:
    /// with nothing unsaved, show the stored body; with unsaved edits, merge
    /// them into it block by block and save the result. Caret, selection and
    /// pending phrases keep their places. Returns whether anything changed.
    @discardableResult
    func absorbExternalChanges() -> Bool {
        guard let note, let textView, note.modelContext != nil, !note.isDeleted,
              note.bodyJSON != loadedBody else { return false }
        // Never in the middle of an input method's composition: taken in
        // as soon as the composed text is committed.
        if textView.hasMarkedText() {
            absorbWhenComposed = true
            return false
        }
        absorbWhenComposed = false
        let theirs = store.document(of: note)
        guard theirs.readOnlyReason == nil else { return false }
        if saveState == .saved {
            textView.reload(theirs)
            loadedBody = note.bodyJSON
        } else {
            let base = NoteDocument.decode(loadedBody)
            let merged = NoteDocument(blocks: NoteDocumentMerge.merge(base: base.blocks, ours: textView.document.blocks,
                                                                      theirs: theirs.blocks))
            textView.reload(merged)
            do {
                try store.save(merged, to: note)
                loadedBody = note.bodyJSON
                saveState = .saved
                autosave?.cancel()
            } catch {
                saveState = .failed(String(describing: error))
            }
        }
        wordCount = textView.document.wordCount
        return true
    }

    isolated deinit {
        autosave?.cancel()
        for observer in observers { NotificationCenter.default.removeObserver(observer) }
    }

    // MARK: Opening

    /// Saves the open note, then shows `note` (nil shows nothing).
    public func open(_ note: Note?) {
        guard note?.id != self.note?.id || note == nil else { return }
        pendingReveal = nil
        if let current = self.note, let textView { lastCaret[current.id] = textView.selectedRange().location }
        saveNow(force: true)
        detachPendingPhrases(from: textView)
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
        if let old = self.textView, old !== textView {
            saveNow(force: true)
            detachPendingPhrases(from: old)
        }
        self.textView = textView
        textView.onTextChange = { [weak self] in self?.textDidChange() }
        textView.onFocusChange = { [weak self] focused in self?.onFocusChange?(focused) }
        textView.transcribeFiles = { [weak self] urls in self?.onMediaDropped?(urls) ?? false }
        textView.onCancel = { [weak self] in self?.onCancel?() ?? false }
        textView.onTimeChipClick = onTimeChipClick
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

    /// The view is going away (leaving Notes): save what it holds now and
    /// keep its waiting phrases for their note.
    func detach(_ textView: NoteTextView) {
        guard self.textView === textView else { return }
        saveNow(force: true)
        if let note { lastCaret[note.id] = textView.selectedRange().location }
        detachPendingPhrases(from: textView)
        self.textView = nil
        // No editor shows any more: it is no longer a place to dictate into.
        onFocusChange?(false)
    }

    /// Moves the phrases waiting in `view` (the open note, just saved) to an
    /// off-screen copy of that note's stored body.
    private func detachPendingPhrases(from view: NoteTextView?) {
        guard let view, let note, !view.pendingPhrases.isEmpty else { return }
        let stored = store.document(of: note)
        let holder = NoteTextView(usingTextLayoutManager: true)
        holder.configure()
        holder.mappingContext = view.mappingContext
        holder.load(stored)
        holder.adoptPendingPhrases(view.pendingPhrases)
        let detachedNote = DetachedNote(note: note, view: holder, snapshot: stored)
        for phrase in view.pendingPhrases { detached[phrase.id] = detachedNote }
        view.dropAllPhrases()
    }

    /// The open note's phrases that were waiting off screen come back to the
    /// live editor, at their places if the body is unchanged, else at the end.
    private func reattachPendingPhrases(stored: NoteDocument) {
        guard let textView, let note else { return }
        let mine = detached.filter { $0.value.note.id == note.id }
        guard !mine.isEmpty else { return }
        var phrases = textView.pendingPhrases
        var seen = Set<ObjectIdentifier>()
        for holder in mine.values where seen.insert(ObjectIdentifier(holder)).inserted {
            let end = (textView.string as NSString).length
            for phrase in holder.view.pendingPhrases where mine[phrase.id] != nil && !phrases.contains(where: { $0.id == phrase.id }) {
                phrases.append((phrase.id, holder.snapshot == stored ? phrase.location : end))
            }
        }
        for id in mine.keys { detached.removeValue(forKey: id) }
        textView.adoptPendingPhrases(phrases)
    }

    /// The editor's one-line explanation for a note it opens read-only.
    static func notice(for reason: NoteDocument.ReadOnlyReason?) -> String {
        switch reason {
        case .newerVersion: "This note was made by a newer version of Mference and opens read-only."
        case .damaged: "Part of this note couldn't be read, so it opens read-only. Nothing in it was deleted."
        case nil: ""
        }
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
        loadedBody = note?.bodyJSON
        textView.notice = Self.notice(for: document.readOnlyReason)
        if let pendingReveal, note != nil {
            textView.reveal(segmentID: pendingReveal)
            self.pendingReveal = nil
        }
        reattachPendingPhrases(stored: document)
        readOnlyReason = document.readOnlyReason
        wordCount = textView.document.wordCount
        saveState = .saved
    }

    // MARK: Saving

    private func textDidChange() {
        guard note != nil else { return }
        if absorbWhenComposed, let textView, !textView.hasMarkedText() { absorbExternalChanges() }
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
        // A write from elsewhere not yet taken in is merged first (a forced
        // save commits an unfinished composition first).
        if force, textView.hasMarkedText(), note.bodyJSON != loadedBody { textView.unmarkText() }
        if absorbExternalChanges(), saveState == .saved { return }
        let document = textView.document
        wordCount = document.wordCount
        do {
            try store.save(document, to: note)
            loadedBody = note.bodyJSON
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

    /// The editor gained or lost keyboard focus (⇧⌘D dictates into the
    /// focused editor).
    @ObservationIgnored public var onFocusChange: ((Bool) -> Void)?
    /// Audio or video dropped on the note; returns true when queued for
    /// transcription (otherwise the files attach).
    @ObservationIgnored public var onMediaDropped: (([URL]) -> Bool)?
    /// Esc in the editor; returns true when handled (dictation stopped).
    @ObservationIgnored public var onCancel: (() -> Bool)?

    /// The note as Markdown, for the share sheet (spec 4.1).
    public var shareText: String {
        _ = selectionRevision
        _ = wordCount
        guard let document = textView?.document ?? note.map({ store.document(of: $0) }) else { return "" }
        return NoteExport.markdown(document, segmentTimes: segmentTimes)
    }

    /// Scrolls to a transcript segment's time chip, if the note has one.
    /// When the editor is not showing yet (⌘K opened Notes from another
    /// section), it happens as soon as the note loads.
    public func reveal(segmentID: UUID) {
        pendingReveal = segmentID
        guard let textView, note != nil else { return }
        textView.reveal(segmentID: segmentID)
        pendingReveal = nil
    }

    public func focus() {
        guard let textView else { return }
        textView.window?.makeFirstResponder(textView)
    }

    // MARK: Dictation and meetings

    /// Holds the caret's place for a dictated phrase.
    /// Binds the dictation now starting to the note shown (so even a phrase
    /// flushed after a switch, before any phrase began, lands here).
    public func dictationWillStart() {
        dictationNote = note
    }

    public func beginPhrase(_ id: UUID) {
        // Without a start call (older drivers): the first phrase binds.
        if dictationNote == nil { dictationNote = note }
        guard let target = dictationNote else { return }
        if target.id == note?.id, let textView {
            textView.beginPhrase(id)
        } else {
            // The person moved to another note: the phrase still belongs to
            // the note the dictation began in.
            let holder = holder(for: target)
            holder.view.beginPhrase(id)
            detached[id] = holder
        }
    }

    /// The off-screen copy of `note` that phrases for it wait in: the one
    /// already holding its phrases, else a new one at its last caret.
    private func holder(for note: Note) -> DetachedNote {
        if let existing = detached.values.first(where: { $0.note.id == note.id }) { return existing }
        let stored = store.document(of: note)
        let view = NoteTextView(usingTextLayoutManager: true)
        view.configure()
        view.mappingContext = textView?.mappingContext ?? NoteTextMapping.Context()
        view.load(stored)
        let length = (view.string as NSString).length
        view.setSelectedRange(NSRange(location: min(lastCaret[note.id] ?? length, length), length: 0))
        return DetachedNote(note: note, view: view, snapshot: stored)
    }

    /// Inserts a phrase where it began, in the note it began in; marks that
    /// note as dictated.
    public func finishPhrase(_ id: UUID, text: String) {
        if let holder = detached.removeValue(forKey: id) {
            // Insert into what is stored now, never into a stale copy: if the
            // note changed since (a summary, another window), the phrase goes
            // at the end of its text.
            let stored = store.document(of: holder.note)
            if stored != holder.snapshot {
                let waiting = holder.view.pendingPhrases
                holder.view.load(stored)
                let end = (holder.view.string as NSString).length
                holder.view.adoptPendingPhrases(waiting.map { ($0.id, end) })
                holder.snapshot = stored
            }
            guard holder.view.finishPhrase(id, text: text) else { return }
            do {
                try store.save(holder.view.document, to: holder.note)
                holder.snapshot = store.document(of: holder.note)
                // The next phrase for this note follows this one.
                lastCaret[holder.note.id] = holder.view.selectedRange().location
                try store.markDictated(holder.note)
            } catch {
                saveState = .failed(String(describing: error))
            }
            return
        }
        // A phrase never begun (its start was missed) still goes to the
        // dictation's note, not to whichever note shows.
        if dictationNote == nil { dictationNote = note }
        if let target = dictationNote, target.id != note?.id || textView == nil,
           textView?.pendingPhraseIDs.contains(id) != true {
            let holder = holder(for: target)
            holder.view.beginPhrase(id)
            detached[id] = holder
            finishPhrase(id, text: text)
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

    /// The dictation ended or failed: its waiting phrases are forgotten and
    /// the next dictation binds to the note shown then.
    public func dropAllPhrases() {
        detached.removeAll()
        textView?.dropAllPhrases()
        dictationNote = nil
    }

    /// Gives up keyboard focus (the setup card covers the editor).
    public func resignFocus() {
        guard let textView, textView.window?.firstResponder === textView else { return }
        textView.window?.makeFirstResponder(nil)
    }

    /// "Mark moment" in a meeting note.
    /// Adds the marker when `noteID` is the open note; returns false
    /// otherwise (the caller writes to that note through the store).
    @discardableResult
    public func insertMarker(time: TimeInterval, in noteID: UUID) -> Bool {
        guard let textView, note?.id == noteID, textView.isEditable else { return false }
        textView.insertMarker(time: time)
        return true
    }

    /// A time chip for a transcript segment, after the caret's line.
    @discardableResult
    public func insertTranscriptAnchor(segmentID: UUID, in noteID: UUID) -> Bool {
        guard let textView, note?.id == noteID, textView.isEditable else { return false }
        if let note, segmentTimes[segmentID] == nil,
           let segment = NoteSearch.transcript(note.recording?.transcriptJSON).first(where: { $0.id == segmentID }) {
            segmentTimes[segmentID] = segment.start
            let times = segmentTimes
            textView.mappingContext.segmentTime = { times[$0] }
        }
        textView.insertAttachment(.transcriptAnchor(segmentID: segmentID))
        return true
    }

    /// Called with the recording time of a clicked time chip or marker
    /// (the meeting player seeks there).
    public var onTimeChipClick: ((TimeInterval) -> Void)? {
        didSet { textView?.onTimeChipClick = onTimeChipClick }
    }
}

extension NoteEditorController: DictationInsertionTarget, MeetingNoteTarget {
    /// The open note; a fixed nil UUID while none is open (dictation is only
    /// offered with a note open).
    /// While dictating: the note the dictation writes into (it may no
    /// longer be the one shown). Otherwise the open note.
    public var noteID: UUID {
        dictationNote?.id ?? note?.id ?? UUID(uuid: (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
    }
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

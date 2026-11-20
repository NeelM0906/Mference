import AppKit
import MferenceAppCore
@testable import MferenceAppUI
import MferenceDesign
import SwiftData
import SwiftUI
import Testing

/// Board 1 fixtures: the notes, folders and counts the Main board shows.
@MainActor
enum NotesFixtures {
    static var calendar: Calendar = {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(identifier: "Europe/Lisbon")!
        return calendar
    }()
    static let locale = Locale(identifier: "en_GB")

    static func date(_ m: Int, _ d: Int, _ h: Int = 12, _ min: Int = 0, year: Int = 2026) -> Date {
        calendar.date(from: DateComponents(year: year, month: m, day: d, hour: h, minute: min))!
    }

    /// Thursday 19 November 2026, 11:40.
    static let now = date(11, 19, 11, 40)

    static let lisbon = NoteDocument(blocks: [
        .heading(level: .title, runs: [TextRun("Lisbon trip, March")]),
        .paragraph(runs: [TextRun("Flights booked for 12–19 March. Still need a hotel in Alfama and a plan for the day trip.")]),
        .heading(level: .heading, runs: [TextRun("To do")]),
        .check(runs: [TextRun("Book flights")], done: true, indent: 0),
        .check(runs: [TextRun("Hotel, four nights in Alfama")], done: false, indent: 0),
        .check(runs: [TextRun("Ask Marta which tram passes she used")], done: false, indent: 0),
        .check(runs: [TextRun("Renew travel insurance before February")], done: false, indent: 0),
        .heading(level: .heading, runs: [TextRun("Ideas")]),
        .bullet(runs: [TextRun("Sintra day trip, leave on the first train")], indent: 0),
        .bullet(runs: [TextRun("Time Out Market for lunch on day one")], indent: 0),
        .bullet(runs: [TextRun("Fado night in Mouraria, book ahead")], indent: 0),
    ])

    /// A model over an in-memory store with the board's 24 notes.
    static func model(speech: FakeSpeechService = FakeSpeechService()) throws -> NotesModel {
        let store = NoteStore(container: try AppModelContainer.inMemory(), attachments: .temporary(), speech: speech)
        let personal = try store.createFolder(named: "Personal")
        let work = try store.createFolder(named: "Work")
        let ideas = try store.createFolder(named: "Ideas")

        @discardableResult func note(_ title: String, _ body: [String], at date: Date, kind: NoteKind = .note, folder: NoteFolder?,
                  pinned: Bool = false, dictated: Bool = false, document: NoteDocument? = nil) throws -> Note {
            let note = try store.createNote(kind: kind, in: folder, now: date)
            let doc = document ?? NoteDocument(blocks: [.heading(level: .title, runs: [TextRun(title)])]
                + body.map { .paragraph(runs: [TextRun($0)]) })
            try store.save(doc, to: note, now: date)
            if pinned { try store.setPinned(true, note) }
            if dictated { try store.markDictated(note) }
            return note
        }

        let lisbon = try note("", [], at: date(11, 19, 11, 40), folder: personal, document: lisbon)
        try note("Weekly groceries", ["Oat milk, eggs, basil, lemons, rice"], at: date(11, 15, 9), folder: personal, pinned: true)
        let review = try note("Design review: Notes", ["Agreed the three-pane layout."], at: date(11, 19, 10, 2),
                              kind: .meeting, folder: work)
        let recording = NoteRecording(startedAt: date(11, 19, 9, 20))
        recording.duration = 42 * 60
        recording.summaryState = .written
        store.context.insert(recording)
        review.recording = recording
        let interview = try note("Interview with Dana Ortiz", ["Talked about the move to Porto."], at: date(11, 18, 16, 15),
                                 kind: .file, folder: work)
        let file = NoteRecording(startedAt: date(11, 18, 15, 30), sourceFileName: "interview-dana.m4a")
        file.duration = 38 * 60
        store.context.insert(file)
        interview.recording = file
        try note("Reading list", ["Thinking in Systems; Four Thousand Weeks"], at: date(11, 18, 9, 12), folder: ideas)
        try note("Dentist questions", ["Ask whether the night guard needs replacing"], at: date(11, 16, 18),
                 folder: personal, dictated: true)
        try note("Kitchen shelf measurements", ["Left wall 182 cm, alcove 64 cm deep"], at: date(11, 15, 14), folder: personal)

        // Older notes to fill the counts (All 24, Meetings 6, Audio 3,
        // Personal 9, Work 11, Ideas 4).
        for index in 0..<5 {
            let meeting = try note("Weekly sync \(index + 1)", ["Notes from the sync."], at: date(10, 20 - index * 3),
                                   kind: .meeting, folder: work)
            let r = NoteRecording(startedAt: meeting.createdAt)
            r.duration = 30 * 60
            store.context.insert(r)
            meeting.recording = r
        }
        for index in 0..<2 {
            let transcript = try note("Voice memo \(index + 1)", ["From the walk home."], at: date(9, 10 + index),
                                      kind: .file, folder: work)
            let r = NoteRecording(startedAt: transcript.createdAt, sourceFileName: "memo-\(index + 1).m4a")
            r.duration = 6 * 60
            store.context.insert(r)
            transcript.recording = r
        }
        for index in 0..<2 { try note("Work note \(index + 1)", ["Follow up."], at: date(8, 3 + index), folder: work) }
        for index in 0..<5 { try note("Personal note \(index + 1)", ["Remember this."], at: date(7, 3 + index), folder: personal) }
        for index in 0..<3 { try note("Idea \(index + 1)", ["Maybe later."], at: date(6, 3 + index), folder: ideas) }
        try store.context.save()
        for index in 0..<2 {
            let gone = try note("Old draft \(index + 1)", ["Not needed."], at: date(11, 1 + index), folder: nil)
            try store.moveToTrash(gone, now: date(11, 10))
        }

        let model = NotesModel(store: store, speech: speech, now: { now })
        model.calendar = calendar
        model.locale = locale
        model.voice = NotesVoiceActions(transcribeFile: {}, recordMeeting: {}, toggleDictation: { _ in })
        model.select(lisbon)
        return model
    }
}

/// The Notes screen (Notes spec 4.1): model behaviour and the G-A3 renders.
@MainActor
@Suite(.serialized) struct NotesScreenTests {
    @Test func fixtureCountsMatchTheBoard() throws {
        let model = try NotesFixtures.model()
        let counts = model.counts
        #expect(counts.all == 24)
        #expect(counts.meetings == 6)
        #expect(counts.audioTranscripts == 3)
        #expect(counts.recentlyDeleted == 2)
        #expect(model.folders.map { counts.folders[$0.id] ?? 0 } == [9, 11, 4])
    }

    @Test func listGroupsRowsAndPreviewsAsTheBoard() throws {
        let model = try NotesFixtures.model()
        let sections = model.sections
        #expect(Array(sections.prefix(4).map(\.title)) == ["Pinned", "Today", "Yesterday", "Previous 7 days"])
        #expect(sections[0].items.map(model.title(of:)) == ["Weekly groceries"])
        #expect(sections[1].items.map(model.title(of:)) == ["Lisbon trip, March", "Design review: Notes"])
        #expect(sections[2].items.map(model.title(of:)) == ["Interview with Dana Ortiz", "Reading list"])
        #expect(sections[3].items.map(model.title(of:)) == ["Dentist questions", "Kitchen shelf measurements"])
        let rows = sections.flatMap(\.items)
        #expect(model.timeLabel(of: rows[0]) == "Sun")
        #expect(model.timeLabel(of: rows[1]) == "11:40")
        #expect(model.preview(of: rows[1]).hasPrefix("Flights booked for 12–19 March."))
        #expect(model.preview(of: rows[2]) == "42 min · summary")
        #expect(model.preview(of: rows[3]) == "From interview-dana.m4a · 38 min")
        #expect(model.timeLabel(of: rows[5]) == "Mon")
        #expect(rows[5].hasDictation)
    }

    @Test func newNoteGoesIntoTheOpenFolderAndIsSelected() throws {
        let model = try NotesFixtures.model()
        let ideas = try #require(model.folders.last)
        model.scope = .folder(ideas.id)
        model.newNote()
        let selected = try #require(model.selectedNote)
        #expect(selected.folder?.id == ideas.id)
        #expect(model.notes.first?.id == selected.id)
        #expect(model.counts.folders[ideas.id] == 5)

        // From Meetings, a new note goes to All notes.
        model.scope = .meetings
        model.newNote()
        #expect(model.scope == .all)
    }

    @Test func deleteSelectsTheNextNoteAndRestoreBringsItBack() throws {
        let model = try NotesFixtures.model()
        let lisbon = try #require(model.selectedNote)
        model.delete(lisbon)
        #expect(model.selectedNote?.id != lisbon.id)
        #expect(model.selectedNote != nil)
        #expect(model.counts.recentlyDeleted == 3)
        model.scope = .recentlyDeleted
        #expect(model.preview(of: lisbon) == "30 days left")
        model.restore(lisbon)
        #expect(model.counts.recentlyDeleted == 2)
    }

    @Test func emptyTrashRemovesNotesAndTheirAudio() async throws {
        let speech = FakeSpeechService()
        let model = try NotesFixtures.model(speech: speech)
        model.scope = .recentlyDeleted
        let ids = model.notes.map(\.id)
        await model.emptyTrash()
        #expect(model.counts.recentlyDeleted == 0)
        for id in ids { #expect(speech.calls.contains(.deleteRecordings(id))) }
    }

    @Test func foldersCreateRenameDeleteAndTakeDroppedNotes() throws {
        let model = try NotesFixtures.model()
        let lisbon = try #require(model.selectedNote)
        model.newFolder()
        let folder = try #require(model.currentFolder)
        #expect(folder.name == "New folder")
        #expect(model.renamingFolderID == folder.id)
        model.rename(folder, to: "Travel")
        #expect(model.renamingFolderID == nil)
        #expect(model.folders.map(\.name) == ["Personal", "Work", "Ideas", "Travel"])

        #expect(model.moveNotes([lisbon.id], to: folder))
        #expect(model.counts.folders[folder.id] == 1)
        #expect(NotesDragItem.ids([NotesDragItem.note(lisbon.id), "other"], prefix: NotesDragItem.notePrefix) == [lisbon.id])

        model.delete(folder)
        #expect(model.scope == .all)
        #expect(lisbon.folder == nil)
    }

    @Test func searchFiltersAndSaysSoWhenNothingMatches() throws {
        let model = try NotesFixtures.model()
        model.query = "alfama"
        // The list follows the field after a short pause, or on Return.
        #expect(model.notes.count == 24)
        model.commitQuery()
        #expect(model.notes.map(model.title(of:)) == ["Lisbon trip, March"])
        model.query = "nothing like this"
        model.commitQuery()
        #expect(model.isSearching)
        #expect(model.sections.isEmpty)
    }

    @Test func searchFollowsTheFieldAfterAPause() async throws {
        let model = try NotesFixtures.model()
        model.query = "dentist"
        #expect(await eventually { model.notes.count == 1 })
        #expect(model.selectedNote.map(model.title(of:)) == "Dentist questions")
    }

    @Test func permanentDeletionsAskFirst() async throws {
        let speech = FakeSpeechService()
        let model = try NotesFixtures.model(speech: speech)
        model.scope = .recentlyDeleted
        let first = try #require(model.notes.first)
        model.requestDeleteForever(first)
        #expect(model.pendingDeletion == .note(first.id, title: "Old draft 2"))
        #expect(model.pendingDeletion?.message == "Their recordings are deleted too. This can't be undone.")
        model.cancelDeletion()
        #expect(model.counts.recentlyDeleted == 2)
        model.requestEmptyTrash()
        #expect(model.pendingDeletion == .trash(count: 2))
        #expect(model.pendingDeletion?.title == "Delete 2 notes forever?")
        await model.confirmDeletion()
        #expect(model.counts.recentlyDeleted == 0)
        #expect(speech.calls.contains(.deleteRecordings(first.id)))
    }

    @Test func emptyDeletesOnlyTheNotesItAskedAbout() async throws {
        let model = try NotesFixtures.model()
        model.requestEmptyTrash()
        #expect(model.pendingDeletion == .trash(count: 2))
        // Another note is deleted while the question is open.
        let lisbon = try #require(model.selectedNote)
        model.delete(lisbon)
        await model.confirmDeletion()
        #expect(model.counts.recentlyDeleted == 1)
        #expect(try model.store.note(id: lisbon.id)?.deletedAt != nil)
    }

    @Test func deletingTheSelectedNoteForeverSelectsAnother() async throws {
        let model = try NotesFixtures.model()
        model.scope = .recentlyDeleted
        let selected = try #require(model.selectedNote)
        model.requestDeleteForever(selected)
        await model.confirmDeletion()
        #expect(model.selectedNote != nil)
        #expect(model.selectedNote?.id != selected.id)
    }

    @Test func selectionStaysOnAVisibleNote() throws {
        let model = try NotesFixtures.model()
        // A search that hides the selected note selects the first match.
        model.query = "kitchen"
        model.commitQuery()
        #expect(model.selectedNote.map(model.title(of:)) == "Kitchen shelf measurements")
        model.query = ""
        model.commitQuery()
        // Moving the selected note out of the open folder selects another.
        let personal = try #require(model.folders.first)
        model.scope = .folder(personal.id)
        let note = try #require(model.selectedNote)
        model.move(note, to: model.folders.last)
        #expect(model.selectedNote?.folder?.id == personal.id)
        // Deleting a note no longer in the list still selects one.
        model.scope = .all
        let lisbon = try #require(model.notes.first { model.title(of: $0) == "Lisbon trip, March" })
        model.select(lisbon)
        model.query = "groceries"
        model.commitQuery()
        model.select(lisbon)
        model.delete(lisbon)
        #expect(model.selectedNote.map(model.title(of:)) == "Weekly groceries")
    }

    @Test func theKeyboardMovesThroughTheListAndDeletes() throws {
        let model = try NotesFixtures.model()
        #expect(model.selectedNote.map(model.title(of:)) == "Lisbon trip, March")
        model.selectAdjacent(-1)
        #expect(model.selectedNote.map(model.title(of:)) == "Weekly groceries")
        model.selectAdjacent(-1)
        #expect(model.selectedNote.map(model.title(of:)) == "Weekly groceries")
        model.selectAdjacent(2)
        #expect(model.selectedNote.map(model.title(of:)) == "Design review: Notes")
        model.deleteSelected()
        #expect(model.counts.recentlyDeleted == 3)
        model.scope = .recentlyDeleted
        model.deleteSelected()
        #expect(model.pendingDeletion != nil)
    }

    @Test func foldersMoveByKeyboardAndByDrop() throws {
        let model = try NotesFixtures.model()
        let names = { model.folders.map(\.name) }
        let ideas = try #require(model.folders.last)
        model.moveFolder(ideas, by: -1)
        #expect(names() == ["Personal", "Ideas", "Work"])
        model.moveFolder(ideas, by: -5)
        #expect(names() == ["Ideas", "Personal", "Work"])
        // Dragged down onto Work: after it. Dragged up onto Ideas: before it.
        let personal = try #require(model.folders.first { $0.name == "Personal" })
        let work = try #require(model.folders.first { $0.name == "Work" })
        #expect(model.dropFolder(personal.id, onto: work.id))
        #expect(names() == ["Ideas", "Work", "Personal"])
        #expect(model.dropFolder(personal.id, onto: ideas.id))
        #expect(names() == ["Personal", "Ideas", "Work"])
        #expect(!model.dropFolder(work.id, onto: work.id))
    }

    @Test func duplicateSelectsTheCopy() throws {
        let model = try NotesFixtures.model()
        let lisbon = try #require(model.selectedNote)
        model.duplicate(lisbon)
        let copy = try #require(model.selectedNote)
        #expect(copy.id != lisbon.id)
        #expect(model.title(of: copy) == "Lisbon trip, March")
        #expect(model.counts.all == 25)
    }

    @Test func speechStatusCardFollowsTheService() async throws {
        let speech = FakeSpeechService(status: SpeechStatus(speech: .downloading(progress: 0.42)))
        let model = try NotesFixtures.model(speech: speech)
        #expect(SpeechStatusCard(status: model.speechStatus.speech).stateTitle == "Downloading 42%")
        #expect(SpeechStatusCard(status: .ready).stateTitle == "Ready")
        #expect(SpeechStatusCard(status: .failed(.offline)).stateTitle == "Needs attention")
    }

    @Test func dateLineReadsLikeTheBoard() {
        #expect(NoteEditorPane.dateLine(NotesFixtures.now, locale: NotesFixtures.locale,
                                        calendar: NotesFixtures.calendar) == "19 November 2026 at 11:40")
    }

    @Test func everySymbolTheScreenNamesExists() {
        for name in ["checklist", "checkmark", "doc", "folder", "magnifyingglass", "mic", "note.text", "paperclip", "pin", "pin.fill", "plus", "square.and.arrow.up", "square.and.pencil", "text.document", "trash", "waveform", "xmark.circle.fill"] {
            #expect(NSImage(systemSymbolName: name, accessibilityDescription: nil) != nil, "\(name)")
        }
    }

    @Test func theShellSlotShowsAnHonestStateWithoutAModel() {
        #expect(ImageRenderer(content: NotesRootView().frame(width: 600, height: 400)).cgImage != nil)
    }

    // MARK: G-A3

    @Test func notesMainBoardLightAndDark() async throws {
        let model = try NotesFixtures.model()
        for scheme in [ColorScheme.light, .dark] {
            let root = RootView(model: ShellModel(selection: .notes, defaults: nil), chrome: .inline)
                .environment(ScreenModels(notes: model))
            let url = try await HostedSnapshot.render(root, name: "notes-01-main", colorScheme: scheme)
            #expect(FileManager.default.fileExists(atPath: url.path(percentEncoded: false)))
        }
        #expect(model.editor.textView?.document == NotesFixtures.lisbon)
    }

    @Test func notesEmptyStatesRender() async throws {
        let store = NoteStore(container: try AppModelContainer.inMemory(), attachments: .temporary(),
                              speech: FakeSpeechService())
        let model = NotesModel(store: store, speech: FakeSpeechService(), now: { NotesFixtures.now })
        model.voice = NotesVoiceActions(transcribeFile: {}, recordMeeting: {}, toggleDictation: { _ in })
        let root = RootView(model: ShellModel(selection: .notes, defaults: nil), chrome: .inline)
            .environment(ScreenModels(notes: model))
        _ = try await HostedSnapshot.render(root, name: "notes-01-main-empty")
    }
}

/// Renders through an `NSHostingView` in a window (`cacheDisplay`), so the
/// AppKit editor draws; `ImageRenderer` would leave it blank. Same board
/// framing, size and scale as `Snapshot.render`, same output folder.
@MainActor
enum HostedSnapshot {
    static func render<V: View>(_ view: V, name: String, colorScheme: ColorScheme = .light) async throws -> URL {
        MFFontRegistry.registerBundledFonts()
        let canvas = Snapshot.canvas
        let content = Snapshot.framed(view, framing: .mockup, canvas: canvas)
            .environment(\.colorScheme, colorScheme)
        let hosting = NSHostingView(rootView: content)
        hosting.frame = CGRect(origin: .zero, size: canvas)
        let window = NSWindow(contentRect: hosting.frame, styleMask: [.borderless], backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        window.appearance = NSAppearance(named: colorScheme == .dark ? .darkAqua : .aqua)
        window.contentView = hosting
        // Let SwiftUI build the AppKit views and the editor load its note.
        for _ in 0..<5 {
            hosting.layoutSubtreeIfNeeded()
            try await Task.sleep(for: .milliseconds(40))
        }
        let rep = try #require(NSBitmapImageRep(bitmapDataPlanes: nil,
                                                pixelsWide: Int(canvas.width * Snapshot.scale),
                                                pixelsHigh: Int(canvas.height * Snapshot.scale),
                                                bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
                                                colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0))
        rep.size = canvas
        hosting.cacheDisplay(in: hosting.bounds, to: rep)
        let png = try #require(rep.representation(using: .png, properties: [:]))
        try FileManager.default.createDirectory(at: Snapshot.outputDirectory, withIntermediateDirectories: true)
        let url = Snapshot.outputDirectory.appendingPathComponent(name + (colorScheme == .dark ? "-dark" : "") + ".png")
        try png.write(to: url)
        window.close()
        return url
    }
}

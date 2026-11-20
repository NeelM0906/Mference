import Foundation
import SwiftData

/// The summary, decisions and action items written into a meeting or file
/// note (spec 4.5, 6.5). After they are written they are ordinary note
/// content: the person can edit or tick them.
public struct MeetingSections: Sendable, Hashable, Codable {
    public var summary: [String]
    public var decisions: [String]
    public var actionItems: [MeetingActionItem]

    public init(summary: [String], decisions: [String] = [], actionItems: [MeetingActionItem] = []) {
        self.summary = summary
        self.decisions = decisions
        self.actionItems = actionItems
    }
}

/// One action item: what to do and, when the meeting named one, who.
public struct MeetingActionItem: Sendable, Hashable, Codable {
    public var text: String
    public var owner: String?

    public init(text: String, owner: String? = nil) {
        self.text = text
        self.owner = owner
    }
}

/// An action-item checklist row as it stands in the note now.
public struct MeetingActionItemRow: Sendable, Hashable {
    public var item: MeetingActionItem
    public var isDone: Bool

    public init(item: MeetingActionItem, isDone: Bool) {
        self.item = item
        self.isDone = isDone
    }
}

/// How the meeting layer writes into a note's body and title. Notes (N1)
/// owns the document; `NoteStore` conforms (in `Meetings/`), so the meeting
/// layer never edits `bodyJSON` itself. Every call saves and keeps the note's
/// `updatedAt` and `searchText` right. Calls for a note that no longer exists
/// do nothing.
@MainActor
public protocol MeetingNoteWriting: AnyObject {
    /// Creates an empty note of `kind` (`.meeting` or `.file`) in the
    /// writer's model context, so `MeetingSession` can attach its recording.
    func createNote(id: UUID, kind: NoteKind, title: String, createdAt: Date) throws
    func setTitle(_ title: String, of noteID: UUID) throws
    /// Writes Summary (bullets), Decisions (bullets, only when there are
    /// some) and Action items (checklist rows, "· owner" after the text) at
    /// the top of the body, replacing the ones written before (Redo); the
    /// person's own notes below stay.
    func replaceMeetingSections(_ sections: MeetingSections, in noteID: UUID) throws
    /// The Action items checklist rows as they are now, in order.
    func actionItems(in noteID: UUID) -> [MeetingActionItemRow]
    /// Adds a paragraph at the end of the body ("Recording ended unexpectedly at 10:42").
    func appendParagraph(_ text: String, to noteID: UUID) throws
    /// Adds a Mark moment marker at the end of the body.
    func appendMarker(at time: TimeInterval, to noteID: UUID) throws
    /// The transcript changed: refresh the note's search text.
    func transcriptDidChange(in noteID: UUID)
}

/// A `MeetingNoteWriting` that keeps the written content in memory and the
/// note rows in a SwiftData context: tests, previews, and the app until
/// `NoteStore` conforms.
@MainActor
public final class InMemoryMeetingNoteWriter: MeetingNoteWriting {
    public struct Body: Sendable, Hashable {
        public var sections: MeetingSections?
        public var actionItems: [MeetingActionItemRow] = []
        public var paragraphs: [String] = []
        public var markers: [TimeInterval] = []
    }

    public let context: ModelContext
    public private(set) var bodies: [UUID: Body] = [:]
    public private(set) var searchRefreshes: [UUID: Int] = [:]
    /// The next write throws this (once).
    public var failNextWrite: (any Error)?

    public init(context: ModelContext) {
        self.context = context
    }

    public func body(of noteID: UUID) -> Body { bodies[noteID] ?? Body() }

    /// Ticks (or unticks) an action item, as the person would in the editor.
    public func setActionItem(at index: Int, done: Bool, in noteID: UUID) {
        guard bodies[noteID]?.actionItems.indices.contains(index) == true else { return }
        bodies[noteID]?.actionItems[index].isDone = done
    }

    public func createNote(id: UUID, kind: NoteKind, title: String, createdAt: Date) throws {
        try takeFailure()
        context.insert(Note(id: id, title: title, kind: kind, createdAt: createdAt))
        try context.save()
        bodies[id] = Body()
    }

    public func setTitle(_ title: String, of noteID: UUID) throws {
        try takeFailure()
        guard let note = note(noteID) else { return }
        note.title = title
        note.updatedAt = .now
        try context.save()
    }

    public func replaceMeetingSections(_ sections: MeetingSections, in noteID: UUID) throws {
        try takeFailure()
        guard note(noteID) != nil else { return }
        bodies[noteID, default: Body()].sections = sections
        bodies[noteID, default: Body()].actionItems = sections.actionItems.map {
            MeetingActionItemRow(item: $0, isDone: false)
        }
    }

    public func actionItems(in noteID: UUID) -> [MeetingActionItemRow] { body(of: noteID).actionItems }

    public func appendParagraph(_ text: String, to noteID: UUID) throws {
        try takeFailure()
        guard note(noteID) != nil else { return }
        bodies[noteID, default: Body()].paragraphs.append(text)
    }

    public func appendMarker(at time: TimeInterval, to noteID: UUID) throws {
        try takeFailure()
        guard note(noteID) != nil else { return }
        bodies[noteID, default: Body()].markers.append(time)
    }

    public func transcriptDidChange(in noteID: UUID) {
        searchRefreshes[noteID, default: 0] += 1
    }

    private func note(_ id: UUID) -> Note? {
        try? context.fetch(FetchDescriptor<Note>(predicate: #Predicate { $0.id == id })).first
    }

    private func takeFailure() throws {
        if let error = failNextWrite {
            failNextWrite = nil
            throw error
        }
    }
}

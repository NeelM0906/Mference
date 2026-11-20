import Foundation

/// Meeting notes (Notes spec 4.4, 4.5): what the meeting layout (N3) asks of
/// the Notes editor (N1). `NoteEditorController` conforms.
@MainActor
public protocol MeetingNoteTarget: AnyObject {
    /// The note the editor shows.
    var noteID: UUID { get }
    /// "Mark moment" (⇧⌘M): a marker line after the caret's line at
    /// recording time `time`, only when `noteID` is the note shown (and
    /// editable). Returns false otherwise: the caller then writes the marker
    /// to that note through the store, so another note is never marked.
    @discardableResult
    func insertMarker(time: TimeInterval, in noteID: UUID) -> Bool
    /// A time chip for a transcript segment after the caret's line, tying
    /// the person's note to that moment; same rule as `insertMarker`.
    @discardableResult
    func insertTranscriptAnchor(segmentID: UUID, in noteID: UUID) -> Bool
    /// Scrolls to and selects a segment's time chip (a ⌘K hit, a click in
    /// the transcript); nothing happens when the note has none.
    func reveal(segmentID: UUID)
    /// Called with the recording time of a clicked time chip or marker, so
    /// the transcript scrolls or the player seeks there.
    var onTimeChipClick: ((TimeInterval) -> Void)? { get set }
}

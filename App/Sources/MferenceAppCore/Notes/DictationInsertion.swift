import Foundation

// Agreed between N1 (notes editor) and N3 (dictation), 2026-11-20. N1 owns
// this file; N3's DictationController drives it. (Carried on N3's branch
// until N1's copy lands; identical.)

/// Where dictated phrases land: the Notes editor conforms. Each phrase is
/// held by a pending marker from the moment speech starts, so the person can
/// keep typing and moving the caret (Notes spec 4.2).
@MainActor
public protocol DictationInsertionTarget: AnyObject {
    var noteID: UUID { get }
    /// Holds the caret position for phrase `id`: inserts a pending,
    /// accent-tinted marker there. Typing and caret moves around it are safe.
    func beginPhrase(_ id: UUID)
    /// Replaces the phrase's marker with `text` joined via PhraseJoiner
    /// (spacing, capitals); sets Note.hasDictation; autosaves.
    func finishPhrase(_ id: UUID, text: String)
    /// Removes the phrase's marker (cough / no words).
    func dropPhrase(_ id: UUID)
    /// Removes every pending marker (session ended, failed, note switched).
    func dropAllPhrases()
    /// A dictation is starting into this target: every phrase of it belongs
    /// to the note shown now (`noteID`), whatever the person opens next,
    /// until `dropAllPhrases`. Called by the dictation controller as it
    /// records `noteID` as the active note.
    func dictationWillStart()
}

extension DictationInsertionTarget {
    /// Targets that keep no binding need nothing here.
    public func dictationWillStart() {}
}

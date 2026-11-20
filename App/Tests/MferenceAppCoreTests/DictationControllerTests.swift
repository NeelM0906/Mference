import Foundation
@testable import MferenceAppCore
import Testing

/// A note body as the editor would keep it: text with pending markers, and
/// a caret. Markers stay where they were put while the caret moves on.
@MainActor
final class FakeDictationTarget: DictationInsertionTarget {
    let noteID: UUID
    /// The body, with each pending phrase as a "⟦id⟧" marker.
    private(set) var text: String
    var caret: String.Index
    private(set) var markers: [UUID: String] = [:]
    private(set) var dictated = false

    init(noteID: UUID = UUID(), text: String = "") {
        self.noteID = noteID
        self.text = text
        caret = text.endIndex
    }

    var visibleText: String {
        markers.values.reduce(text) { $0.replacingOccurrences(of: $1, with: "") }
    }

    /// Types at the caret (the person keeps typing while dictating).
    func type(_ string: String) {
        let offset = text.distance(from: text.startIndex, to: caret)
        text.insert(contentsOf: string, at: caret)
        caret = text.index(text.startIndex, offsetBy: offset + string.count)
    }

    func moveCaretToEnd() { caret = text.endIndex }

    func beginPhrase(_ phrase: UUID) {
        let marker = "⟦\(phrase.uuidString.prefix(8))⟧"
        markers[phrase] = marker
        let offset = text.distance(from: text.startIndex, to: caret)
        text.insert(contentsOf: marker, at: caret)
        // The caret stays after the marker, so typing goes on after it.
        caret = text.index(text.startIndex, offsetBy: offset + marker.count)
    }

    private func surroundings(of phrase: UUID) -> TestSurroundings? {
        guard let marker = markers[phrase], let range = text.range(of: marker) else { return nil }
        let lineStart = text[..<range.lowerBound].lastIndex(of: "\n").map { text.index(after: $0) } ?? text.startIndex
        var before = String(text[lineStart..<range.lowerBound])
        for other in markers.values { before = before.replacingOccurrences(of: other, with: "") }
        let after = text[range.upperBound...].first
        return TestSurroundings(lineBefore: before, after: after == "\n" ? nil : after)
    }

    func finishPhrase(_ phrase: UUID, text raw: String) {
        guard let surroundings = surroundings(of: phrase) else { return }
        let replacement = TestPhraseJoiner.join(raw, in: surroundings)
        guard let marker = markers.removeValue(forKey: phrase), let range = text.range(of: marker) else { return }
        let caretOffset = text.distance(from: text.startIndex, to: caret)
        let markerOffset = text.distance(from: text.startIndex, to: range.lowerBound)
        text.replaceSubrange(range, with: replacement)
        let shift = caretOffset > markerOffset ? replacement.count - marker.count : 0
        caret = text.index(text.startIndex, offsetBy: caretOffset + shift)
        dictated = true
    }

    func dropAllPhrases() {
        for id in Array(markers.keys) { dropPhrase(id) }
    }

    func dropPhrase(_ phrase: UUID) {
        guard let marker = markers.removeValue(forKey: phrase), let range = text.range(of: marker) else { return }
        let caretOffset = text.distance(from: text.startIndex, to: caret)
        let markerOffset = text.distance(from: text.startIndex, to: range.lowerBound)
        text.removeSubrange(range)
        caret = text.index(text.startIndex, offsetBy: caretOffset > markerOffset ? caretOffset - marker.count : caretOffset)
    }
}

/// Lets the controller's consumer task handle what the fake just yielded.
@MainActor
func settle() async {
    for _ in 0..<20 { await Task.yield() }
}

/// Plays one scripted event and lets the controller handle it.
@MainActor
func step(_ speech: FakeSpeechService, _ count: Int = 1) async {
    for _ in 0..<count {
        speech.advance()
        await settle()
    }
}

@MainActor
@Suite struct DictationControllerTests {
    private func controller(_ speech: FakeSpeechService, broker: FakePermissionBroker = FakePermissionBroker(),
                            preferences: SpeechPreferences = SpeechPreferences(store: InMemorySettingsStore()))
        -> DictationController {
        DictationController(speech: speech, permissions: broker, preferences: preferences)
    }

    @Test func phrasesLandAtTheCaretJoinedNaturally() async {
        let speech = FakeSpeechService(dictation: .saying("ask about the sensitivity.", "And whether it can move"))
        let dictation = controller(speech)
        let note = FakeDictationTarget(text: "Questions for Dr. Hale.")
        await dictation.start(in: note)
        await dictation.stop()
        #expect(note.visibleText == "Questions for Dr. Hale. Ask about the sensitivity. And whether it can move")
        #expect(note.dictated)
        #expect(!dictation.isActive && dictation.activeNoteID == nil)
        #expect(speech.calls.contains(.startDictation(DictationOptions())))
    }

    @Test func eachPhraseGoesWhereTheCaretWasWhenItStarted() async throws {
        let speech = FakeSpeechService(dictation: .saying("first phrase", "second phrase"), stepped: true)
        let dictation = controller(speech)
        let note = FakeDictationTarget(text: "Start.\n")
        await dictation.start(in: note)
        await step(speech) // level
        await step(speech) // phrase 1 started at "Start.\n|"
        note.type("typed while talking")
        await step(speech) // phrase 1 finished
        note.moveCaretToEnd()
        note.type("\n")
        await step(speech) // phrase 2 started on the new line
        await step(speech) // phrase 2 finished
        await dictation.stop()
        #expect(note.visibleText == "Start.\nFirst phrase typed while talking\nSecond phrase")
    }

    @Test func stopFinishesTheLastPhraseFirst() async {
        let speech = FakeSpeechService(dictation: .saying("one", "two"), stepped: true)
        let dictation = controller(speech)
        let note = FakeDictationTarget()
        await dictation.start(in: note)
        await step(speech, 2) // "one" started, still pending
        #expect(dictation.pendingPhrases.count == 1)
        await dictation.stop()
        #expect(note.visibleText == "One two", "the phrase in progress finishes before dictation lets go")
        #expect(dictation.pendingPhrases.isEmpty)
        #expect(speech.calls.filter { $0 == .stopDictation }.count == 1)
    }

    @Test func aCoughLeavesNoMarker() async {
        let speech = FakeSpeechService(dictation: FakeSpeechService.DictationScript(phrases: [(nil, nil), ("hello", "en")]))
        let dictation = controller(speech)
        let note = FakeDictationTarget()
        await dictation.start(in: note)
        await dictation.stop()
        #expect(note.text == "Hello")
    }

    @Test func toggleStopsInTheSameNoteAndSwitchingNotesFinishesTheOldOne() async {
        let speech = FakeSpeechService(dictation: .saying("hello there"), stepped: true)
        let dictation = controller(speech)
        let first = FakeDictationTarget(text: "A.")
        let second = FakeDictationTarget(text: "B.")
        await dictation.toggle(first)
        #expect(dictation.isDictating(into: first.noteID))
        await step(speech, 2) // phrase started in the first note
        await dictation.toggle(second) // switching notes
        #expect(first.visibleText == "A. Hello there", "the old note gets its last phrase")
        #expect(dictation.isDictating(into: second.noteID))
        await dictation.toggle(second)
        #expect(!dictation.isActive)
    }

    @Test func deniedMicrophoneSaysSo() async {
        let speech = FakeSpeechService()
        let broker = FakePermissionBroker(answers: [.microphone: .denied])
        let dictation = controller(speech, broker: broker)
        await dictation.start(in: FakeDictationTarget())
        #expect(dictation.error == .permissionDenied(.microphone))
        #expect(!dictation.isActive)
        #expect(!speech.calls.contains { if case .startDictation = $0 { true } else { false } })
        #expect(broker.calls == [.request(.microphone)])
    }

    @Test func busyWhileAMeetingRecords() async throws {
        let speech = FakeSpeechService()
        _ = try await speech.startMeeting(MeetingOptions(noteID: UUID()))
        let dictation = controller(speech)
        await dictation.start(in: FakeDictationTarget())
        #expect(dictation.error == .busy(.meeting))
        #expect(dictation.error?.message == "A meeting is recording. Stop it first.")
        #expect(!dictation.isActive)
    }

    @Test func notInstalledFailsPlainly() async {
        let dictation = controller(FakeSpeechService.notInstalled())
        await dictation.start(in: FakeDictationTarget())
        #expect(dictation.error == .notInstalled(.speech))
    }

    @Test func chipsWriteTheDefaults() async {
        let speech = FakeSpeechService(stepped: true)
        let preferences = SpeechPreferences(store: InMemorySettingsStore())
        let dictation = controller(speech, preferences: preferences)
        await dictation.start(in: FakeDictationTarget())
        await dictation.setLanguage(.fixed("de"))
        dictation.setCleanUpFillers(false)
        #expect(preferences.language == .fixed("de") && !preferences.cleanUpFillers)
        #expect(speech.calls.contains(.setLanguage(.fixed("de"))))
        await dictation.stop()
    }

    @Test func detectedLanguageAndLevelFollowTheSession() async {
        let speech = FakeSpeechService(dictation: FakeSpeechService.DictationScript(phrases: [("Hallo", "de")]),
                                       stepped: true)
        let dictation = controller(speech)
        await dictation.start(in: FakeDictationTarget())
        await step(speech)
        #expect(dictation.level == 0.4)
        await step(speech, 2)
        #expect(dictation.detectedLanguage == "de")
        await dictation.stop()
        #expect(dictation.level == 0)
    }
}

/// Checks the test notes' joiner (N1's PhraseJoiner has its own tests).
@Suite struct TestPhraseJoinerTests {
    private func join(_ phrase: String, _ before: String, _ after: Character? = nil) -> String {
        TestPhraseJoiner.join(phrase, in: TestSurroundings(lineBefore: before, after: after))
    }

    @Test func spacesBetweenPhrasesNeverDoubled() {
        #expect(join("and more", "Some text") == " and more")
        #expect(join("and more", "Some text ") == "and more", "no double space")
        #expect(join("  spaced   out  ", "Text") == " spaced out")
    }

    @Test func noLeadingSpaceAtALineStart() {
        #expect(join("hello", "") == "Hello")
        #expect(join("hello", "   ") == "Hello")
    }

    @Test func capitalsWhereASentenceStarts() {
        #expect(join("next sentence", "Done.") == " Next sentence")
        #expect(join("next sentence", "Done. ") == "Next sentence")
        #expect(join("really", "Why?") == " Really")
        #expect(join("And whether it moves", "Ask about the sensitivity") == " and whether it moves",
                "a phrase continuing a sentence goes on in lower case")
        #expect(join("Lisbon in March", "Flights to") == " Lisbon in March", "names keep their capital")
        #expect(join("I think so", "Well,") == " I think so")
    }

    @Test func textThatFollowsGetsItsSpace() {
        #expect(join("very", "It is", "g") == " very ")
        #expect(join("done", "It is", ".") == " done")
        #expect(join(", then", "First") == ", then", "no space before closing punctuation")
        #expect(join("aside", "Note (") == "aside", "no space after an opening bracket")
    }

    @Test func emptyPhrasesInsertNothing() {
        #expect(join("   ", "Text") == "")
    }
}

import Foundation
@testable import MferenceAppCore
import Testing

/// The dictation controller tells its target when a dictation starts, before
/// any phrase, so the Notes editor binds the dictation to the note shown then.
@MainActor
@Suite struct DictationStartBindingTests {
    @MainActor
    final class StartTarget: DictationInsertionTarget {
        let noteID = UUID()
        private(set) var calls: [String] = []
        func dictationWillStart() { calls.append("start") }
        func beginPhrase(_ id: UUID) { calls.append("begin") }
        func finishPhrase(_ id: UUID, text: String) { calls.append("finish") }
        func dropPhrase(_ id: UUID) { calls.append("drop") }
        func dropAllPhrases() { calls.append("dropAll") }
    }

    @Test func startIsAnnouncedBeforeAnyPhrase() async {
        let speech = FakeSpeechService(dictation: .saying("one"), stepped: true)
        let controller = DictationController(speech: speech,
                                             permissions: FakePermissionBroker(states: [.microphone: .granted]),
                                             preferences: SpeechPreferences(store: InMemorySettingsStore()))
        let target = StartTarget()
        await controller.start(in: target)
        #expect(target.calls == ["start"])
        #expect(controller.activeNoteID == target.noteID)
        await controller.stop()
        #expect(target.calls.first == "start")
        #expect(target.calls.filter { $0 == "start" }.count == 1)
    }

    @Test func targetsWithoutABindingNeedNothing() {
        // The protocol's default does nothing (doubles need not implement it).
        final class Plain: DictationInsertionTarget {
            let noteID = UUID()
            func beginPhrase(_ id: UUID) {}
            func finishPhrase(_ id: UUID, text: String) {}
            func dropPhrase(_ id: UUID) {}
            func dropAllPhrases() {}
        }
        Plain().dictationWillStart()
    }
}

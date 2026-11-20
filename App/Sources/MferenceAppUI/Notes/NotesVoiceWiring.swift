import MferenceAppCore
import SwiftUI

extension NotesModel {
    /// Hooks Notes to the voice team's models (N3): the shell bar's voice
    /// toolbar, the Dictate button, ⇧⌘D's focused editor, Esc while
    /// dictating, stopping dictation when the note changes, and files
    /// dropped on Notes.
    func connect(_ voice: VoiceModels) {
        let editor = self.editor
        self.voice = NotesVoiceActions(
            transcribeFile: { [weak voice] in voice?.transcribeFiles() },
            recordMeeting: { [weak voice] in voice?.recordMeeting() },
            toggleDictation: { [weak voice] target in voice?.toggleDictation(in: target) },
            toolbar: { [weak voice] in voice.map { AnyView(NotesVoiceToolbar(voice: $0)) } ?? AnyView(EmptyView()) },
            isDictating: { [weak voice] noteID in voice?.dictation.isDictating(into: noteID) ?? false },
            stopDictation: { [weak voice] in
                guard let voice else { return }
                Task { await voice.stopDictation() }
            },
            editorFocus: { [weak voice] target in voice?.focusedDictationTarget = target },
            transcribe: { [weak voice] urls in voice?.transcribe(urls) ?? false })
        editor.onFocusChange = { [weak self, weak editor] focused in
            self?.voice.editorFocus?(focused ? editor : nil)
        }
        // Esc while dictating stops it (the bar's Done may not get Esc while
        // the text has focus).
        editor.onCancel = { [weak voice] in
            guard let voice, voice.dictation.isActive else { return false }
            Task { await voice.dictation.stop() }
            return true
        }
    }
}

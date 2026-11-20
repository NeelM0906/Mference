import MferenceAppCore
import SwiftUI

/// The voice menu commands (Notes spec D10), in the Edit menu: Dictate ⇧⌘D
/// for the note editor that has focus, Mark Moment ⇧⌘M only while a meeting
/// records, Record Meeting ⇧⌘R while none records (it opens Notes), and
/// Transcribe Audio File ⇧⌘O only while Notes shows. The toolbar buttons
/// bind no keys of their own. None clash with ⌘N, ⌘K, ⌘1–⌘7 or ⌘,.
public struct VoiceCommands: Commands {
    let voice: VoiceModels
    let shell: ShellModel

    public init(voice: VoiceModels, shell: ShellModel) {
        self.voice = voice
        self.shell = shell
    }

    public var body: some Commands {
        CommandGroup(after: .textEditing) {
            Divider()
            VoiceMenuItems(voice: voice, shell: shell)
        }
    }
}

/// The voice menu items, as a view so they observe `VoiceModels` like any
/// view (focus moving into a note re-enables Dictate) and can be rendered
/// in tests.
struct VoiceMenuItems: View {
    let voice: VoiceModels
    let shell: ShellModel

    var body: some View {
        Button(voice.dictation.isActive ? "Stop Dictating" : "Dictate") { voice.toggleDictation() }
            .keyboardShortcut("d", modifiers: [.command, .shift])
            .disabled(!voice.canToggleDictation)
        Button("Mark Moment") { voice.markMoment() }
            .keyboardShortcut("m", modifiers: [.command, .shift])
            .disabled(!voice.canMarkMoment)
        Button("Record Meeting…") {
            shell.show(.notes)
            voice.recordMeeting()
        }
        .keyboardShortcut("r", modifiers: [.command, .shift])
        .disabled(!voice.canRecordMeeting)
        Button("Transcribe Audio File…") { voice.transcribeFiles() }
            .keyboardShortcut("o", modifiers: [.command, .shift])
            .disabled(shell.selection != .notes)
    }
}

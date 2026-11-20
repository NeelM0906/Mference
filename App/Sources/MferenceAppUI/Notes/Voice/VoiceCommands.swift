import MferenceAppCore
import SwiftUI

/// The voice menu commands (Notes spec D10), in the Edit menu: Dictate ⇧⌘D
/// for the note editor that has focus, and Mark Moment ⇧⌘M only while a
/// meeting records. Record meeting ⇧⌘R and Transcribe file ⇧⌘O belong to
/// the Notes toolbar buttons, so they work only inside Notes. None clash
/// with ⌘N, ⌘K, ⌘1–⌘7 or ⌘,.
public struct VoiceCommands: Commands {
    let voice: VoiceModels

    public init(voice: VoiceModels) { self.voice = voice }

    public var body: some Commands {
        CommandGroup(after: .textEditing) {
            Divider()
            Button(voice.dictation.isActive ? "Stop Dictating" : "Dictate") { voice.toggleDictation() }
                .keyboardShortcut("d", modifiers: [.command, .shift])
                .disabled(!voice.canToggleDictation)
            Button("Mark Moment") { voice.markMoment() }
                .keyboardShortcut("m", modifiers: [.command, .shift])
                .disabled(!voice.canMarkMoment)
        }
    }
}

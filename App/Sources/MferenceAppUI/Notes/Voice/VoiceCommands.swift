import MferenceAppCore
import SwiftUI

/// The voice menu commands (Notes spec D10): Dictate ⇧⌘D in the Edit menu,
/// for the note editor that has focus.
public struct VoiceCommands: Commands {
    let voice: VoiceModels

    public init(voice: VoiceModels) { self.voice = voice }

    public var body: some Commands {
        CommandGroup(after: .textEditing) {
            Divider()
            Button(voice.dictation.isActive ? "Stop Dictating" : "Dictate") { voice.toggleDictation() }
                .keyboardShortcut("d", modifiers: [.command, .shift])
        }
    }
}

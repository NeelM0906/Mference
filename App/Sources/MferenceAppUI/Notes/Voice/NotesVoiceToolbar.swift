import MferenceAppCore
import MferenceDesign
import SwiftUI

/// Notes' voice actions in the chrome (boards 1–3, 5–7): Record meeting
/// with its popover (Transcribe file joins it with the import sheet). N1 puts it in Notes' `shellToolbar`
/// before New note. Both go through the setup card on first use. While a
/// meeting records, Record meeting steps back (the red timer chip shows).
public struct NotesVoiceToolbar: View {
    @Bindable var voice: VoiceModels

    public init(voice: VoiceModels) { self.voice = voice }

    public var body: some View {
        HStack(spacing: MFSpacing.s) {
            if voice.liveMeeting == nil {
                Button {
                    voice.recordMeeting()
                } label: {
                    HStack(spacing: MFSpacing.snug) {
                        Circle().fill(MFColor.danger)
                            .frame(width: RecordMetrics.dot, height: RecordMetrics.dot)
                            .accessibilityHidden(true)
                        Text("Record meeting")
                    }
                }
                .buttonStyle(.mfSecondary)
                .keyboardShortcut("r", modifiers: [.command, .shift])
                .help("Record a meeting (⇧⌘R)")
                .popover(isPresented: $voice.showsRecordPopover, arrowEdge: .bottom) {
                    RecordMeetingPopover(model: voice.recordMeetingModel) { voice.showsRecordPopover = false }
                }
            }
        }
    }
}

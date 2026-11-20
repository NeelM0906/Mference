import MferenceAppCore
import MferenceDesign
import SwiftUI

/// Notes' voice actions in the chrome (boards 1–3, 5–7): Transcribe file
/// with the import sheet, and Record meeting with its popover. N1 puts it in Notes' `shellToolbar`
/// before New note. Both go through the setup card on first use. While a
/// meeting records, Record meeting steps back (the red timer chip shows).
public struct NotesVoiceToolbar: View {
    @Bindable var voice: VoiceModels

    public init(voice: VoiceModels) { self.voice = voice }

    public var body: some View {
        HStack(spacing: MFSpacing.s) {
            Button {
                voice.transcribeFiles()
            } label: {
                Label("Transcribe file", systemImage: "doc.text")
            }
            .buttonStyle(.mfSecondary)
            .help("Transcribe audio files (⇧⌘O, in the Edit menu)")
            .sheet(isPresented: $voice.showsImportSheet) {
                TranscribeFilesSheet(queue: voice.fileQueue, preferences: voice.preferences,
                                     onOpenNote: { noteID in
                                         voice.showsImportSheet = false
                                         voice.openNote?(noteID)
                                     },
                                     onClose: { voice.showsImportSheet = false })
            }
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
                .help("Record a meeting (⇧⌘R, in the Edit menu)")
                .popover(isPresented: $voice.showsRecordPopover, arrowEdge: .bottom) {
                    RecordMeetingPopover(model: voice.recordMeetingModel) { voice.showsRecordPopover = false }
                }
            }
        }
    }
}

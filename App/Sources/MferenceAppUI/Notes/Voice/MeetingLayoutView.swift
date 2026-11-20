import MferenceAppCore
import MferenceDesign
import SwiftUI

/// The meeting layout (board 4, spec 4.4): the recording bar, then your
/// notes (540 pt) beside the live transcript. NotesRootView (N1) shows it
/// for the live meeting note, with the sidebar and list collapsed, and
/// passes the note editor as `notes`. The editor's time chips call
/// `onTimeChip`, which scrolls the transcript to that moment.
public struct MeetingLayoutView<Notes: View>: View {
    let session: MeetingSession
    let voice: VoiceModels
    let onShowSidebar: () -> Void
    let notes: (_ onTimeChip: @escaping (TimeInterval) -> Void) -> Notes
    @State private var scrollTarget: TimeInterval?

    public init(session: MeetingSession, voice: VoiceModels, onShowSidebar: @escaping () -> Void,
                @ViewBuilder notes: @escaping (_ onTimeChip: @escaping (TimeInterval) -> Void) -> Notes) {
        self.session = session
        self.voice = voice
        self.onShowSidebar = onShowSidebar
        self.notes = notes
    }

    public var body: some View {
        VStack(spacing: 0) {
            RecordingBar(session: session, live: voice.liveMeeting,
                         microphoneName: voice.recordMeetingModel.microphoneName,
                         callSourceName: Self.callSourceName(voice.preferences.callAudio),
                         summaryOn: voice.preferences.writesMeetingSummary,
                         onToggleSidebar: onShowSidebar)
            if let message = Self.notice(session) {
                Label(message, systemImage: "exclamationmark.circle")
                    .font(MFFont.preview)
                    .foregroundStyle(MFColor.preview)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal, LiveMetrics.barPadding)
                    .padding(.vertical, MFSpacing.s)
                    .background(MFColor.previewFill)
            }
            HStack(spacing: 0) {
                notes { time in scrollTarget = time }
                    .padding(.horizontal, LiveMetrics.notesPadding.width)
                    .padding(.vertical, LiveMetrics.notesPadding.height)
                    .frame(width: LiveMetrics.notesWidth)
                    .frame(maxHeight: .infinity, alignment: .top)
                    .background(MFColor.surface)
                Rectangle().fill(MFColor.chromeDivider).frame(width: MFLayout.hairline)
                if let transcript = session.transcript {
                    LiveTranscriptView(transcript: transcript, scrollTarget: scrollTarget)
                } else {
                    Color.clear.background(MFColor.canvas)
                }
            }
        }
        .task { await voice.recordMeetingModel.load() }
    }

    /// The bar's call-audio name; nil when call audio is off.
    static func callSourceName(_ source: CallAudioSource) -> String? {
        switch source {
        case .off: nil
        case .allApps: "All apps"
        case .app(_, let name): name
        }
    }

    /// A line under the bar: the session's last problem.
    static func notice(_ session: MeetingSession) -> String? {
        session.lastError
    }
}

// MARK: - Controls for the menu bar and quit

extension MeetingSession: LiveRecordingControls {
    public func pauseRecording() async { await pause() }
    public func resumeRecording() async { await resume() }
    public func markMomentInRecording() async { await markMoment() }

    /// Stops and waits until the transcript is saved (quit waits on it).
    public func stopRecording() async {
        await stop()
        await waitUntilFinished()
    }
}

extension VoiceModels {
    /// Connects the voice screens to N4's `MeetingSession` (the app calls it
    /// once `AppServices.meetings` exists): the popover starts meetings
    /// through it, and the menu bar and quit control it.
    public func connect(meetings: MeetingSession) {
        recordingControls = meetings
        startMeeting = { request in
            try await meetings.start(MeetingStartOptions(
                microphoneID: request.microphoneID, callAudio: request.callAudio,
                speakerLabels: request.speakerLabels, keepAudio: request.keepAudio,
                language: request.language, summaryWhenStopped: request.summaryWhenYouStop))
        }
    }
}

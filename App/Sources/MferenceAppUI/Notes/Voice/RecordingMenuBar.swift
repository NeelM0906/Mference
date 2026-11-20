import MferenceAppCore
import MferenceDesign
import SwiftUI

/// What the menu bar item and the quit confirmation can do to the meeting
/// that is recording. N4's `MeetingSession` provides it (through an adapter
/// in `VoiceModels`); tests use a double.
@MainActor
public protocol LiveRecordingControls: AnyObject {
    func pauseRecording() async
    func resumeRecording() async
    /// Mark moment (⇧⌘M): a marker line at the recording time.
    func markMomentInRecording() async
    /// Stop and summarize (the menu bar): a normal stop; the last phrases,
    /// the speaker pass and the summary follow as usual.
    func stopRecording() async
    /// Quit: the session ends the meeting through the speech service's quick
    /// shutdown (finishing deferred to the next launch), saves that end and
    /// leaves the summary to the next launch.
    func finishForQuit() async
}

/// The menu bar item's menu while a meeting records (D7): the time, the
/// call-audio hint when there is one, Open, Pause or Resume, Stop.
public struct RecordingMenuContent: View {
    let live: LiveSessionStatus
    let controls: (any LiveRecordingControls)?
    let callAudioHint: CallAudioHintActions?
    let onOpen: () -> Void

    /// The hint line's text and actions, from `VoiceModels`.
    public struct CallAudioHintActions {
        let message: String
        let openSettings: () -> Void
        let dismiss: () -> Void

        public init(message: String, openSettings: @escaping () -> Void, dismiss: @escaping () -> Void) {
            self.message = message
            self.openSettings = openSettings
            self.dismiss = dismiss
        }
    }

    public init(live: LiveSessionStatus, controls: (any LiveRecordingControls)?,
                callAudioHint: CallAudioHintActions? = nil, onOpen: @escaping () -> Void) {
        self.live = live
        self.controls = controls
        self.callAudioHint = callAudioHint
        self.onOpen = onOpen
    }

    public var body: some View {
        Text(Self.title(live, at: .now))
        if let callAudioHint {
            Divider()
            Text(callAudioHint.message)
            Button("Open Call Audio in System Settings…", action: callAudioHint.openSettings)
            Button("Hide This Hint", action: callAudioHint.dismiss)
        }
        Divider()
        Button("Open the meeting note", action: onOpen)
        if live.isPaused {
            Button("Resume") { Task { await controls?.resumeRecording() } }
                .disabled(controls == nil)
        } else {
            Button("Pause") { Task { await controls?.pauseRecording() } }
                .disabled(controls == nil)
        }
        Button("Stop and summarize") { Task { await controls?.stopRecording() } }
            .disabled(controls == nil)
    }

    static func title(_ live: LiveSessionStatus, at now: Date) -> String {
        (live.isPaused ? "Paused · " : "Recording · ") + VoiceFormat.clock(live.recordedTime(at: now))
    }
}

/// The menu bar label: a red dot and the time.
public struct RecordingMenuLabel: View {
    let live: LiveSessionStatus

    public init(live: LiveSessionStatus) { self.live = live }

    public var body: some View {
        TimelineView(.periodic(from: .now, by: 1)) { context in
            Label(VoiceFormat.clock(live.recordedTime(at: context.date)),
                  systemImage: live.isPaused ? "pause.circle" : "record.circle")
                .monospacedDigit()
        }
    }
}

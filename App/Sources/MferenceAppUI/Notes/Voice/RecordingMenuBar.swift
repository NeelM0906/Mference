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
    /// Stop and summarize (the summary runs only if it was asked for).
    func stopRecording() async
}

/// The menu bar item's menu while a meeting records (D7): the time, Open,
/// Pause or Resume, Stop.
public struct RecordingMenuContent: View {
    let live: LiveSessionStatus
    let controls: (any LiveRecordingControls)?
    let onOpen: () -> Void

    public init(live: LiveSessionStatus, controls: (any LiveRecordingControls)?, onOpen: @escaping () -> Void) {
        self.live = live
        self.controls = controls
        self.onOpen = onOpen
    }

    public var body: some View {
        Text(Self.title(live, at: .now))
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

import MferenceAppCore
import MferenceDesign
import SwiftUI

/// The red recording timer chip (D7, board 4 `.recchip`): in the chrome on
/// every section while a meeting records; clicking it opens the meeting
/// note. It reads `SpeechStatus.liveSession`, so it keeps time with the
/// speech service whatever screen is showing.
public struct RecordingTimerChip: View {
    let live: LiveSessionStatus
    let onOpen: () -> Void
    @Environment(\.mfRendersStatically) private var rendersStatically

    public init(live: LiveSessionStatus, onOpen: @escaping () -> Void) {
        self.live = live
        self.onOpen = onOpen
    }

    public var body: some View {
        Button(action: onOpen) {
            TimelineView(.periodic(from: .now, by: 1)) { context in
                HStack(spacing: RecordingChipMetrics.gap) {
                    Circle()
                        .fill(MFColor.danger)
                        .frame(width: RecordingChipMetrics.dot, height: RecordingChipMetrics.dot)
                        .opacity(live.isPaused ? 0.4 : 1)
                    Text(VoiceFormat.clock(live.recordedTime(at: rendersStatically ? live.updatedAt : context.date)))
                        .font(MFFont.calloutSemibold)
                        .monospacedDigit()
                    if live.isPaused {
                        Text("Paused")
                            .font(MFFont.calloutEmphasis)
                    }
                }
                .foregroundStyle(MFColor.danger)
                .padding(.horizontal, MFSpacing.m)
                .frame(height: RecordingChipMetrics.height)
                .background(Capsule().fill(MFColor.danger.opacity(RecordingChipMetrics.fillOpacity)))
                .contentShape(Capsule())
            }
        }
        .buttonStyle(.plain)
        .accessibilityLabel(live.isPaused ? "Recording paused" : "Recording")
        .accessibilityValue(VoiceFormat.spokenClock(live.recordedTime(at: .now)))
        .accessibilityHint("Opens the meeting note")
        .help("Open the meeting note")
    }
}

enum RecordingChipMetrics {
    static let height: CGFloat = 30
    static let dot: CGFloat = 9
    static let gap: CGFloat = 7
    /// The chip's pale red fill (`#FDECEA` over white).
    static let fillOpacity: Double = 0.1
}

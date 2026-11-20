import MferenceAppCore
import MferenceDesign
import SwiftUI

/// The recording bar over a meeting note (board 4 `.rbar`): the sidebar
/// toggle, red dot, "Recording" and the time, the two level meters, Mark
/// moment ⇧⌘M, Pause or Resume, Stop and summarize. After Stop it says what
/// is finishing.
struct RecordingBar: View {
    let session: MeetingSession
    let live: LiveSessionStatus?
    let microphoneName: String
    /// "Zoom", "All apps"; nil when call audio is off.
    let callSourceName: String?
    let summaryOn: Bool
    let onToggleSidebar: () -> Void
    @Environment(\.mfRendersStatically) private var rendersStatically

    private var isPaused: Bool { session.phase == .paused }
    private var finishing: MeetingFinishStep? {
        if case .finishing(let step) = session.phase { return step }
        return nil
    }
    private var canControl: Bool { session.phase == .recording || session.phase == .paused }

    var body: some View {
        HStack(spacing: MFSpacing.l) {
            Button(action: onToggleSidebar) {
                Image(systemName: "sidebar.left")
                    .font(MFFont.body)
                    .foregroundStyle(MFColor.secondary)
            }
            .buttonStyle(.plain)
            .accessibilityLabel("Show the sidebar and notes list")
            .help("Show the sidebar and notes list")
            status
            meter(level: session.micLevel, label: "You · \(microphoneName)", color: SpeakerPalette.color(0))
            if session.callAudioAvailable, let callSourceName {
                meter(level: session.callLevel ?? 0, label: "Call audio · \(callSourceName)",
                      color: SpeakerPalette.color(1))
            } else if Self.showsMicrophoneOnly(chosen: callSourceName, session: session) {
                Text("Call audio off · microphone only")
                    .font(MFFont.caption)
                    .foregroundStyle(MFColor.tertiary)
            }
            Spacer(minLength: MFSpacing.s)
            Button {
                Task { await session.markMoment() }
            } label: {
                HStack(spacing: MFSpacing.s) {
                    Text("Mark moment")
                    MFKeycap("⇧⌘M")
                }
            }
            .buttonStyle(MFButtonStyle(.secondary, size: .medium))
            .keyboardShortcut("m", modifiers: [.command, .shift])
            .disabled(!canControl)
            Button {
                Task { isPaused ? await session.resume() : await session.pause() }
            } label: {
                Label(isPaused ? "Resume" : "Pause", systemImage: isPaused ? "play.fill" : "pause.fill")
            }
            .buttonStyle(MFButtonStyle(.secondary, size: .medium))
            .disabled(!canControl)
            Button {
                Task { await session.stop() }
            } label: {
                Label(summaryOn ? "Stop and summarize" : "Stop", systemImage: "stop.fill")
            }
            .buttonStyle(MFButtonStyle(.primary, size: .medium))
            .disabled(!canControl)
        }
        .padding(.horizontal, LiveMetrics.barPadding)
        .frame(height: LiveMetrics.barHeight)
        .background(MFColor.surface)
        .overlay(alignment: .bottom) {
            Rectangle().fill(MFColor.chromeDivider).frame(height: MFLayout.hairline)
        }
    }

    private var status: some View {
        HStack(spacing: MFSpacing.s) {
            Circle().fill(MFColor.danger)
                .frame(width: LiveMetrics.recordDot, height: LiveMetrics.recordDot)
                .opacity(isPaused || finishing != nil ? LiveMetrics.dimmed : 1)
                .accessibilityHidden(true)
            Text(Self.title(phase: session.phase))
                .font(MFFont.headline)
                .foregroundStyle(MFColor.ink)
            TimelineView(.periodic(from: .now, by: 1)) { context in
                Text(VoiceFormat.clock(live.map { $0.recordedTime(at: rendersStatically ? $0.updatedAt : context.date) }
                                       ?? 0))
                    .font(MFFont.body)
                    .monospacedDigit()
                    .foregroundStyle(MFColor.secondary)
            }
            if let finishing {
                Text(Self.finishingText(finishing))
                    .font(MFFont.caption)
                    .foregroundStyle(MFColor.secondary)
            }
        }
        .accessibilityElement(children: .combine)
    }

    /// "Call audio off": when it was off from the start, or, when chosen,
    /// only once capture is running without it (not in the moment before
    /// the session's start event).
    static func showsMicrophoneOnly(chosen: String?, session: MeetingSession) -> Bool {
        guard session.phase != .starting else { return false }
        return chosen == nil || session.micLevel > 0
    }

    static func title(phase: MeetingSession.Phase) -> String {
        switch phase {
        case .idle, .recording: "Recording"
        case .starting: "Starting…"
        case .paused: "Paused"
        case .finishing: "Finishing"
        }
    }

    static func finishingText(_ step: MeetingFinishStep) -> String {
        switch step {
        case .transcribingLastPhrases: "Transcribing the last phrases…"
        case .labellingSpeakers(let progress): "Labelling speakers · \(MFProgress.percent(progress) ?? 0)%"
        }
    }

    private func meter(level: Float, label: String, color: Color) -> some View {
        HStack(spacing: MFSpacing.s) {
            HStack(alignment: .bottom, spacing: MFSpacing.xxs) {
                ForEach(Array(LiveMetrics.meterHeights.enumerated()), id: \.offset) { index, height in
                    RoundedRectangle(cornerRadius: LiveMetrics.meterBarRadius)
                        .fill(Self.isLit(index, level: level) ? color : MFColor.hairlineStrong)
                        .frame(width: LiveMetrics.meterBarWidth, height: height)
                }
            }
            .frame(height: LiveMetrics.meterHeight, alignment: .bottom)
            .accessibilityHidden(true)
            Text(label)
                .font(MFFont.caption)
                .foregroundStyle(MFColor.secondary)
                .lineLimit(1)
        }
        .accessibilityElement(children: .combine)
    }

    /// Bars lit for a level: at least one while capturing.
    static func isLit(_ index: Int, level: Float) -> Bool {
        let count = LiveMetrics.meterHeights.count
        return Float(index) < max(1, (min(max(level, 0), 1) * Float(count)).rounded(.up))
    }
}

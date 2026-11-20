import MferenceAppCore
import MferenceDesign
import SwiftUI

// The finished meeting or file note (board 5, spec 4.5). N1's editor shows
// the note; these pieces sit around its body: `RecordingPlayerBar` above
// it, `RecordingChips` under the title, `SummaryHeader` and `AddToTodosButton`
// with the written sections (N4's summary state), and `TranscriptSection`
// below your notes.

/// The player bar at the top of the note (board 5): play/pause, the time, a
/// scrubber coloured by speaker, the length, the speed.
public struct RecordingPlayerBar: View {
    @Bindable var player: RecordingPlayerModel
    let transcript: TranscriptModel
    @State private var dragging: TimeInterval?

    public init(player: RecordingPlayerModel, transcript: TranscriptModel) {
        self.player = player
        self.transcript = transcript
    }

    public var body: some View {
        HStack(spacing: MFSpacing.l) {
            Button(action: player.togglePlay) {
                Image(systemName: player.isPlaying ? "pause.fill" : "play.fill")
                    .font(MFFont.body)
                    .foregroundStyle(MFColor.onInk)
                    .frame(width: FinishedMetrics.playButton, height: FinishedMetrics.playButton)
                    .background(Circle().fill(MFColor.ink))
            }
            .buttonStyle(.plain)
            .disabled(player.hasAudio != true)
            .accessibilityLabel(player.isPlaying ? "Pause" : "Play")
            Text(VoiceFormat.clock(dragging ?? player.currentTime))
                .font(MFFont.keycap)
                .foregroundStyle(MFColor.secondary)
                .monospacedDigit()
            scrubber
            Text(VoiceFormat.clock(player.duration))
                .font(MFFont.keycap)
                .foregroundStyle(MFColor.secondary)
            Button(RecordingPlayerModel.speedLabel(player.speed), action: player.nextSpeed)
                .buttonStyle(MFButtonStyle(.secondary))
                .accessibilityLabel("Speed")
                .accessibilityValue(RecordingPlayerModel.speedLabel(player.speed))
                .disabled(player.hasAudio != true)
        }
        .padding(.horizontal, MFSpacing.xl)
        .frame(height: FinishedMetrics.playerHeight)
        .overlay(alignment: .bottom) {
            Rectangle().fill(MFColor.chromeDivider).frame(height: MFLayout.hairline)
        }
        // Keyboard: focus the player (Tab), then Space plays or pauses and the
        // arrows skip 5 s (Space is not bound globally: it types in the note).
        .focusable()
        .onKeyPress(.space) {
            player.togglePlay()
            return .handled
        }
        .onKeyPress(.leftArrow) {
            player.skip(-FinishedMetrics.arrowSkip)
            return .handled
        }
        .onKeyPress(.rightArrow) {
            player.skip(FinishedMetrics.arrowSkip)
            return .handled
        }
        .onDisappear { player.pause() }
        .overlay(alignment: .center) {
            if player.hasAudio == false {
                Text(player.loadError ?? "The audio was deleted after transcribing. The transcript stays.")
                    .font(MFFont.caption)
                    .foregroundStyle(MFColor.tertiary)
                    .padding(.horizontal, MFSpacing.s)
                    .background(MFColor.surface)
            }
        }
        .task { await player.load() }
    }

    private var scrubber: some View {
        GeometryReader { proxy in
            let width = proxy.size.width
            ZStack(alignment: .leading) {
                ForEach(Self.bands(transcript.finalSegments, speakers: transcript, duration: player.duration)) { band in
                    Capsule()
                        .fill(SpeakerPalette.color(band.colorIndex))
                        .frame(width: max(2, width * band.length), height: FinishedMetrics.scrubberHeight)
                        .offset(x: width * band.start)
                }
                Rectangle()
                    .fill(MFColor.ink)
                    .frame(width: FinishedMetrics.playhead, height: FinishedMetrics.playheadHeight)
                    .offset(x: width * Self.fraction(dragging ?? player.currentTime, of: player.duration)
                                - FinishedMetrics.playhead / 2)
            }
            .frame(maxHeight: .infinity)
            .background(Capsule().fill(MFColor.hairline).frame(height: FinishedMetrics.scrubberHeight))
            .contentShape(Rectangle())
            .gesture(DragGesture(minimumDistance: 0)
                .onChanged { value in dragging = Self.time(at: value.location.x, width: width, duration: player.duration) }
                .onEnded { value in
                    player.seek(to: Self.time(at: value.location.x, width: width, duration: player.duration))
                    dragging = nil
                })
        }
        .frame(height: FinishedMetrics.playheadHeight)
        .accessibilityElement()
        .accessibilityLabel("Position")
        .accessibilityValue(VoiceFormat.spokenClock(player.currentTime))
        .accessibilityAdjustableAction { direction in
            let step: TimeInterval = direction == .increment ? 15 : -15
            player.seek(to: player.currentTime + step)
        }
    }

    /// One stretch of the scrubber in a speaker's colour, as fractions of the length.
    struct Band: Identifiable, Hashable {
        var id: UUID
        var start: CGFloat
        var length: CGFloat
        var colorIndex: Int
    }

    /// Each speaker's colour runs from where they start speaking until the
    /// next speaker starts (the board's continuous scrubber); silence before
    /// the first line stays the plain track.
    static func bands(_ segments: [TranscriptSegment], speakers: TranscriptModel, duration: TimeInterval) -> [Band] {
        guard duration > 0 else { return [] }
        let ordered = segments.sorted { $0.start < $1.start }
        return ordered.enumerated().map { index, segment in
            let end = index + 1 < ordered.count ? max(ordered[index + 1].start, segment.end) : max(segment.end, duration)
            return Band(id: segment.id, start: fraction(segment.start, of: duration),
                        length: fraction(end, of: duration) - fraction(segment.start, of: duration),
                        colorIndex: speakers.speaker(for: segment).colorIndex)
        }
    }

    static func fraction(_ time: TimeInterval, of duration: TimeInterval) -> CGFloat {
        duration > 0 ? CGFloat(min(max(time / duration, 0), 1)) : 0
    }

    static func time(at x: CGFloat, width: CGFloat, duration: TimeInterval) -> TimeInterval {
        width > 0 ? TimeInterval(min(max(x / width, 0), 1)) * duration : 0
    }
}

/// The chips under the title (board 5): date and time, length, one per
/// speaker in its colour, "Transcribed on this Mac".
public struct RecordingChips: View {
    let startedAt: Date
    let duration: TimeInterval
    let transcript: TranscriptModel
    let locale: Locale
    let timeZone: TimeZone

    public init(startedAt: Date, duration: TimeInterval, transcript: TranscriptModel,
                locale: Locale = .current, timeZone: TimeZone = .current) {
        self.startedAt = startedAt
        self.duration = duration
        self.transcript = transcript
        self.locale = locale
        self.timeZone = timeZone
    }

    public var body: some View {
        HStack(spacing: MFSpacing.s) {
            chip(Self.dateText(startedAt, locale: locale, timeZone: timeZone))
            chip(VoiceFormat.length(duration))
            ForEach(Self.speakersInOrder(transcript)) { speaker in
                HStack(spacing: MFSpacing.snug) {
                    Circle().fill(SpeakerPalette.color(speaker.colorIndex))
                        .frame(width: FinishedMetrics.speakerDot, height: FinishedMetrics.speakerDot)
                        .accessibilityHidden(true)
                    Text(speaker.name)
                }
                .modifier(ChipLook())
                .accessibilityLabel("Speaker \(speaker.name)")
            }
            MFChip("Transcribed on this Mac", tone: .success)
        }
    }

    private func chip(_ text: String) -> some View {
        Text(text).modifier(ChipLook())
    }

    /// "19 Nov · 10:02".
    static func dateText(_ date: Date, locale: Locale = .current, timeZone: TimeZone = .current) -> String {
        var style = Date.FormatStyle(date: .omitted, time: .omitted, locale: locale, timeZone: timeZone)
        style = style.day().month(.abbreviated)
        let time = Date.FormatStyle(date: .omitted, time: .shortened, locale: locale, timeZone: timeZone)
        return "\(date.formatted(style)) · \(date.formatted(time))"
    }

    /// The speakers who spoke: You first, as on the board, then the others
    /// in the order they first spoke.
    static func speakersInOrder(_ transcript: TranscriptModel) -> [SpeakerInfo] {
        var seen: Set<String> = []
        let all = transcript.finalSegments.compactMap { segment -> SpeakerInfo? in
            let speaker = transcript.speaker(for: segment)
            return seen.insert(speaker.id).inserted ? speaker : nil
        }
        return all.filter { $0.id == SpeakerInfo.youID } + all.filter { $0.id != SpeakerInfo.youID }
    }
}

/// The chips' grey capsule (board 5 `.chip`, a little larger than `MFChip`).
struct ChipLook: ViewModifier {
    func body(content: Content) -> some View {
        content
            .font(MFFont.calloutEmphasis)
            .foregroundStyle(MFColor.secondary)
            .padding(.horizontal, MFSpacing.compact)
            .frame(height: FinishedMetrics.chipHeight)
            .background(Capsule().fill(MFColor.fill))
    }
}

/// "Summary · Written by Quick on this Mac · Redo" (board 5); the progress or
/// failure line while it is written. N4's `MeetingSummaries` says which.
public struct SummaryHeader: View {
    public enum State: Hashable {
        case written(String)
        case working(String)
        case failed(String)
    }

    let state: State
    let onRedo: () -> Void
    let onRetry: () -> Void

    public init(state: State, onRedo: @escaping () -> Void, onRetry: @escaping () -> Void) {
        self.state = state
        self.onRedo = onRedo
        self.onRetry = onRetry
    }

    public var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: MFSpacing.compact) {
            Text("Summary")
                .font(MFFont.headline)
                .foregroundStyle(MFColor.ink)
                .accessibilityAddTraits(.isHeader)
            switch state {
            case .written(let label):
                Text(label).font(MFFont.callout).foregroundStyle(MFColor.tertiary)
                Spacer()
                Button("Redo", action: onRedo)
                    .buttonStyle(.plain)
                    .font(MFFont.calloutEmphasis)
                    .foregroundStyle(MFColor.secondary)
                    .help("Write the summary and action items again")
            case .working(let label):
                ProgressView().controlSize(.small)
                Text(label).font(MFFont.callout).foregroundStyle(MFColor.secondary)
                Spacer()
            case .failed(let message):
                Text(message).font(MFFont.callout).foregroundStyle(MFColor.danger)
                Spacer()
                Button("Retry", action: onRetry)
                    .buttonStyle(.mfSecondary)
            }
        }
    }
}

/// "Add 4 to To-dos", then "Added to To-dos" (board 5, spec 4.5).
public struct AddToTodosButton: View {
    let count: Int
    let added: Bool
    let action: () -> Void

    public init(count: Int, added: Bool, action: @escaping () -> Void) {
        self.count = count
        self.added = added
        self.action = action
    }

    public var body: some View {
        Button(Self.title(count: count, added: added), action: action)
            .buttonStyle(.mfSecondary)
            .disabled(added || count == 0)
    }

    static func title(count: Int, added: Bool) -> String {
        added ? "Added to To-dos" : "Add \(count) to To-dos"
    }
}

/// The transcript under your notes (board 5): the first lines and "Show all
/// (n min)"; each row a time chip, the speaker and the text; clicking a time
/// plays from there; the row playing is highlighted.
public struct TranscriptSection: View {
    let transcript: TranscriptModel
    let player: RecordingPlayerModel
    /// Lines shown before "Show all".
    let previewCount: Int
    @State private var showsAll = false

    public init(transcript: TranscriptModel, player: RecordingPlayerModel, previewCount: Int = 2) {
        self.transcript = transcript
        self.player = player
        self.previewCount = previewCount
    }

    public var body: some View {
        // Read once per transcript change: the rows read the player's time
        // themselves, so the section is not rebuilt on every tick.
        let segments = transcript.finalSegments
        let shown = showsAll ? segments : Array(segments.prefix(previewCount))
        VStack(alignment: .leading, spacing: MFSpacing.s) {
            HStack(alignment: .firstTextBaseline, spacing: MFSpacing.compact) {
                Text("Transcript")
                    .font(MFFont.headline)
                    .foregroundStyle(MFColor.ink)
                    .accessibilityAddTraits(.isHeader)
                Text(player.hasAudio == false ? "Audio not kept" : "Click a time to play from there")
                    .font(MFFont.callout)
                    .foregroundStyle(MFColor.tertiary)
                Spacer()
                if segments.count > previewCount {
                    Button(showsAll ? "Show less" : Self.showAllTitle(duration: player.duration)) {
                        showsAll.toggle()
                    }
                    .buttonStyle(.plain)
                    .font(MFFont.calloutEmphasis)
                    .foregroundStyle(MFColor.secondary)
                }
            }
            if segments.isEmpty {
                Text("No speech was found in this recording.")
                    .font(MFFont.body)
                    .foregroundStyle(MFColor.tertiary)
            }
            // Lazy: "Show all" on a long meeting builds only the rows in view.
            LazyVStack(alignment: .leading, spacing: MFSpacing.s) {
                ForEach(Array(shown.enumerated()), id: \.element.id) { index, segment in
                    // `shown` is a prefix of `segments`: the next line may be hidden.
                    TranscriptRow(segment: segment, nextStart: index + 1 < segments.count ? segments[index + 1].start : nil,
                                  speaker: transcript.speaker(for: segment), player: player)
                }
            }
        }
    }

    static func showAllTitle(duration: TimeInterval) -> String {
        "Show all (\(VoiceFormat.length(duration)))"
    }
}

/// One transcript line; it alone follows the player's time (the line
/// playing is highlighted), so a tick rebuilds only the rows in view.
struct TranscriptRow: View {
    let segment: TranscriptSegment
    /// The next shown line's start: this one plays until then.
    let nextStart: TimeInterval?
    let speaker: SpeakerInfo
    let player: RecordingPlayerModel

    /// The same line `RecordingPlayerModel.playingSegment(in:)` picks.
    static func isPlaying(start: TimeInterval, nextStart: TimeInterval?, time: TimeInterval, playing: Bool) -> Bool {
        guard playing || time > 0 else { return false }
        let now = time + 0.05
        return start <= now && (nextStart.map { $0 > now } ?? true)
    }

    var body: some View {
        let isPlaying = Self.isPlaying(start: segment.start, nextStart: nextStart, time: player.currentTime,
                                       playing: player.isPlaying)
        return HStack(alignment: .firstTextBaseline, spacing: MFSpacing.m) {
            Button {
                player.play(from: segment.start)
            } label: {
                Text(VoiceFormat.clock(segment.start))
                    .font(MFFont.keycap)
                    .foregroundStyle(MFColor.accent)
                    .padding(.horizontal, MFSpacing.snug)
                    .padding(.vertical, MFSpacing.xxs)
                    .background(RoundedRectangle(cornerRadius: MFRadius.xs).fill(MFColor.accentFillFaint))
            }
            .buttonStyle(.plain)
            .disabled(player.hasAudio != true)
            .accessibilityLabel("Play from \(VoiceFormat.spokenClock(segment.start))")
            Text(speaker.name)
                .font(MFFont.calloutSemibold)
                .foregroundStyle(SpeakerPalette.color(speaker.colorIndex))
                .frame(width: FinishedMetrics.speakerColumn, alignment: .leading)
                .lineLimit(1)
            Text(verbatim: segment.text)
                .font(MFFont.body)
                .foregroundStyle(MFColor.ink)
                .fixedSize(horizontal: false, vertical: true)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(.horizontal, MFSpacing.s)
        .padding(.vertical, MFSpacing.snug)
        .background(RoundedRectangle(cornerRadius: MFRadius.s).fill(isPlaying ? MFColor.accentFillFaint : .clear))
        .accessibilityElement(children: .contain)
        .accessibilityAddTraits(isPlaying ? .isSelected : [])
    }
}

/// Board 5 sizes (`MeetingDone.dc.html`).
enum FinishedMetrics {
    static let playerHeight: CGFloat = 64
    static let playButton: CGFloat = 38
    static let scrubberHeight: CGFloat = 6
    static let playhead: CGFloat = 2
    static let playheadHeight: CGFloat = 26
    static let chipHeight: CGFloat = 26
    static let speakerDot: CGFloat = 8
    static let speakerColumn: CGFloat = 104
    /// Arrow keys on the player skip this many seconds.
    static let arrowSkip: TimeInterval = 5
}

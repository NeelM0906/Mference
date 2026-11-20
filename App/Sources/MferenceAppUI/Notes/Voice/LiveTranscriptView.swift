import MferenceAppCore
import MferenceDesign
import SwiftUI

/// The live transcript column (board 4 `.tr`): segments as each phrase
/// finishes, labelled You / Speaker n with their time; the one still being
/// spoken greyed with a caret; clicking a name renames that speaker
/// everywhere in the note (N4's `TranscriptModel.rename`).
public struct LiveTranscriptView: View {
    let transcript: TranscriptModel
    let languageCode: String?
    /// Scrolls here when it changes (a time chip in your notes was clicked).
    let scrollTarget: TimeInterval?
    @State private var renaming: SpeakerInfo?
    @State private var newName = ""

    public init(transcript: TranscriptModel, languageCode: String? = nil, scrollTarget: TimeInterval? = nil) {
        self.transcript = transcript
        self.languageCode = languageCode
        self.scrollTarget = scrollTarget
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: MFSpacing.m) {
            HStack(spacing: MFSpacing.s) {
                Text("Live transcript")
                    .font(MFFont.headline)
                    .foregroundStyle(MFColor.ink)
                    .accessibilityAddTraits(.isHeader)
                MFChip("On this Mac", tone: .success)
                MFChip(VoiceFormat.languageChip(.automatic, detected: Self.language(languageCode, transcript)))
                Spacer()
                Text("Click a name to rename")
                    .font(MFFont.caption)
                    .foregroundStyle(MFColor.tertiary)
            }
            ScrollViewReader { proxy in
                MFScroll {
                    LazyVStack(alignment: .leading, spacing: MFSpacing.comfortable) {
                        if transcript.segments.isEmpty {
                            Text("Listening. Lines appear here after each pause.")
                                .font(MFFont.body)
                                .foregroundStyle(MFColor.tertiary)
                        }
                        ForEach(transcript.segments) { segment in
                            row(segment).id(segment.id)
                        }
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
                .onChange(of: transcript.segments.last?.id) { _, last in
                    guard scrollTarget == nil, let last else { return }
                    withAnimation { proxy.scrollTo(last, anchor: .bottom) }
                }
                .onChange(of: scrollTarget) { _, time in
                    guard let time, let hit = Self.segment(at: time, in: transcript.segments) else { return }
                    withAnimation { proxy.scrollTo(hit.id, anchor: .top) }
                }
            }
        }
        .padding(.horizontal, LiveMetrics.transcriptPadding.width)
        .padding(.vertical, LiveMetrics.transcriptPadding.height)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .background(MFColor.canvas)
        .alert("Rename speaker", isPresented: Binding(get: { renaming != nil }, set: { if !$0 { renaming = nil } }),
               presenting: renaming) { speaker in
            TextField("Name", text: $newName)
            Button("Rename") { transcript.rename(speakerID: speaker.id, to: newName) }
                .keyboardShortcut(.defaultAction)
            Button("Cancel", role: .cancel) {}
        } message: { speaker in
            Text("Renames \(speaker.name) everywhere in this note.")
        }
    }

    /// The language to show: the session's, else the last segment's.
    static func language(_ code: String?, _ transcript: TranscriptModel) -> String? {
        code ?? transcript.segments.last(where: { $0.languageCode != nil })?.languageCode
    }

    /// The segment a note line's time stamp points at: the last one that
    /// started at or before it.
    static func segment(at time: TimeInterval, in segments: [TranscriptSegment]) -> TranscriptSegment? {
        segments.last { $0.start <= time } ?? segments.first
    }

    private func row(_ segment: TranscriptSegment) -> some View {
        let speaker = transcript.speaker(for: segment)
        return HStack(alignment: .top, spacing: MFSpacing.m) {
            VStack(alignment: .leading, spacing: MFSpacing.xxs) {
                Button {
                    newName = speaker.name
                    renaming = speaker
                } label: {
                    Text(speaker.name)
                        .font(MFFont.calloutSemibold)
                        .foregroundStyle(SpeakerPalette.color(speaker.colorIndex))
                        .lineLimit(1)
                }
                .buttonStyle(.plain)
                .accessibilityLabel("\(speaker.name). Rename")
                .help("Rename \(speaker.name) everywhere in this note")
                Text(VoiceFormat.clock(segment.start))
                    .font(MFFont.keycap)
                    .foregroundStyle(MFColor.tertiary)
            }
            .frame(width: LiveMetrics.whoWidth, alignment: .leading)
            // Verbatim: transcript text is never read as Markdown.
            (segment.isFinal ? Text(verbatim: segment.text)
                             : Text("\(Text(verbatim: segment.text))\(Text(verbatim: " ▍").foregroundStyle(MFColor.accent))"))
                .font(MFFont.body)
                .lineSpacing(MFLineSpacing.normal)
                .foregroundStyle(segment.isFinal ? MFColor.ink : MFColor.tertiary)
                .fixedSize(horizontal: false, vertical: true)
                .accessibilityLabel(segment.isFinal ? segment.text : "\(segment.text), still speaking")
        }
    }
}

/// Meeting layout sizes from `MeetingLive.dc.html`.
enum LiveMetrics {
    static let transcriptPadding = CGSize(width: 28, height: 16)
    static let whoWidth: CGFloat = 92
    /// The your-notes column (`.notes`).
    static let notesWidth: CGFloat = 540
    static let notesPadding = CGSize(width: 36, height: 22)
    /// The recording bar (`.rbar`).
    static let barHeight: CGFloat = 58
    static let barPadding: CGFloat = 18
    static let recordDot: CGFloat = 10
    /// Level meter bars (`.bars`).
    static let meterBarWidth: CGFloat = 3
    static let meterBarRadius: CGFloat = 1
    static let meterHeight: CGFloat = 16
    static let meterHeights: [CGFloat] = [6, 10, 14, 16, 8]
}

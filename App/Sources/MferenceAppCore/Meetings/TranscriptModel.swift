import Foundation
import Observation

/// The segments and speakers of one meeting or file note, as the live
/// transcript and the finished-meeting view show them (spec 4.4, 4.5).
///
/// Segments stay in recording-time order; the one still being spoken
/// (`isFinal == false`) shows greyed. Renaming a speaker changes its name
/// everywhere in the note; `MeetingSession` saves the change. Renames made
/// while live carry over the final speaker pass (`applyFinal`).
@MainActor
@Observable
public final class TranscriptModel {
    public private(set) var segments: [TranscriptSegment]
    public private(set) var speakers: [SpeakerInfo]
    public private(set) var gaps: [RecordingGap]
    /// Names the person gave, by speaker id; they win over the diarizer's.
    public private(set) var renames: [String: String]

    /// Called after a rename, so the owner can save it.
    @ObservationIgnored var onChange: (@MainActor (TranscriptModel) -> Void)?

    public init(segments: [TranscriptSegment] = [], speakers: [SpeakerInfo] = [], gaps: [RecordingGap] = [],
                renames: [String: String] = [:]) {
        self.segments = Self.ordered(segments)
        self.speakers = speakers
        self.gaps = gaps
        self.renames = renames
    }

    /// What a finished (or interrupted) note holds.
    public convenience init(recording: NoteRecording) {
        let speakers = recording.speakers
        self.init(segments: recording.segments, speakers: speakers, gaps: recording.gaps)
    }

    // MARK: Reading

    /// The final segments only (what is saved).
    public var finalSegments: [TranscriptSegment] { segments.filter(\.isFinal) }

    /// The speaker a segment belongs to; an unknown id reads as "You" for the
    /// microphone and "Others" for the rest, so a row never shows a bare id.
    public func speaker(for segment: TranscriptSegment) -> SpeakerInfo {
        speaker(id: segment.speakerID, track: segment.track)
    }

    public func speaker(id: String, track: TrackKind = .call) -> SpeakerInfo {
        if let known = speakers.first(where: { $0.id == id }) { return known }
        var fallback = track == .microphone ? SpeakerInfo.you : SpeakerInfo.others
        fallback.id = id
        if let name = renames[id] { fallback.name = name }
        return fallback
    }

    /// The recording time the transcript reaches.
    public var duration: TimeInterval { segments.map(\.end).max() ?? 0 }

    // MARK: Live changes (MeetingSession)

    /// Adds a segment or replaces the one with the same id.
    func upsert(_ segment: TranscriptSegment) {
        // Updates are nearly always to the newest segments: look from the back.
        if let index = segments.lastIndex(where: { $0.id == segment.id }) {
            segments[index] = segment
            let before = index > 0 && segments[index - 1].start > segment.start
            let after = index + 1 < segments.count && segments[index + 1].start < segment.start
            if before || after { segments = Self.ordered(segments) }
        } else {
            // Most segments arrive in order: append, and sort only when not.
            let outOfOrder = segments.last.map { $0.start > segment.start } ?? false
            segments.append(segment)
            if outOfOrder { segments = Self.ordered(segments) }
        }
        ensureSpeaker(for: segment)
    }

    func remove(_ id: UUID) {
        segments.removeAll { $0.id == id }
    }

    /// The live diarizer's speakers; the person's names stay.
    func updateSpeakers(_ updated: [SpeakerInfo]) {
        var merged = updated.map(applyingRename)
        // Speakers already shown that the update left out stay (their rows still need a name).
        for existing in speakers where !merged.contains(where: { $0.id == existing.id }) {
            merged.append(existing)
        }
        speakers = merged
    }

    func addGap(_ gap: RecordingGap) {
        guard !gaps.contains(gap) else { return }
        gaps.append(gap)
    }

    /// The finished recording replaces the live state. Segment ids are
    /// stable; speaker ids may change in the final pass. Renames follow
    /// `mapping` (live id → final id) and, for live speakers it leaves out,
    /// the final speaker their segments overlap most (spec 6.3). A result
    /// without segments keeps the live final segments rather than erasing them.
    func applyFinal(_ result: TranscriptResult) {
        let finals = result.segments.filter(\.isFinal)
        let kept = finals.isEmpty ? finalSegments : finals
        let carried = Self.carriedRenames(renames, liveSegments: segments, finalSegments: kept,
                                          mapping: result.speakerMapping)
        renames = carried
        segments = Self.ordered(kept)
        var finalSpeakers = result.speakers.map(applyingRename)
        for segment in segments where !finalSpeakers.contains(where: { $0.id == segment.speakerID }) {
            finalSpeakers.append(applyingRename(speaker(for: segment)))
        }
        speakers = finalSpeakers
        for gap in result.gaps { addGap(gap) }
    }

    // MARK: Renaming

    /// Renames a speaker everywhere in the note. An empty name is ignored.
    public func rename(speakerID: String, to name: String) {
        let trimmed = name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        renames[speakerID] = trimmed
        if let index = speakers.firstIndex(where: { $0.id == speakerID }) {
            speakers[index].name = trimmed
        } else {
            let track = segments.first { $0.speakerID == speakerID }?.track ?? .call
            var added = speaker(id: speakerID, track: track)
            added.name = trimmed
            added.colorIndex = nextColorIndex()
            speakers.append(added)
        }
        onChange?(self)
    }

    // MARK: Helpers

    private func applyingRename(_ speaker: SpeakerInfo) -> SpeakerInfo {
        guard let name = renames[speaker.id] else { return speaker }
        var renamed = speaker
        renamed.name = name
        return renamed
    }

    private func ensureSpeaker(for segment: TranscriptSegment) {
        guard !speakers.contains(where: { $0.id == segment.speakerID }) else { return }
        var added = speaker(for: segment)
        if segment.speakerID == SpeakerInfo.youID { added.colorIndex = SpeakerInfo.you.colorIndex }
        else if segment.speakerID != SpeakerInfo.othersID { added.colorIndex = nextColorIndex() }
        speakers.append(applyingRename(added))
    }

    private func nextColorIndex() -> Int { (speakers.map(\.colorIndex).max() ?? -1) + 1 }

    static func ordered(_ segments: [TranscriptSegment]) -> [TranscriptSegment] {
        // Stable: equal starts keep their arrival order.
        segments.enumerated().sorted { lhs, rhs in
            lhs.element.start != rhs.element.start ? lhs.element.start < rhs.element.start : lhs.offset < rhs.offset
        }.map(\.element)
    }

    /// Renames keyed by final speaker id. An explicit mapping wins; otherwise
    /// each renamed live speaker goes to the final speaker whose segments
    /// overlap its own the most (by segment id, then by time).
    static func carriedRenames(_ renames: [String: String], liveSegments: [TranscriptSegment],
                               finalSegments: [TranscriptSegment],
                               mapping: [String: String]) -> [String: String] {
        guard !renames.isEmpty else { return [:] }
        var carried: [String: String] = [:]
        let finalByID = Dictionary(finalSegments.map { ($0.id, $0) }, uniquingKeysWith: { first, _ in first })
        for (liveID, name) in renames.sorted(by: { $0.key < $1.key }) {
            if let finalID = mapping[liveID] {
                carried[finalID] = name
                continue
            }
            var overlap: [String: TimeInterval] = [:]
            let mine = liveSegments.filter { $0.speakerID == liveID }
            for segment in mine {
                if let final = finalByID[segment.id] {
                    overlap[final.speakerID, default: 0] += max(segment.end - segment.start, 0.001)
                } else {
                    for final in finalSegments {
                        let shared = min(segment.end, final.end) - max(segment.start, final.start)
                        if shared > 0 { overlap[final.speakerID, default: 0] += shared }
                    }
                }
            }
            if mine.isEmpty, finalSegments.contains(where: { $0.speakerID == liveID }) {
                // Renamed before speaking, and the id survived the final pass.
                overlap[liveID] = 1
            }
            guard let best = overlap.max(by: { $0.value != $1.value ? $0.value < $1.value : $0.key > $1.key })?.key
            else { continue }
            // A rename chosen explicitly for that final id stays.
            if carried[best] == nil { carried[best] = name }
        }
        return carried
    }
}

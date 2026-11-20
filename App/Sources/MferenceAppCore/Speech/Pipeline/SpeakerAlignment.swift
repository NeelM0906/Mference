import Foundation

/// Puts diarizer turns on transcript segments (spec 6.3, D3): a call segment
/// gets the speaker whose turns overlap it most; after the final pass, each
/// live speaker maps to the final speaker it overlaps most, so names given
/// during the meeting carry over.
public enum SpeakerAlignment {
    /// The speaker who talks most inside `start...end`, or nil when no turn overlaps.
    public static func speaker(from start: TimeInterval, to end: TimeInterval, in turns: [SpeakerTurn]) -> String? {
        var totals: [String: TimeInterval] = [:]
        for turn in turns {
            let overlap = min(end, turn.end) - max(start, turn.start)
            if overlap > 0 { totals[turn.speaker, default: 0] += overlap }
        }
        return totals.max { $0.value < $1.value || ($0.value == $1.value && $0.key > $1.key) }?.key
    }

    /// Live speaker id → final speaker id by total overlap of each live
    /// speaker's segments with the final turns. Live speakers with no
    /// overlap are left out.
    public static func mapping(liveSegments: [TranscriptSegment], finalTurns: [SpeakerTurn]) -> [String: String] {
        var overlap: [String: [String: TimeInterval]] = [:]
        for segment in liveSegments where segment.track != .microphone {
            for turn in finalTurns {
                let shared = min(segment.end, turn.end) - max(segment.start, turn.start)
                if shared > 0 { overlap[segment.speakerID, default: [:]][turn.speaker, default: 0] += shared }
            }
        }
        return overlap.compactMapValues { totals in
            totals.max { $0.value < $1.value || ($0.value == $1.value && $0.key > $1.key) }?.key
        }
    }

    /// Speakers as the transcript names them: "Speaker 2", "Speaker 3", …
    /// in order of first appearance ("You" is the microphone, colour 0).
    public static func speakers(for segments: [TranscriptSegment]) -> [SpeakerInfo] {
        var seen: [String] = []
        for segment in segments.sorted(by: { $0.start < $1.start }) where !seen.contains(segment.speakerID) {
            seen.append(segment.speakerID)
        }
        var speakers: [SpeakerInfo] = []
        var number = 2
        for id in seen {
            switch id {
            case SpeakerInfo.youID: speakers.append(.you)
            case SpeakerInfo.othersID: speakers.append(.others)
            default:
                speakers.append(SpeakerInfo(id: id, name: "Speaker \(number)", colorIndex: number - 1))
                number += 1
            }
        }
        return speakers
    }

    /// The final labels: each call segment gets the final turn speaker it
    /// overlaps most (or its live speaker's mapping when none overlaps).
    public static func relabel(_ segments: [TranscriptSegment], finalTurns: [SpeakerTurn],
                               mapping: [String: String]) -> [TranscriptSegment] {
        segments.map { segment in
            guard segment.track != .microphone else { return segment }
            var relabelled = segment
            relabelled.speakerID = speaker(from: segment.start, to: segment.end, in: finalTurns)
                ?? mapping[segment.speakerID] ?? segment.speakerID
            return relabelled
        }
    }
}

import Foundation

// The meeting layer's reading and writing of `NoteRecording`'s JSON fields
// (spec 8.1). The values are the speech contract's Codable types; only
// `Meetings/` writes `transcriptJSON`, `speakersJSON` and `gapsJSON`.

extension NoteRecording {
    /// The saved segments, in recording-time order. Unreadable data reads as empty.
    public var segments: [TranscriptSegment] {
        get { RecordingCoding.decode([TranscriptSegment].self, from: transcriptJSON) ?? [] }
        set { transcriptJSON = RecordingCoding.encode(newValue) }
    }

    public var speakers: [SpeakerInfo] {
        get { RecordingCoding.decode([SpeakerInfo].self, from: speakersJSON) ?? [] }
        set { speakersJSON = RecordingCoding.encode(newValue) }
    }

    public var gaps: [RecordingGap] {
        get { RecordingCoding.decode([RecordingGap].self, from: gapsJSON) ?? [] }
        set { gapsJSON = RecordingCoding.encode(newValue) }
    }

    public var tracks: [RecordedTrack] {
        get { RecordingCoding.decode([RecordedTrack].self, from: tracksJSON) ?? [] }
        set { tracksJSON = RecordingCoding.encode(newValue) }
    }
}

enum RecordingCoding {
    static func encode<Value: Encodable>(_ value: Value) -> Data? {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        return try? encoder.encode(value)
    }

    static func decode<Value: Decodable>(_ type: Value.Type, from data: Data?) -> Value? {
        guard let data else { return nil }
        return try? JSONDecoder().decode(type, from: data)
    }
}

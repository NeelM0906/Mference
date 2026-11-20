import Foundation

/// Without headphones the microphone hears the call. A microphone segment is
/// an echo when call segments overlap it in time and their words are nearly
/// the same: at least `threshold` (0.8, spec 6.3) of the microphone
/// segment's normalised words appear in the overlapping call text.
public enum EchoDeduplicator {
    public static let threshold = 0.8

    /// Whether `microphone` repeats the overlapping parts of `call`.
    public static func isEcho(_ microphone: TranscriptSegment, of call: [TranscriptSegment],
                              threshold: Double = threshold) -> Bool {
        let overlapping = call.filter { $0.track == .call && overlaps($0, microphone) }
        guard !overlapping.isEmpty else { return false }
        let mine = tokens(microphone.text)
        guard !mine.isEmpty else { return false }
        var theirs: [String: Int] = [:]
        for segment in overlapping {
            for token in tokens(segment.text) { theirs[token, default: 0] += 1 }
        }
        var shared = 0
        for token in mine where (theirs[token] ?? 0) > 0 {
            shared += 1
            theirs[token]! -= 1
        }
        return Double(shared) / Double(mine.count) >= threshold
    }

    static func overlaps(_ a: TranscriptSegment, _ b: TranscriptSegment) -> Bool {
        a.start < b.end && b.start < a.end
    }

    /// Lowercased, diacritics folded, punctuation stripped.
    static func tokens(_ text: String) -> [String] {
        text.folding(options: [.caseInsensitive, .diacriticInsensitive], locale: nil)
            .components(separatedBy: CharacterSet.alphanumerics.inverted)
            .filter { !$0.isEmpty }
    }
}

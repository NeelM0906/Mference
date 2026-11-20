import MferenceAppCore
import MferenceDesign
import SwiftUI

/// Speaker colours (`SpeakerInfo.colorIndex`), from the boards: You in the
/// accent blue, Speaker 2 in amber, Speaker 3 in green, then the cycle
/// continues through the other text tokens. Each passes 4.5:1 as text on
/// the surface in both modes (they are the design's text tokens).
enum SpeakerPalette {
    static let colors: [Color] = [MFColor.accent, MFColor.preview, MFColor.success, MFColor.accentStrong,
                                  MFColor.danger, MFColor.secondary]

    static func color(_ index: Int) -> Color {
        colors[((index % colors.count) + colors.count) % colors.count]
    }

    static func color(for speakerID: String, in speakers: [SpeakerInfo]) -> Color {
        color(speakers.first { $0.id == speakerID }?.colorIndex ?? (speakerID == SpeakerInfo.youID ? 0 : 1))
    }

    static func name(for speakerID: String, in speakers: [SpeakerInfo]) -> String {
        if let speaker = speakers.first(where: { $0.id == speakerID }) { return speaker.name }
        return speakerID == SpeakerInfo.youID ? "You" : "Speaker"
    }
}

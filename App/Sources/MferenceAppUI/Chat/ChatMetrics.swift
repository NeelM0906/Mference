import MferenceDesign
import SwiftUI

/// Chat-only sizes measured from the mockup source (`Main.dc.html`, CSS px =
/// points). Spacing, radius, color and type come from `MferenceDesign`; these
/// are the few component dimensions the shared tokens do not cover.
enum ChatMetrics {
    /// The message column and composer (`max-width: 820px`).
    static let columnWidth: CGFloat = 820
    /// The user bubble (`max-width: 520px`).
    static let bubbleWidth: CGFloat = 520
    /// The model picker panel (`width: 400px`, `max-height: 470px`).
    static let pickerWidth: CGFloat = 400
    static let pickerMaxHeight: CGFloat = 470
    /// Composer controls: chips and the round + button (`height: 28px`).
    static let controlHeight: CGFloat = 28
    /// The send / stop button (`width: 32px`).
    static let sendSize: CGFloat = 32
    /// The radio dot in a picker row (16 px ring, 8 px dot).
    static let radioSize: CGFloat = 16
    static let radioDot: CGFloat = 8
    /// The radio ring's stroke.
    static let hairlineWidth: CGFloat = 1
    static let radioStroke: CGFloat = 1.5
    /// Keycap in the New chat button (`min-width: 20px`).
    static let keycap: CGFloat = 20
    /// The composer text area: two lines tall, at most eight.
    static let composerLines = 2...8
    /// The hover card on a source chip.
    static let hoverCardWidth: CGFloat = 300
    /// Section headers: `letter-spacing: .06em` at 11 px.
    static let sectionTracking: CGFloat = 0.66
    /// Reasoning shown inside the steps disclosure before it scrolls.
    static let reasoningMaxHeight: CGFloat = 220
}

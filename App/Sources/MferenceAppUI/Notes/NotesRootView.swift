import MferenceAppCore
import MferenceDesign
import SwiftUI

/// The Notes section (Notes program spec 4.1, board 1). Team N1 replaces this
/// body with the three panes; until then it says plainly that the screen is
/// still being built.
public struct NotesRootView: View {
    public init() {}

    public var body: some View {
        SectionPlaceholderView(section: .notes)
    }
}

import Observation
import SwiftUI

/// The view-facing models the shell hands to U1's screens; not U2's
/// `AppServices` composition root in MferenceAppCore. The app fills it at
/// launch from U2's adapters (MferenceAppUI/Adapters); tests and snapshots
/// fill it with fixtures. A nil member shows that section's honest empty
/// state.
@Observable
public final class ScreenModels {
    public var library: (any ModelLibraryModel)?

    public init(library: (any ModelLibraryModel)? = nil) {
        self.library = library
    }
}

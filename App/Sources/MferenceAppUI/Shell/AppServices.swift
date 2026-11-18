import Observation
import SwiftUI

/// The live models the shell hands to U1's screens. The app fills it at
/// launch (U2's model library and memory store, U4's connections); tests and
/// snapshots fill it with fixtures. A nil member shows that section's honest
/// empty state.
@Observable
public final class AppServices {
    public var library: (any ModelLibraryModel)?

    public init(library: (any ModelLibraryModel)? = nil) {
        self.library = library
    }
}

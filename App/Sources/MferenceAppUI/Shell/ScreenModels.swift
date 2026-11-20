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
    public var memory: (any MemoryListModel)?
    public var settings: (any AppSettingsModel)?
    /// The Notes screen (N1); nil shows Notes' starting state.
    public var notes: NotesModel?

    public init(library: (any ModelLibraryModel)? = nil, memory: (any MemoryListModel)? = nil,
                settings: (any AppSettingsModel)? = nil, notes: NotesModel? = nil) {
        self.library = library
        self.memory = memory
        self.settings = settings
        self.notes = notes
    }
}

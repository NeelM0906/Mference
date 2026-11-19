import SwiftUI

private nonisolated struct PhoneSheetsBelowFoldKey: EnvironmentKey {
    static let defaultValue = false
}

extension EnvironmentValues {
    /// True on a turned Duo, where every sheet must keep its top below the
    /// fold line that runs across the middle of the window. Set by the shell.
    nonisolated var phoneSheetsBelowFold: Bool {
        get { self[PhoneSheetsBelowFoldKey.self] }
        set { self[PhoneSheetsBelowFoldKey.self] = newValue }
    }
}

extension View {
    /// `presentationDetents`, kept below the fold on a turned Duo: there every
    /// sheet opens at `.medium`, whose top sits under the fold (measured on
    /// the 669 × 951 window: top at 490 pt, the fold at 475.5 plus the 12 pt
    /// keep-out is 487.5). Elsewhere the detents pass through. Use it on every
    /// sheet's content instead of `presentationDetents`.
    func phoneSheetDetents(_ detents: Set<PresentationDetent>) -> some View {
        modifier(PhoneSheetDetents(detents: detents))
    }
}

struct PhoneSheetDetents: ViewModifier {
    let detents: Set<PresentationDetent>
    @Environment(\.phoneSheetsBelowFold) private var belowFold

    func body(content: Content) -> some View {
        content.presentationDetents(Self.detents(detents, belowFold: belowFold))
    }

    static func detents(_ detents: Set<PresentationDetent>, belowFold: Bool) -> Set<PresentationDetent> {
        belowFold ? [.medium] : detents
    }
}

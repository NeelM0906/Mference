import MferenceDesign
import SwiftUI

/// One screen action in the top bar (Compose, New event, Capture...).
public struct ShellToolbarItem: Identifiable, Equatable, @unchecked Sendable {
    public let id: String
    let view: AnyView

    /// Equal by id: the shell keeps the item's view while its id is the same,
    /// so the view must read live state itself (an `@Observable` model, a
    /// binding), not values copied in when the screen last rendered.
    public static func == (lhs: ShellToolbarItem, rhs: ShellToolbarItem) -> Bool { lhs.id == rhs.id }
}

/// Collects the toolbar items the visible screen contributes.
struct ShellToolbarKey: PreferenceKey {
    static let defaultValue: [ShellToolbarItem] = []

    static func reduce(value: inout [ShellToolbarItem], nextValue: () -> [ShellToolbarItem]) {
        value.append(contentsOf: nextValue())
    }
}

extension View {
    /// Adds screen actions to the shell's top bar, left of "On this Mac",
    /// while this screen shows. For example, in MailView:
    ///
    ///     .shellToolbar(id: "mail.compose") {
    ///         Button { model.compose() } label: { Label("Compose", systemImage: "square.and.pencil") }
    ///             .buttonStyle(.mfPrimary)
    ///             .keyboardShortcut("c", modifiers: [])
    ///     }
    ///
    /// Use a stable, unique `id` per item. The items' views must read live
    /// state (see `ShellToolbarItem`). Works in the window toolbar and in
    /// snapshots (`RootView(chrome: .inline)`).
    public func shellToolbar<Items: View>(id: String, @ViewBuilder _ items: () -> Items) -> some View {
        preference(key: ShellToolbarKey.self, value: [ShellToolbarItem(id: id, view: AnyView(items()))])
    }
}

/// The contributed items, side by side.
struct ShellToolbarItems: View {
    let items: [ShellToolbarItem]

    var body: some View {
        if !items.isEmpty {
            HStack(spacing: MFSpacing.s) {
                ForEach(items) { $0.view }
            }
        }
    }
}

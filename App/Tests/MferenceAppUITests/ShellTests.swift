import Foundation
import MferenceAppCore
@testable import MferenceAppUI
import SwiftUI
import Testing

@Suite struct ShellTests {
    @Test func sectionShortcuts() {
        let keys = AppSection.pills.map { String(ShellModel.shortcut(for: $0).character) }
        #expect(keys == ["1", "2", "3", "4", "5", "6"])
        #expect(ShellModel.shortcut(for: .settings).character == ",")
    }

    @Test func showSelectsAndClosesSearch() {
        let model = ShellModel(defaults: nil)
        #expect(model.selection == .chat)
        model.isSearchPresented = true
        model.show(.scan)
        #expect(model.selection == .scan)
        #expect(!model.isSearchPresented)
    }

    @Test func settingsGearReturnsToThePreviousSection() {
        let model = ShellModel(selection: .mail, defaults: nil)
        model.toggleSettings()
        #expect(model.selection == .settings)
        model.toggleSettings()
        #expect(model.selection == .mail)
        model.show(.scan)
        model.show(.settings)
        model.show(.settings)
        model.toggleSettings()
        #expect(model.selection == .scan, "opening Settings twice keeps the section to return to")
    }

    @Test func appearanceCyclesAndPersists() throws {
        let suite = "ShellTests.\(UUID().uuidString)"
        let defaults = try #require(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }
        let model = ShellModel(defaults: defaults)
        #expect(model.appearance == .system)
        model.setAppearance(model.appearance.next)
        #expect(model.appearance == .light)
        #expect(ShellModel(defaults: defaults).appearance == .light)
        #expect(ShellModel.Appearance.dark.next == .system)
        #expect(ShellModel.Appearance.system.colorScheme == nil)
    }

    @Test func notesIsBuiltWithItsNoteIcon() {
        #expect(!AppSection.notes.isComingLater)
        #expect(AppSection.notes.systemImage == "note.text")
        #expect(AppSection.allCases.filter(\.isComingLater) == [.errands])
    }

    @Test func phaseCSectionsHaveHonestCopy() {
        for section in AppSection.allCases where section.isComingLater {
            let copy = ComingLaterView.copy(for: section)
            #expect(copy.headline != section.title)
            #expect(copy.detail.contains("later version"))
            #expect(!copy.promise.isEmpty)
        }
    }
}

/// G-A3 renders of the shell: the chrome over each section, as the mockup
/// boards frame them. Written to a temp directory unless MFERENCE_SNAPSHOT_DIR is set.
@Suite struct ShellSnapshotTests {
    static let boards: [(AppSection, String)] = [
        (.chat, "02-chat-web"),
        (.errands, "05-errands"),
        (.notes, "notes-01-main"),
    ]

    @Test(arguments: [ColorScheme.light, .dark])
    func chromeBoards(colorScheme: ColorScheme) throws {
        for (section, name) in Self.boards {
            let model = ShellModel(selection: section, defaults: nil)
            let url = try Snapshot.render(RootView(model: model, chrome: .inline), name: name, colorScheme: colorScheme)
            #expect(FileManager.default.fileExists(atPath: url.path))
        }
    }

    @Test func everySectionRenders() throws {
        for section in AppSection.allCases {
            let model = ShellModel(selection: section, defaults: nil)
            let view = Snapshot.framed(RootView(model: model, chrome: .inline), framing: .mockup)
            #expect(ImageRenderer(content: view).cgImage != nil, "\(section) did not render")
        }
    }
}

/// Screens contribute top-bar actions through `shellToolbar`.
@Suite struct ShellToolbarSlotTests {
    final class Seen { var ids: [String] = [] }

    @Test func itemsFromTheScreenReachTheShell() {
        let seen = Seen()
        let screen = VStack {
            Text("Mail").shellToolbar(id: "mail.compose") { Button("Compose") {} }
            Text("List").shellToolbar(id: "mail.refresh") { Button("Refresh") {} }
        }
        let view = screen.overlayPreferenceValue(ShellToolbarKey.self) { items in
            let _ = (seen.ids = items.map(\.id))
            Color.clear
        }
        #expect(ImageRenderer(content: view.frame(width: 300, height: 100)).cgImage != nil)
        #expect(seen.ids == ["mail.compose", "mail.refresh"])
    }

    @Test func screenActionsTakeTheGeneralControlsPlace() {
        #expect(ShellTrailingControls.showsGeneralControls(screenItems: []))
        let scan = ShellToolbarItem(id: "scan.actions", view: AnyView(Text("Open file")))
        #expect(!ShellTrailingControls.showsGeneralControls(screenItems: [scan]))
    }

    @Test func itemsAreEqualByID() {
        let a = ShellToolbarItem(id: "x", view: AnyView(Text("1")))
        let b = ShellToolbarItem(id: "x", view: AnyView(Text("2")))
        #expect(a == b)
    }
}

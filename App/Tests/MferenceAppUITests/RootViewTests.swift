import MferenceAppCore
import MferenceAppUI
import SwiftUI
import Testing

@Suite struct RootViewTests {
    @Test func rootViewRenders() {
        let root = RootView(model: ShellModel(defaults: nil), chrome: .inline)
        let renderer = ImageRenderer(content: root.frame(width: 960, height: 640))
        #expect(renderer.nsImage != nil)
    }

    @Test func pillsFollowTheMockupOrder() {
        #expect(AppSection.pills.map(\.title) == ["Chat", "Mail", "Calendar", "Notes", "Errands", "Scan"])
    }
}

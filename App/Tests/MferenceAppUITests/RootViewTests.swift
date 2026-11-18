import MferenceAppCore
import MferenceAppUI
import SwiftUI
import Testing

@Suite struct RootViewTests {
    @Test func rootViewRenders() {
        let renderer = ImageRenderer(content: RootView().frame(width: 960, height: 640))
        #expect(renderer.nsImage != nil)
    }

    @Test func pillsFollowTheMockupOrder() {
        #expect(AppSection.pills.map(\.title) == ["Chat", "Mail", "Calendar", "Notes", "Errands", "Scan"])
    }
}

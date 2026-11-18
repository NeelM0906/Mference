import MferenceAppCore
import SwiftData
import SwiftUI

extension View {
    /// Everything the screens read from the environment, in one call at the
    /// root: U1's `ScreenModels`, U4's Scan, Calendar and Mail models, and
    /// the SwiftData container.
    public func mferenceEnvironment(services: AppServices, screens: ScreenModels) -> some View {
        modifier(MferenceEnvironment(services: services, screens: screens))
    }
}

struct MferenceEnvironment: ViewModifier {
    let services: AppServices
    let screens: ScreenModels

    func body(content: Content) -> some View {
        let base = content
            .environment(screens)
            .modelContainer(services.modelContainer)
        if let apple = services.apple {
            base
                .environment(apple.scan)
                .environment(apple.calendar)
                .environment(apple.mail)
        } else {
            base
        }
    }
}

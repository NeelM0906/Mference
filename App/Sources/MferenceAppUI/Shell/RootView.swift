import MferenceAppCore
import MferenceDesign
import SwiftUI

/// Placeholder root until U1 lands the shell (window chrome, pill bar,
/// routing). Screens register here by section.
public struct RootView: View {
    @State private var section: AppSection = .chat

    public init() {}

    public var body: some View {
        VStack(spacing: MFSpacing.l) {
            Text("Mference")
                .font(MFFont.title)
                .foregroundStyle(MFColor.ink)
            Text(section.title)
                .font(MFFont.body)
                .foregroundStyle(MFColor.secondary)
        }
        .frame(minWidth: 960, minHeight: 640)
        .background(MFColor.canvas)
    }
}

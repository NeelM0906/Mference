import MferenceAppCore
import MferenceDesign
import SwiftUI

/// The empty slot a team's screen replaces. Shown only until that screen
/// lands; it says so plainly instead of drawing sample content.
public struct SectionPlaceholderView: View {
    let section: AppSection

    public init(section: AppSection) { self.section = section }

    public var body: some View {
        VStack(spacing: MFSpacing.m) {
            Image(systemName: section.systemImage)
                .font(MFFont.largeTitle)
                .foregroundStyle(MFColor.tertiary)
                .accessibilityHidden(true)
            Text(section.title)
                .font(MFFont.title)
                .foregroundStyle(MFColor.ink)
            Text("This screen is still being built.")
                .font(MFFont.body)
                .foregroundStyle(MFColor.secondary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .accessibilityElement(children: .combine)
    }
}

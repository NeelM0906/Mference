import MferenceAppCore
import MferenceDesign
import SwiftUI

/// The honest screen for a section that is not built yet (Errands, P7): what
/// it will do, and that nothing of it runs today.
public struct ComingLaterView: View {
    let section: AppSection

    public init(section: AppSection) { self.section = section }

    struct Copy {
        let headline: String
        let detail: String
        let promise: String
    }

    static func copy(for section: AppSection) -> Copy {
        switch section {
        case .errands:
            Copy(headline: "Errands, with your approval",
                 detail: "Hand off bookings, orders and forms to an assistant working in its own browser window. This arrives in a later version.",
                 promise: "Every purchase, booking, send or delete will stop and wait for you.")
        default:
            Copy(headline: section.title, detail: "This arrives in a later version.", promise: "")
        }
    }

    public var body: some View {
        let copy = Self.copy(for: section)
        VStack(spacing: MFSpacing.l) {
            Image(systemName: section.systemImage)
                .font(MFFont.largeTitle)
                .foregroundStyle(MFColor.tertiary)
                .accessibilityHidden(true)
            MFBadge(.custom("Coming later", .preview))
            Text(copy.headline)
                .font(MFFont.largeTitle)
                .foregroundStyle(MFColor.ink)
                .multilineTextAlignment(.center)
            Text(copy.detail)
                .font(MFFont.body)
                .foregroundStyle(MFColor.secondary)
                .multilineTextAlignment(.center)
            if !copy.promise.isEmpty {
                Text(copy.promise)
                    .font(MFFont.callout)
                    .foregroundStyle(MFColor.tertiary)
                    .multilineTextAlignment(.center)
            }
        }
        .frame(maxWidth: ShellMetrics.comingLaterWidth)
        .padding(MFSpacing.xxl)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .accessibilityElement(children: .combine)
    }
}

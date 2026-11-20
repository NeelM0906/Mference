import MferenceDesign
import SwiftUI

/// A drop-down field as the boards draw it (`.sel`): the current choice and a
/// chevron in a bordered rounded box; the menu holds the choices.
struct VoiceMenuField<Items: View>: View {
    let title: String
    let accessibilityLabel: String
    let items: Items

    init(_ title: String, accessibilityLabel: String, @ViewBuilder items: () -> Items) {
        self.title = title
        self.accessibilityLabel = accessibilityLabel
        self.items = items()
    }

    var body: some View {
        Menu {
            items
        } label: {
            HStack(spacing: MFSpacing.s) {
                Text(title)
                    .lineLimit(1)
                Image(systemName: "chevron.down")
                    .font(MFFont.micro)
                    .foregroundStyle(MFColor.secondary)
                    .accessibilityHidden(true)
            }
            .font(MFFont.callout)
            .foregroundStyle(MFColor.ink)
            .padding(.horizontal, MFSpacing.compact)
            .frame(height: VoiceMetrics.fieldHeight)
            .background(RoundedRectangle(cornerRadius: MFRadius.s).fill(MFColor.surface))
            .overlay(RoundedRectangle(cornerRadius: MFRadius.s).strokeBorder(MFColor.hairlineStrong,
                                                                            lineWidth: MFLayout.hairline))
            .contentShape(Rectangle())
        }
        .menuStyle(.button)
        .buttonStyle(.plain)
        .menuIndicator(.hidden)
        .fixedSize()
        .accessibilityLabel(accessibilityLabel)
        .accessibilityValue(title)
    }
}

/// Sizes the voice screens share.
enum VoiceMetrics {
    /// A drop-down field's height and chevron.
    static let fieldHeight: CGFloat = 30
}

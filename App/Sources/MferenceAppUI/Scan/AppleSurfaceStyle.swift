import MferenceDesign
import SwiftUI

// Small building blocks shared by the Scan, Calendar and Mail screens, from
// MferenceDesign tokens only. They move to U1's MferenceDesign components
// once those land on mference-app.

/// A rounded capsule chip ("Sure", "Check", "2 trackers removed").
struct AppleChip: View {
    enum Tone { case success, preview, accent, neutral }

    var text: String
    var tone: Tone = .neutral

    var body: some View {
        Text(text)
            .font(MFFont.calloutEmphasis)
            .foregroundStyle(foreground)
            .padding(.horizontal, MFSpacing.s + MFSpacing.xxs)
            .padding(.vertical, MFSpacing.xs)
            .background(Capsule().fill(background))
    }

    private var foreground: Color {
        switch tone {
        case .success: MFColor.success
        case .preview: MFColor.preview
        case .accent: MFColor.accentStrong
        case .neutral: MFColor.secondary
        }
    }

    private var background: Color {
        switch tone {
        case .success: MFColor.successFill
        case .preview: MFColor.previewFill
        case .accent: MFColor.accentFill
        case .neutral: MFColor.fill
        }
    }
}

/// Primary (ink) and secondary (outlined) buttons from the mockups.
struct AppleButtonStyle: ButtonStyle {
    enum Kind { case primary, secondary, quiet }

    var kind: Kind = .secondary

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(MFFont.bodyEmphasis)
            .foregroundStyle(kind == .primary ? MFColor.onInk : MFColor.ink)
            .padding(.horizontal, MFSpacing.l)
            .padding(.vertical, MFSpacing.s)
            .background(
                Capsule().fill(kind == .primary ? MFColor.ink : (kind == .quiet ? MFColor.fill : MFColor.surface)))
            .overlay(Capsule().strokeBorder(kind == .secondary ? MFColor.hairlineStrong : .clear))
            .opacity(configuration.isPressed ? 0.8 : 1)
            .contentShape(Capsule())
    }
}

extension ButtonStyle where Self == AppleButtonStyle {
    static var applePrimary: AppleButtonStyle { AppleButtonStyle(kind: .primary) }
    static var appleSecondary: AppleButtonStyle { AppleButtonStyle(kind: .secondary) }
    static var appleQuiet: AppleButtonStyle { AppleButtonStyle(kind: .quiet) }
}

/// Uppercase tracked section header ("ALSO READS", "SORTED FOR YOU").
struct AppleSectionHeader: View {
    var text: String

    var body: some View {
        Text(text.uppercased())
            .font(MFFont.sectionHeader)
            .tracking(0.8)
            .foregroundStyle(MFColor.tertiary)
            .accessibilityAddTraits(.isHeader)
    }
}

/// The accent-tinted card for one-click actions (Add row, Add to calendar).
struct AppleActionCard<Content: View>: View {
    var dashed = false
    @ViewBuilder var content: Content

    var body: some View {
        content
            .padding(MFSpacing.l)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(RoundedRectangle(cornerRadius: MFRadius.l).fill(MFColor.accentFillFaint))
            .overlay(RoundedRectangle(cornerRadius: MFRadius.l)
                .strokeBorder(MFColor.accentFill, style: StrokeStyle(lineWidth: 1, dash: dashed ? [4, 3] : [])))
    }
}

/// A message for a screen that cannot show its data yet, with one action.
struct AppleEmptyState: View {
    var systemImage: String
    var title: String
    var message: String
    var actionTitle: String?
    var action: (() -> Void)?

    var body: some View {
        VStack(spacing: MFSpacing.m) {
            Image(systemName: systemImage)
                .font(.system(size: 30, weight: .regular))
                .foregroundStyle(MFColor.tertiary)
                .accessibilityHidden(true)
            Text(title)
                .font(MFFont.headline)
                .foregroundStyle(MFColor.ink)
            Text(message)
                .font(MFFont.body)
                .foregroundStyle(MFColor.secondary)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 420)
            if let actionTitle, let action {
                Button(actionTitle, action: action)
                    .buttonStyle(.applePrimary)
                    .padding(.top, MFSpacing.xs)
            }
        }
        .padding(MFSpacing.xxl)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .accessibilityElement(children: .contain)
    }
}

/// Diagonal stripes behind the scanned image (mockup 07).
struct StripedBackground: View {
    var body: some View {
        Canvas { context, size in
            context.fill(Path(CGRect(origin: .zero, size: size)), with: .color(MFColor.fill))
            var path = Path()
            let step: CGFloat = 14
            var x: CGFloat = -size.height
            while x < size.width {
                path.move(to: CGPoint(x: x, y: size.height))
                path.addLine(to: CGPoint(x: x + size.height, y: 0))
                x += step
            }
            context.stroke(path, with: .color(MFColor.hairlineFaint), lineWidth: 5)
        }
        .accessibilityHidden(true)
    }
}

/// Shows the content whole when it fits, else in a scroll view (snapshots
/// render only the first: `ImageRenderer` does not draw scroll contents).
struct FitOrScroll<Content: View>: View {
    @ViewBuilder var content: Content

    var body: some View {
        ViewThatFits(in: .vertical) {
            content
            ScrollView { content }.scrollIndicators(.never)
        }
        .frame(maxHeight: .infinity, alignment: .top)
    }
}

extension EnvironmentValues {
    /// Snapshot tests draw text fields as plain text (`ImageRenderer` cannot
    /// draw AppKit-backed controls). Never set by the app.
    @Entry var appleStaticRendering = false
}

/// A one-line text field that snapshots draw as its text.
struct AppleTextField: View {
    var title: String
    @Binding var text: String
    @Environment(\.appleStaticRendering) private var staticRendering

    var body: some View {
        if staticRendering {
            Text(text.isEmpty ? title : text)
                .foregroundStyle(text.isEmpty ? MFColor.tertiary : MFColor.ink)
                .frame(maxWidth: .infinity, alignment: .leading)
        } else {
            TextField(title, text: $text).textFieldStyle(.plain)
        }
    }
}

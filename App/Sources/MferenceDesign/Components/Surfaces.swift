import SwiftUI

// Cards, list rows, the approval card and the frosted window background
// (`.box`, `.r`, the errand approval card and the board backdrop).

// MARK: - Card

/// A white card with a hairline border (`.box`). Its direct subviews are
/// stacked with a faint hairline between them, so a card of `MFListRow`s
/// reads like the Settings boxes.
public struct MFCard<Content: View>: View {
    let content: Content

    public init(@ViewBuilder content: () -> Content) {
        self.content = content()
    }

    public var body: some View {
        VStack(spacing: 0) {
            Group(subviews: content) { subviews in
                ForEach(Array(subviews.enumerated()), id: \.element.id) { index, subview in
                    if index > 0 {
                        Rectangle().fill(MFColor.hairlineFaint).frame(height: 1)
                    }
                    subview
                }
            }
        }
        .background(MFColor.surface)
        .clipShape(RoundedRectangle(cornerRadius: MFRadius.l))
        .overlay(RoundedRectangle(cornerRadius: MFRadius.l).strokeBorder(MFColor.hairline))
    }
}

// MARK: - List row

/// A card row: bold title, secondary line, and trailing controls (badges,
/// buttons, a toggle) on the right (`.r`).
public struct MFListRow<Trailing: View>: View {
    let title: String
    let subtitle: String?
    let trailing: Trailing

    public init(_ title: String, subtitle: String? = nil, @ViewBuilder trailing: () -> Trailing) {
        self.title = title
        self.subtitle = subtitle
        self.trailing = trailing()
    }

    public var body: some View {
        HStack(alignment: .center, spacing: MFSpacing.m + 2) {
            VStack(alignment: .leading, spacing: MFSpacing.xxs) {
                Text(title)
                    .font(MFFont.rowTitle)
                    .foregroundStyle(MFColor.ink)
                if let subtitle {
                    Text(subtitle)
                        .font(MFFont.rowSubtitle)
                        .foregroundStyle(MFColor.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .accessibilityElement(children: .combine)
            HStack(spacing: MFSpacing.m - 2) {
                trailing
            }
        }
        .padding(.horizontal, MFSpacing.l)
        .padding(.vertical, MFSpacing.m)
    }
}

extension MFListRow where Trailing == EmptyView {
    public init(_ title: String, subtitle: String? = nil) {
        self.init(title, subtitle: subtitle) { EmptyView() }
    }
}

// MARK: - Approval card

/// The amber card that stops an action until the person approves it (L10).
public struct MFApprovalCard<Actions: View>: View {
    let eyebrow: String
    let title: String
    let detail: String?
    let actions: Actions

    /// - Parameters:
    ///   - eyebrow: the uppercase header, e.g. "Approve to book".
    ///   - title: what will happen.
    ///   - detail: terms, costs, recipients.
    public init(eyebrow: String, title: String, detail: String? = nil, @ViewBuilder actions: () -> Actions) {
        self.eyebrow = eyebrow
        self.title = title
        self.detail = detail
        self.actions = actions()
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: MFSpacing.m - 2) {
            Text(eyebrow.uppercased())
                .font(MFFont.sectionHeader)
                .tracking(0.7)
                .foregroundStyle(MFColor.preview)
                .accessibilityAddTraits(.isHeader)
            Text(title)
                .font(MFFont.bodyEmphasis)
                .fontWeight(.semibold)
                .foregroundStyle(MFColor.ink)
            if let detail {
                Text(detail)
                    .font(MFFont.callout)
                    .foregroundStyle(MFColor.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            HStack(spacing: MFSpacing.s) {
                actions
            }
        }
        .padding(MFSpacing.m + 2)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(RoundedRectangle(cornerRadius: MFRadius.l).fill(MFColor.previewFill.opacity(0.55)))
        .overlay(RoundedRectangle(cornerRadius: MFRadius.l).strokeBorder(MFColor.preview.opacity(0.25)))
        .accessibilityElement(children: .contain)
    }
}

// MARK: - Window background

/// The frosted window: `.regularMaterial` over the pastel wallpaper.
/// `clipped` rounds it to `MFRadius.window` (for snapshots and sheets; the
/// real window is rounded by the system).
public struct MFWindowBackground: View {
    let clipped: Bool

    public init(clipped: Bool = false) { self.clipped = clipped }

    public var body: some View {
        ZStack {
            Rectangle().fill(MFColor.wallpaper)
            Rectangle().fill(.regularMaterial)
            // The mockup window is white at 92% over the wallpaper.
            Rectangle().fill(MFColor.surface.opacity(0.88))
        }
        .clipShape(RoundedRectangle(cornerRadius: clipped ? MFRadius.window : 0))
        .ignoresSafeArea()
        .accessibilityHidden(true)
    }
}

extension View {
    /// Puts the frosted window background behind this view.
    public func mfWindowBackground(clipped: Bool = false) -> some View {
        background(MFWindowBackground(clipped: clipped))
    }
}

// MARK: - Scrolling

extension EnvironmentValues {
    /// True while a view is rendered off screen (snapshots): `MFScroll` then
    /// lays its content out in full instead of in a `ScrollView`, which
    /// `ImageRenderer` does not draw.
    @Entry public var mfRendersStatically = false
}

/// A vertical scroll view that snapshots can render (see `mfRendersStatically`).
public struct MFScroll<Content: View>: View {
    @Environment(\.mfRendersStatically) private var rendersStatically
    let content: Content

    public init(@ViewBuilder content: () -> Content) {
        self.content = content()
    }

    public var body: some View {
        if rendersStatically {
            content
                .frame(maxHeight: .infinity, alignment: .top)
        } else {
            ScrollView { content }
        }
    }
}

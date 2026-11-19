import MferenceAppCore
import MferenceDesign
import SwiftUI

/// "Searched the web · 3 searches · 7 pages read ▾" (mockup 02, expanded in
/// 02c): the queries that left the Mac, the pages read, and the model's
/// reasoning, collapsed by default.
struct StepsDisclosure: View {
    let title: String
    let steps: [ChatStep]
    let reasoning: String?
    @State private var isExpanded: Bool

    init(title: String, steps: [ChatStep], reasoning: String?, isExpanded: Bool = false) {
        self.title = title
        self.steps = steps
        self.reasoning = reasoning
        _isExpanded = State(initialValue: isExpanded)
    }

    private var usedWeb: Bool { !steps.isEmpty }

    var body: some View {
        VStack(alignment: .leading, spacing: MFSpacing.s + MFSpacing.xxs) {
            Button {
                isExpanded.toggle()
            } label: {
                HStack(spacing: MFSpacing.xs + 1) {
                    Image(systemName: usedWeb ? "globe" : "sparkles")
                        .imageScale(.small)
                    Text(title)
                    Image(systemName: isExpanded ? "chevron.up" : "chevron.down")
                        .imageScale(.small)
                        .foregroundStyle(MFColor.tertiary)
                }
                .font(MFFont.chip)
                .foregroundStyle(MFColor.secondary)
                .padding(.horizontal, MFSpacing.s + 1)
                .frame(height: ChatMetrics.controlHeight)
                .background(MFColor.fill.opacity(0.7), in: Capsule())
                .contentShape(Capsule())
            }
            .buttonStyle(.plain)
            .accessibilityLabel(title)
            .accessibilityValue(isExpanded ? "Expanded" : "Collapsed")
            .accessibilityHint("Shows the searches and pages behind this answer")

            if isExpanded {
                details
            }
        }
    }

    private var details: some View {
        VStack(alignment: .leading, spacing: MFSpacing.xs + MFSpacing.xxs) {
            ForEach(Array(steps.filter { $0.kind != .readPage }.enumerated()), id: \.offset) { _, step in
                stepLine(step)
            }
            if let pages = pagesLine {
                Text(pages)
                    .font(MFFont.preview)
                    .foregroundStyle(MFColor.secondary)
                    .lineLimit(2)
            }
            if let reasoning, !reasoning.isEmpty {
                ScrollView {
                    Text(reasoning)
                        .font(MFFont.preview)
                        .foregroundStyle(MFColor.secondary)
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                .frame(maxHeight: ChatMetrics.reasoningMaxHeight)
                .fixedSize(horizontal: false, vertical: true)
                .accessibilityLabel("Reasoning")
            }
            if usedWeb {
                Text(ChatStepsSummary.privacyNote)
                    .font(MFFont.caption)
                    .foregroundStyle(MFColor.tertiary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal, MFSpacing.m)
        .padding(.vertical, MFSpacing.s + MFSpacing.xxs)
        .background(MFColor.canvas, in: RoundedRectangle(cornerRadius: MFRadius.m))
        .overlay(RoundedRectangle(cornerRadius: MFRadius.m).strokeBorder(MFColor.hairlineFaint))
    }

    /// "Read 7 pages: stride-desks.example, oakline.example, …"
    private var pagesLine: String? {
        let hosts = steps.filter { $0.kind == .readPage }.compactMap { $0.url?.host() }
        guard !hosts.isEmpty else { return nil }
        var unique: [String] = []
        for host in hosts where !unique.contains(host) { unique.append(host) }
        let shown = unique.prefix(4).joined(separator: ", ") + (unique.count > 4 ? ", …" : "")
        return hosts.count == 1 ? "Read 1 page: \(shown)" : "Read \(hosts.count) pages: \(shown)"
    }

    @ViewBuilder
    private func stepLine(_ step: ChatStep) -> some View {
        switch step.kind {
        case .search:
            HStack(spacing: 0) {
                Text("“\(step.query ?? "")”").font(MFFont.preview.monospaced())
                Text(" · \(Self.results(step.resultCount ?? 0))").font(MFFont.preview)
            }
            .foregroundStyle(MFColor.secondary)
            .lineLimit(1)
            .accessibilityElement(children: .combine)
        case .readPage:
            Text("Read \(step.title.map { "“\($0)” · " } ?? "")\(step.url?.host() ?? "")")
                .font(MFFont.preview)
                .foregroundStyle(MFColor.secondary)
                .lineLimit(1)
        case .failed:
            Text(step.detail ?? "A web step failed.")
                .font(MFFont.preview)
                .foregroundStyle(MFColor.preview)
        }
    }

    static func results(_ count: Int) -> String {
        count == 1 ? "1 result" : "\(count) results"
    }
}

/// Numbered source chips under an answer, with a hover card each.
struct SourcesRow: View {
    let citations: [ChatCitation]

    var body: some View {
        FlowLayout(spacing: MFSpacing.xs + MFSpacing.xxs) {
            ForEach(citations) { citation in
                SourceChip(citation: citation)
            }
        }
        .accessibilityElement(children: .contain)
        .accessibilityLabel("Sources")
    }
}

struct SourceChip: View {
    let citation: ChatCitation
    @Environment(\.openURL) private var openURL
    @State private var showsCard = false

    var body: some View {
        Button {
            openURL(citation.url)
        } label: {
            HStack(spacing: MFSpacing.xs + 1) {
                Text("\(citation.number)")
                    .font(MFFont.chip.weight(.semibold))
                    .foregroundStyle(MFColor.accent)
                Text(citation.label)
                    .font(MFFont.chip)
                    .foregroundStyle(MFColor.secondary)
                    .lineLimit(1)
            }
            .padding(.horizontal, MFSpacing.s + 1)
            .frame(height: ChatMetrics.controlHeight)
            .background(MFColor.surface, in: Capsule())
            .overlay(Capsule().strokeBorder(MFColor.hairline))
            .contentShape(Capsule())
        }
        .buttonStyle(.plain)
        .onHover { showsCard = $0 }
        .popover(isPresented: $showsCard, arrowEdge: .bottom) {
            SourceCard(citation: citation)
        }
        .accessibilityLabel("Source \(citation.number), \(citation.title)")
        .accessibilityHint("Opens \(citation.label) in your browser")
    }
}

struct SourceCard: View {
    let citation: ChatCitation

    var body: some View {
        VStack(alignment: .leading, spacing: MFSpacing.xs) {
            Text(citation.title)
                .font(MFFont.calloutEmphasis)
                .foregroundStyle(MFColor.ink)
                .lineLimit(2)
            Text(citation.url.host() ?? citation.url.absoluteString)
                .font(MFFont.caption)
                .foregroundStyle(MFColor.accent)
            if let snippet = citation.snippet, !snippet.isEmpty {
                Text(snippet)
                    .font(MFFont.caption)
                    .foregroundStyle(MFColor.secondary)
                    .lineLimit(4)
            }
        }
        .frame(width: ChatMetrics.hoverCardWidth, alignment: .leading)
        .padding(MFSpacing.m)
    }
}

/// Lays children left to right, wrapping to new lines.
struct FlowLayout: Layout {
    var spacing: CGFloat

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let rows = arrange(width: proposal.width ?? .infinity, subviews: subviews)
        let width = rows.map(\.width).max() ?? 0
        let height = rows.map(\.height).reduce(0, +) + spacing * CGFloat(max(rows.count - 1, 0))
        return CGSize(width: proposal.width.map { min($0, width) } ?? width, height: height)
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        var y = bounds.minY
        for row in arrange(width: bounds.width, subviews: subviews) {
            var x = bounds.minX
            for index in row.indices {
                let size = subviews[index].sizeThatFits(.unspecified)
                subviews[index].place(at: CGPoint(x: x, y: y + (row.height - size.height) / 2),
                                      proposal: ProposedViewSize(size))
                x += size.width + spacing
            }
            y += row.height + spacing
        }
    }

    private struct Row {
        var indices: [Int] = []
        var width: CGFloat = 0
        var height: CGFloat = 0
    }

    private func arrange(width: CGFloat, subviews: Subviews) -> [Row] {
        var rows: [Row] = []
        var current = Row()
        for index in subviews.indices {
            let size = subviews[index].sizeThatFits(.unspecified)
            let needed = current.indices.isEmpty ? size.width : current.width + spacing + size.width
            if needed > width, !current.indices.isEmpty {
                rows.append(current)
                current = Row()
            }
            current.width = current.indices.isEmpty ? size.width : current.width + spacing + size.width
            current.height = max(current.height, size.height)
            current.indices.append(index)
        }
        if !current.indices.isEmpty { rows.append(current) }
        return rows
    }
}

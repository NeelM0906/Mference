import MferenceAppCore
import MferenceDesign
import SwiftUI

/// An answer rendered from Markdown, with numbered citation chips that open
/// their source.
struct MarkdownView: View {
    let text: String
    let citations: [ChatCitation]

    var body: some View {
        VStack(alignment: .leading, spacing: MFSpacing.m) {
            ForEach(Array(MarkdownBlock.parse(text).enumerated()), id: \.offset) { _, block in
                blockView(block)
            }
        }
        .foregroundStyle(MFColor.ink)
        .textSelection(.enabled)
        .environment(\.openURL, OpenURLAction { url in
            // Only web links and citation chips open; a model can write any
            // URL scheme into its answer.
            switch url.scheme?.lowercased() {
            case InlineText.citationScheme:
                let number = Int(url.host() ?? url.path()) ?? 0
                guard let source = citations.first(where: { $0.number == number }),
                      InlineText.isWebLink(source.url) else { return .discarded }
                return .systemAction(source.url)
            case "http", "https":
                return .systemAction(url)
            default:
                return .discarded
            }
        })
    }

    @ViewBuilder
    private func blockView(_ block: MarkdownBlock) -> some View {
        let inline = InlineText(citations: citations)
        switch block {
        case .paragraph(let text):
            Text(inline.attributed(text)).font(MFFont.body)
                .lineSpacing(MFLineSpacing.relaxed)
                .fixedSize(horizontal: false, vertical: true)
        case .heading(let level, let text):
            Text(inline.attributed(text))
                .font(level <= 2 ? MFFont.headline : MFFont.bodyEmphasis)
                .padding(.top, MFSpacing.xs)
        case .bulletList(let items):
            VStack(alignment: .leading, spacing: MFSpacing.xs) {
                ForEach(Array(items.enumerated()), id: \.offset) { _, item in
                    HStack(alignment: .firstTextBaseline, spacing: MFSpacing.s) {
                        Text("•").foregroundStyle(MFColor.tertiary)
                        Text(inline.attributed(item)).fixedSize(horizontal: false, vertical: true)
                    }
                }
            }
            .font(MFFont.body)
        case .orderedList(let start, let items):
            VStack(alignment: .leading, spacing: MFSpacing.xs) {
                ForEach(Array(items.enumerated()), id: \.offset) { offset, item in
                    HStack(alignment: .firstTextBaseline, spacing: MFSpacing.s) {
                        Text("\(start + offset).").foregroundStyle(MFColor.tertiary).monospacedDigit()
                        Text(inline.attributed(item)).fixedSize(horizontal: false, vertical: true)
                    }
                }
            }
            .font(MFFont.body)
        case .code(_, let code):
            Text(code)
                .font(MFFont.callout.monospaced())
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(MFSpacing.m)
                .background(MFColor.fill, in: RoundedRectangle(cornerRadius: MFRadius.m))
        case .quote(let text):
            HStack(spacing: MFSpacing.m) {
                Rectangle().fill(MFColor.hairlineStrong).frame(width: MFSpacing.xxs + 1)
                Text(inline.attributed(text)).foregroundStyle(MFColor.secondary)
            }
            .font(MFFont.body)
        case .table(let header, let rows):
            AnswerTable(header: header, rows: rows, inline: inline)
        case .rule:
            Rectangle().fill(MFColor.hairline).frame(height: ChatMetrics.hairlineWidth)
        }
    }
}

/// A pipe table (mockup 02's warranty table).
struct AnswerTable: View {
    let header: [String]
    let rows: [[String]]
    let inline: InlineText

    var body: some View {
        Grid(alignment: .leading, horizontalSpacing: 0, verticalSpacing: 0) {
            GridRow {
                ForEach(Array(header.enumerated()), id: \.offset) { _, cell in
                    Text(inline.attributed(cell))
                        .font(MFFont.calloutSemibold)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(.horizontal, MFSpacing.m)
                        .padding(.vertical, MFSpacing.s)
                }
            }
            .background(MFColor.fillSubtle.opacity(0.5))
            ForEach(Array(rows.enumerated()), id: \.offset) { _, row in
                Rectangle().fill(MFColor.hairlineFaint).frame(height: ChatMetrics.hairlineWidth)
                    .gridCellUnsizedAxes(.horizontal)
                GridRow {
                    ForEach(Array(row.enumerated()), id: \.offset) { _, cell in
                        Text(inline.attributed(cell))
                            .font(MFFont.callout)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding(.horizontal, MFSpacing.m)
                            .padding(.vertical, MFSpacing.s)
                    }
                }
            }
        }
        .clipShape(RoundedRectangle(cornerRadius: MFRadius.m + MFSpacing.xxs))
        .overlay(RoundedRectangle(cornerRadius: MFRadius.m + MFSpacing.xxs).strokeBorder(MFColor.hairlineFaint))
    }
}

/// Inline Markdown (bold, italics, links, code spans) plus citation chips.
struct InlineText {
    static let citationScheme = "mference-cite"

    let citations: [ChatCitation]

    func attributed(_ text: String) -> AttributedString {
        let valid = Set(citations.map(\.number))
        var result = AttributedString()
        for segment in CitationMarkers.segments(text, valid: valid) {
            switch segment {
            case .text(let run):
                result += Self.markdown(run)
            case .citation(let number):
                var gap = AttributedString("\u{2009}")
                gap.swiftUI.font = MFFont.micro
                var chip = AttributedString("\u{2009}\(number)\u{2009}")
                chip.swiftUI.font = MFFont.micro
                chip.swiftUI.foregroundColor = MFColor.accent
                chip.swiftUI.backgroundColor = MFColor.accentFill
                chip.swiftUI.baselineOffset = 1
                chip.link = URL(string: "\(Self.citationScheme)://\(number)")
                result += gap + chip
            }
        }
        return result
    }

    static func markdown(_ text: String) -> AttributedString {
        let options = AttributedString.MarkdownParsingOptions(
            interpretedSyntax: .inlineOnlyPreservingWhitespace, failurePolicy: .returnPartiallyParsedIfPossible)
        guard var parsed = try? AttributedString(markdown: text, options: options) else {
            return AttributedString(text)
        }
        for run in parsed.runs {
            guard let link = run.link else { continue }
            if isWebLink(link) {
                parsed[run.range].swiftUI.foregroundColor = MFColor.accent
            } else {
                parsed[run.range].link = nil
            }
        }
        return parsed
    }

    static func isWebLink(_ url: URL) -> Bool {
        guard let scheme = url.scheme?.lowercased() else { return false }
        return scheme == "http" || scheme == "https"
    }
}

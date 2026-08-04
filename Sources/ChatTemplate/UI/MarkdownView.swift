import AppKit
import ChatTemplateCore
import SwiftUI

/// Renders assistant markdown: headings, paragraphs, lists, quotes,
/// dividers, GFM tables, and fenced code blocks with a copy button —
/// mirroring the template's custom markdown renderer.
public struct MarkdownView: View {
    public let text: String

    public init(text: String) {
        self.text = text
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            ForEach(Array(MarkdownBlockParser.parse(text).enumerated()), id: \.offset) { _, block in
                blockView(block)
            }
        }
        .textSelection(.enabled)
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    @ViewBuilder
    private func blockView(_ block: MarkdownBlock) -> some View {
        switch block {
        case .heading(let level, let text):
            Text(inlineMarkdown(text))
                .font(headingFont(level))
                .padding(.top, 4)
        case .paragraph(let text):
            Text(inlineMarkdown(text))
        case .codeBlock(let language, let code):
            CodeBlockView(language: language, code: code)
        case .bulletList(let items):
            listView(items: items) { _ in Text("•") }
        case .orderedList(let items):
            listView(items: items) { index in
                Text("\(index + 1).").monospacedDigit()
            }
        case .quote(let text):
            HStack(spacing: 10) {
                RoundedRectangle(cornerRadius: 2)
                    .fill(.tertiary)
                    .frame(width: 3)
                Text(inlineMarkdown(text))
                    .foregroundStyle(.secondary)
            }
            .fixedSize(horizontal: false, vertical: true)
        case .table(let header, let rows):
            TableBlockView(header: header, rows: rows)
        case .divider:
            Divider()
        }
    }

    private func listView(
        items: [String],
        marker: @escaping (Int) -> Text
    ) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            ForEach(Array(items.enumerated()), id: \.offset) { index, item in
                HStack(alignment: .firstTextBaseline, spacing: 8) {
                    marker(index)
                        .foregroundStyle(.secondary)
                    Text(inlineMarkdown(item))
                }
            }
        }
        .padding(.leading, 4)
    }

    private func headingFont(_ level: Int) -> Font {
        switch level {
        case 1: .title2.weight(.bold)
        case 2: .title3.weight(.semibold)
        default: .headline
        }
    }
}

/// Inline markdown (bold, italics, inline code, links) via Foundation's
/// markdown-aware AttributedString.
func inlineMarkdown(_ text: String) -> AttributedString {
    (try? AttributedString(
        markdown: text,
        options: AttributedString.MarkdownParsingOptions(
            interpretedSyntax: .inlineOnlyPreservingWhitespace)))
        ?? AttributedString(text)
}

public struct CodeBlockView: View {
    public let language: String?
    public let code: String

    @State private var copied = false

    public init(language: String?, code: String) {
        self.language = language
        self.code = code
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Text(language ?? "code")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Spacer()
                Button {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(code, forType: .string)
                    copied = true
                    Task {
                        try? await Task.sleep(for: .seconds(1.5))
                        copied = false
                    }
                } label: {
                    Label(
                        copied ? "Copied" : "Copy",
                        systemImage: copied ? "checkmark" : "doc.on.doc")
                    .font(.caption)
                }
                .buttonStyle(.plain)
                .foregroundStyle(.secondary)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 7)
            .background(Color(nsColor: .quaternarySystemFill))

            ScrollView(.horizontal) {
                Text(code)
                    .font(.system(.callout, design: .monospaced))
                    .textSelection(.enabled)
                    .padding(12)
            }
        }
        .background(Color(nsColor: .quinarySystemFill))
        .clipShape(RoundedRectangle(cornerRadius: 10))
        .overlay(
            RoundedRectangle(cornerRadius: 10)
                .strokeBorder(.separator, lineWidth: 0.5))
    }
}

public struct TableBlockView: View {
    public let header: [String]
    public let rows: [[String]]

    public init(header: [String], rows: [[String]]) {
        self.header = header
        self.rows = rows
    }

    public var body: some View {
        ScrollView(.horizontal) {
            Grid(alignment: .leading, horizontalSpacing: 0, verticalSpacing: 0) {
                GridRow {
                    ForEach(Array(header.enumerated()), id: \.offset) { _, cell in
                        Text(inlineMarkdown(cell))
                            .fontWeight(.semibold)
                            .padding(.horizontal, 12)
                            .padding(.vertical, 6)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                }
                .background(Color(nsColor: .quaternarySystemFill))
                ForEach(Array(rows.enumerated()), id: \.offset) { index, row in
                    Divider().gridCellUnsizedAxes(.horizontal)
                    GridRow {
                        ForEach(Array(row.enumerated()), id: \.offset) { _, cell in
                            Text(inlineMarkdown(cell))
                                .padding(.horizontal, 12)
                                .padding(.vertical, 6)
                                .frame(maxWidth: .infinity, alignment: .leading)
                        }
                    }
                }
            }
        }
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .strokeBorder(.separator, lineWidth: 0.5))
    }
}

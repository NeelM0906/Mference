import Testing

@testable import ChatTemplateCore

@Suite struct MarkdownBlockParserTests {
    @Test func parsesParagraphsAndHeadings() {
        let blocks = MarkdownBlockParser.parse("# Title\n\nHello **world**\nsecond line\n\nNext")
        #expect(blocks == [
            .heading(level: 1, text: "Title"),
            .paragraph("Hello **world**\nsecond line"),
            .paragraph("Next"),
        ])
    }

    @Test func parsesFencedCodeWithLanguage() {
        let blocks = MarkdownBlockParser.parse("```swift\nlet x = 1\n\nprint(x)\n```\ntail")
        #expect(blocks == [
            .codeBlock(language: "swift", code: "let x = 1\n\nprint(x)"),
            .paragraph("tail"),
        ])
    }

    @Test func toleratesUnclosedFenceWhileStreaming() {
        let blocks = MarkdownBlockParser.parse("intro\n```python\nasync for chunk")
        #expect(blocks == [
            .paragraph("intro"),
            .codeBlock(language: "python", code: "async for chunk"),
        ])
    }

    @Test func parsesLists() {
        let blocks = MarkdownBlockParser.parse("- a\n- b\n\n1. one\n2. two")
        #expect(blocks == [
            .bulletList(items: ["a", "b"]),
            .orderedList(items: ["one", "two"]),
        ])
    }

    @Test func parsesQuoteAndDivider() {
        let blocks = MarkdownBlockParser.parse("> wisdom\n> here\n\n---")
        #expect(blocks == [.quote("wisdom\nhere"), .divider])
    }

    @Test func parsesGFMTable() {
        let text = "| A | B |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |"
        let blocks = MarkdownBlockParser.parse(text)
        #expect(blocks == [
            .table(header: ["A", "B"], rows: [["1", "2"], ["3", "4"]]),
        ])
    }

    @Test func pipeWithoutSeparatorIsParagraph() {
        let blocks = MarkdownBlockParser.parse("a | b")
        #expect(blocks == [.paragraph("a | b")])
    }
}

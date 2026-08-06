import Testing

@testable import ChatTemplateCore

@Suite struct SSELineParserTests {
    @Test func parsesDataLine() {
        #expect(SSELineParser.parse(line: #"data: {"x":1}"#) == .data(#"{"x":1}"#))
    }

    @Test func parsesDoneSentinel() {
        #expect(SSELineParser.parse(line: "data: [DONE]") == .done)
    }

    @Test func ignoresBlankCommentAndOtherFields() {
        #expect(SSELineParser.parse(line: "") == nil)
        #expect(SSELineParser.parse(line: ": keep-alive") == nil)
        #expect(SSELineParser.parse(line: "event: message") == nil)
        #expect(SSELineParser.parse(line: "data:") == nil)
    }

    @Test func extractsContentDelta() {
        let json = #"{"object":"chat.completion.chunk","choices":[{"delta":{"content":"Hi"}}]}"#
        #expect(SSELineParser.contentDelta(fromChunkJSON: json) == "Hi")
    }

    @Test func skipsChunksWithoutText() {
        let role = #"{"choices":[{"delta":{"role":"assistant"}}]}"#
        let finish = #"{"choices":[{"delta":{},"finish_reason":"stop"}]}"#
        let usage = #"{"choices":[],"usage":{"total_tokens":3}}"#
        #expect(SSELineParser.contentDelta(fromChunkJSON: role) == nil)
        #expect(SSELineParser.contentDelta(fromChunkJSON: finish) == nil)
        #expect(SSELineParser.contentDelta(fromChunkJSON: usage) == nil)
        #expect(SSELineParser.contentDelta(fromChunkJSON: "not json") == nil)
    }

    @Test func decodesErrorEnvelope() {
        let json = #"{"error":{"message":"context_length_exceeded"}}"#
        #expect(OpenAICompatBackend.errorMessage(fromEventJSON: json) == "context_length_exceeded")
        #expect(OpenAICompatBackend.errorMessage(fromEventJSON: #"{"choices":[]}"#) == nil)
    }
}

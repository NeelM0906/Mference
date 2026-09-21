import Testing
@testable import MferenceCLICore

/// `--show-reasoning` streams thoughts to standard error as one framed block
/// per thought, so they never mix with the answer on standard output.
@Suite struct ReasoningEchoTests {
    @Test func theFirstDeltaOpensTheBlockAndLaterDeltasOnlyAppend() {
        var echo = ReasoningEcho()
        #expect(echo.reasoning("The user") == "[reasoning]\nThe user")
        #expect(echo.reasoning(" wants 9 more.") == " wants 9 more.")
    }

    @Test func closingEndsAnOpenBlockOnce() {
        var echo = ReasoningEcho()
        _ = echo.reasoning("391 + 9 = 400.")
        #expect(echo.close() == "\n[/reasoning]\n")
        #expect(echo.close() == "")
    }

    @Test func closingWithoutReasoningWritesNothing() {
        var echo = ReasoningEcho()
        #expect(echo.close() == "")
    }

    @Test func aThoughtThatAlreadyEndsItsLineClosesWithoutABlankLine() {
        var echo = ReasoningEcho()
        _ = echo.reasoning("391 + 9 = 400.\n")
        #expect(echo.close() == "[/reasoning]\n")
    }

    @Test func aLaterThoughtOpensANewBlock() {
        var echo = ReasoningEcho()
        _ = echo.reasoning("first")
        _ = echo.close()
        #expect(echo.reasoning("second") == "[reasoning]\nsecond")
    }

    @Test func emptyDeltasDoNotOpenABlock() {
        var echo = ReasoningEcho()
        #expect(echo.reasoning("") == "")
        #expect(echo.close() == "")
    }
}

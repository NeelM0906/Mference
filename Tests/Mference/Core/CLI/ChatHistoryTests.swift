import Mference
import Testing
@testable import MferenceCLICore

@Suite struct ChatHistoryTests {
    /// Ten tokens per message keeps the arithmetic obvious: the limit alone
    /// decides how many turns survive.
    private func tenTokensPerMessage(_ messages: [MFTokenizer.Message]) -> Int {
        messages.count * 10
    }

    private func user(_ content: String) -> MFTokenizer.Message {
        MFTokenizer.Message(role: .user, content: content)
    }

    private func assistant(_ content: String) -> MFTokenizer.Message {
        MFTokenizer.Message(role: .assistant, content: content)
    }

    @Test func historyUnderTheLimitIsUnchanged() {
        let history = [user("one"), assistant("two"), user("three")]
        let trimmed = trimChatHistory(history, limit: 100, measure: tenTokensPerMessage)
        #expect(trimmed.messages == history)
        #expect(trimmed.dropped == 0)
        #expect(trimmed.tokens == 30)
        #expect(trimmed.fits)
    }

    @Test func oldestTurnsDropUntilTheHistoryFits() {
        let history = [user("1"), assistant("2"), user("3"), assistant("4"), user("5")]
        let trimmed = trimChatHistory(history, limit: 35, measure: tenTokensPerMessage)
        #expect(trimmed.messages == [user("3"), assistant("4"), user("5")])
        #expect(trimmed.dropped == 2)
        #expect(trimmed.tokens == 30)
        #expect(trimmed.fits)
    }

    @Test func systemMessageSurvivesTrimming() {
        let system = MFTokenizer.Message(role: .system, content: "be terse")
        let history = [system, user("1"), assistant("2"), user("3")]
        let trimmed = trimChatHistory(history, limit: 25, measure: tenTokensPerMessage)
        #expect(trimmed.messages == [system, user("3")])
        #expect(trimmed.dropped == 2)
        #expect(trimmed.fits)
    }

    @Test func newestMessageIsNeverDropped() {
        let history = [user("1"), assistant("2"), user("3")]
        let trimmed = trimChatHistory(history, limit: 5, measure: tenTokensPerMessage)
        #expect(trimmed.messages == [user("3")])
        #expect(trimmed.dropped == 2)
        #expect(trimmed.tokens == 10)
        #expect(!trimmed.fits)
    }

    @Test func systemMessageSurvivesAHistoryThatCannotFit() {
        let system = MFTokenizer.Message(role: .system, content: "be terse")
        let history = [system, user("1"), user("2")]
        let trimmed = trimChatHistory(history, limit: 15, measure: tenTokensPerMessage)
        #expect(trimmed.messages == [system, user("2")])
        #expect(!trimmed.fits)
    }

    @Test func aPromptExactlyAtTheLimitDoesNotFit() {
        let history = [user("1")]
        let trimmed = trimChatHistory(history, limit: 10, measure: tenTokensPerMessage)
        #expect(!trimmed.fits)
    }

    @Test func trimmingRerendersInsteadOfAssumingPerMessageCost() {
        // A dialect whose template adds a fixed per-render preamble: only a
        // re-render after each drop reports the true cost.
        let measure = { (messages: [MFTokenizer.Message]) in 7 + messages.count * 10 }
        let history = [user("1"), assistant("2"), user("3")]
        let trimmed = trimChatHistory(history, limit: 20, measure: measure)
        #expect(trimmed.messages == [user("3")])
        #expect(trimmed.tokens == 17)
        #expect(trimmed.fits)
    }
}

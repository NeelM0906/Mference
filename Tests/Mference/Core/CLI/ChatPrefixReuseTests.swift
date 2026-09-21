import Testing
import Mference
@testable import MferenceCLICore

/// `--reuse-prefix` asks the shared `PromptPrefixCache` whether a turn may
/// continue; these tests cover what the chat loop does with its verdict.
@Suite struct ChatPrefixReuseTests {
    private let rendered: [Int32] = [1, 2, 3, 4, 5, 6]

    @Test func aMissRunsTheRenderedPromptFromAResetCache() {
        let plan = ChatPrefixReuse.plan(match: .miss, renderedPromptIDs: rendered, maxContext: 64)
        #expect(plan.promptIDs == rendered)
        #expect(plan.start == .reset)
    }

    @Test func aHitRunsTheEffectivePromptAfterTheCachedTokens() {
        // ChatML continues from the generated tokens plus a bridge, so the
        // effective prompt is not the fresh render.
        let effective: [Int32] = [1, 2, 3, 9, 9, 7, 8]
        let plan = ChatPrefixReuse.plan(match: .hit(effectivePromptIDs: effective, cachedPromptTokens: 5),
                                        renderedPromptIDs: rendered, maxContext: 64)
        #expect(plan.promptIDs == effective)
        #expect(plan.start == .resume(cachedPromptTokens: 5))
    }

    @Test func aHitThatNoLongerFitsTheContextFallsBackToTheRenderedPrompt() {
        // History was trimmed against the rendered prompt; the effective one
        // can be longer because it keeps the turn as it was generated.
        let effective: [Int32] = [1, 2, 3, 9, 9, 7, 8, 8]
        let plan = ChatPrefixReuse.plan(match: .hit(effectivePromptIDs: effective, cachedPromptTokens: 5),
                                        renderedPromptIDs: rendered, maxContext: 8)
        #expect(plan.promptIDs == rendered)
        #expect(plan.start == .reset)
    }
}

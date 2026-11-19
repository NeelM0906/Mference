import Foundation

/// What running one tool call produced.
public struct ToolOutcome: Sendable, Hashable {
    /// The tool message sent back to the model.
    public var content: String
    /// Steps for the disclosure.
    public var steps: [ChatStep]
    /// Sources the answer may cite, numbered for this answer.
    public var citations: [ChatCitation]

    public init(content: String, steps: [ChatStep] = [], citations: [ChatCitation] = []) {
        self.content = content
        self.steps = steps
        self.citations = citations
    }
}

/// Tools the chat model may call while answering one message (spec 4.5). The
/// web toolbox (`web_search`, `open_page`) is the first; the view model asks
/// for a fresh session per answer so citation numbers start at 1.
public protocol ChatToolbox: Sendable {
    /// A system prompt to put first in the history, if the tools need one.
    func systemPrompt(now: Date) -> String?
    /// The tools offered to the model.
    var definitions: [ToolDefinition] { get }
    /// Runs one call. Never throws: failures become a tool message the model
    /// can read and a `.failed` step.
    func run(_ call: ToolCall) async -> ToolOutcome
}

/// Makes one `ChatToolbox` per answer.
public protocol ChatToolboxProvider: Sendable {
    /// Nil when the tools are unavailable (web off).
    @MainActor func makeToolbox(webEnabled: Bool) -> (any ChatToolbox)?
    /// Called when a chat turns web on or off (start or stop SearXNG).
    @MainActor func webUsageChanged(enabled: Bool)
    /// Web search is opt-in (L7): true until the person has agreed once.
    @MainActor var needsWebConsent: Bool { get }
    /// Records the person's agreement (Settings > Privacy shows it).
    @MainActor func grantWebConsent()
    /// Why web search cannot run right now even with consent ("Offline mode
    /// is on"), shown on the chat's Web toggle; nil when it can.
    @MainActor var webUnavailableReason: String? { get }
}

extension ChatToolboxProvider {
    public var webUnavailableReason: String? { nil }
}

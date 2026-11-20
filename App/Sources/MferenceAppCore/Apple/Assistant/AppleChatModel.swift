import Foundation

// Chat with Apple's on-device model (FoundationModels). macOS picks the
// variant for each Mac (AFM 3 Core, or AFM 3 Core Advanced on Macs it unlocks
// that for); the app reads which, and its context size, and cannot choose.
// The live model is `FoundationChatModel`; tests use `ScriptedAppleChatModel`.

/// What macOS reports about its on-device model, read without loading it.
public struct AppleModelInfo: Sendable, Hashable {
    public enum State: Sendable, Hashable {
        case ready
        /// This Mac cannot run Apple Intelligence: the picker hides the row.
        case notEligible
        /// Apple Intelligence is off in System Settings.
        case turnedOff
        /// macOS is still downloading or preparing the model.
        case gettingReady
    }

    public var state: State
    public var labels: AppleModelLabels

    public init(state: State, labels: AppleModelLabels = .fallback) {
        self.state = state
        self.labels = labels
    }
}

/// The names the picker, composer and Settings show for Apple's model.
public struct AppleModelLabels: Sendable, Hashable {
    /// "AFM 3 Core", from the system.
    public var variantName: String
    /// "Apple 3B": the composer button and the label under answers.
    public var shortName: String
    /// Tokens the model reads at once, prompt and answer together.
    public var contextSize: Int

    public init(variantName: String, shortName: String, contextSize: Int) {
        self.variantName = variantName
        self.shortName = shortName
        self.contextSize = contextSize
    }

    /// Before the system was asked, and where it has no variant name.
    public static let fallback = AppleModelLabels(variantName: "Apple on-device", shortName: "Apple",
                                                  contextSize: 4_096)

    /// "AFM 3 Core · 4K context".
    public var detail: String { "\(variantName) · \(contextLabel) context" }

    /// "4K" for 4,096 tokens.
    public var contextLabel: String {
        let thousands = max(1, Int((Double(contextSize) / 1_024).rounded()))
        return "\(thousands)K"
    }

    /// The labels last read from the system. `AppModel.apple`'s names read
    /// these; the live model updates them each time it is asked.
    public static var current: AppleModelLabels {
        lock.withLock { stored }
    }

    static func update(_ labels: AppleModelLabels) {
        lock.withLock { stored = labels }
    }

    private static let lock = NSLock()
    nonisolated(unsafe) private static var stored = AppleModelLabels.fallback
}

/// Apple's on-device model as a chat model.
public protocol AppleChatModeling: Sendable {
    /// Whether it can run, and its names. Cheap: loads nothing.
    func info() async -> AppleModelInfo
    /// One answer for `turns` (system, then user/assistant, ending with the
    /// user's message). The oldest turns are left out when the chat does
    /// not fit the context; the summary says so (`historyTrimmed`).
    func generate(turns: [ChatTurn], options: GenerationOptions) -> GenerationStream
}

/// The chat's view of the engine: the engine for every engine model, and
/// Apple's on-device model as `AppModel.apple` while Settings > Models >
/// Apple Intelligence model is on and this Mac can run it.
public final class AppleChatRouting: EngineServicing, @unchecked Sendable {
    public static let turnedOffReason = "Apple Intelligence is turned off in System Settings."
    public static let gettingReadyReason = "Apple’s on-device model is still getting ready."
    public static let switchedOffReason = "The Apple Intelligence model is off in Settings."
    /// System Settings > Apple Intelligence & Siri.
    public static let systemSettingsURL = URL(string: "x-apple.systempreferences:com.apple.Siri-Settings.extension")!

    private let engine: any EngineServicing
    private let apple: any AppleChatModeling
    private let isEnabled: @Sendable () async -> Bool
    private let lock = NSLock()
    private var nudges: [UUID: AsyncStream<EngineServiceStatus>.Continuation] = [:]

    /// `isEnabled` reads the Settings switch; it is read on every call so a
    /// change applies at once.
    public init(engine: any EngineServicing, apple: any AppleChatModeling,
                isEnabled: @escaping @Sendable () async -> Bool) {
        self.engine = engine
        self.apple = apple
        self.isEnabled = isEnabled
    }

    /// The picker short-hand for a dimmed Apple row ("Turned off").
    public static func status(forUnavailable reason: String) -> String {
        switch reason {
        case turnedOffReason: "Turned off"
        case gettingReadyReason: "Getting ready"
        default: "Unavailable"
        }
    }

    /// The Settings switch or Apple Intelligence changed: subscribers re-read
    /// `models()`.
    public func appleModelChanged() {
        let status = engine.status
        for continuation in lock.withLock({ Array(nudges.values) }) {
            continuation.yield(status)
        }
    }

    /// The Apple row, or nil when it is hidden (switch off, Mac not eligible).
    func appleOption() async -> ModelOption? {
        guard await isEnabled() else { return nil }
        let info = await apple.info()
        switch info.state {
        case .notEligible:
            return nil
        case .ready:
            return ModelOption(model: .apple, availability: .ready, contextLength: info.labels.contextSize)
        case .turnedOff:
            return ModelOption(model: .apple, availability: .notInstalled(reason: Self.turnedOffReason))
        case .gettingReady:
            return ModelOption(model: .apple, availability: .notInstalled(reason: Self.gettingReadyReason))
        }
    }

    // MARK: EngineServicing

    public func models() async -> [ModelOption] {
        var options = await engine.models()
        if let apple = await appleOption() { options.append(apple) }
        return options
    }

    public func refreshCatalog() async { await engine.refreshCatalog() }

    public var status: EngineServiceStatus { engine.status }

    public func statusUpdates() -> AsyncStream<EngineServiceStatus> {
        let (stream, continuation) = AsyncStream<EngineServiceStatus>.makeStream()
        let id = UUID()
        lock.withLock { nudges[id] = continuation }
        let updates = engine.statusUpdates()
        let forward = Task {
            for await status in updates { continuation.yield(status) }
            continuation.finish()
        }
        continuation.onTermination = { [weak self] _ in
            forward.cancel()
            self?.lock.withLock { _ = self?.nudges.removeValue(forKey: id) }
        }
        return stream
    }

    /// Apple's model needs no load: macOS keeps it ready. The engine's model
    /// stays resident.
    public func load(_ model: AppModel) async throws {
        guard model.isEngineModel else { return }
        try await engine.load(model)
    }

    public func unload() async throws { try await engine.unload() }

    public func generate(model: AppModel, turns: [ChatTurn], options: GenerationOptions) -> GenerationStream {
        guard model == .apple else { return engine.generate(model: model, turns: turns, options: options) }
        let apple = self.apple
        let isEnabled = self.isEnabled
        return GenerationStream { continuation in
            Task {
                guard await isEnabled() else {
                    continuation.finish(throwing: AssistantModelError.noModelAvailable(reason: Self.switchedOffReason))
                    return
                }
                do {
                    for try await update in apple.generate(turns: turns, options: options) {
                        continuation.yield(update)
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    public func cancelActiveGeneration() async { await engine.cancelActiveGeneration() }

    public func shutdown() async {
        for continuation in lock.withLock({ Array(nudges.values) }) { continuation.finish() }
        await engine.shutdown()
    }
}

/// Fits a chat into a context budget: instructions stay, the newest
/// messages stay, the oldest user/assistant turns go first.
enum AppleChatContext {
    struct Plan: Equatable, Sendable {
        var instructions: String?
        /// Earlier turns kept, oldest first (user and assistant only).
        var history: [ChatTurn]
        /// The person's last message.
        var prompt: String
        var trimmed: Bool
    }

    /// Splits `turns` into instructions, history and the last user message.
    /// Tool turns and tool-call-only assistant turns are left out: Apple's
    /// chats run without tools.
    static func split(_ turns: [ChatTurn]) -> Plan? {
        let instructions = turns.filter { $0.role == .system }.compactMap(\.content)
            .filter { !$0.isEmpty }.joined(separator: "\n\n")
        var dialogue = turns.filter { turn in
            switch turn.role {
            case .user: return true
            case .assistant: return !(turn.content ?? "").isEmpty
            case .system, .tool: return false
            }
        }
        guard let last = dialogue.last, last.role == .user, let prompt = last.content else { return nil }
        dialogue.removeLast()
        return Plan(instructions: instructions.isEmpty ? nil : instructions, history: dialogue, prompt: prompt,
                    trimmed: false)
    }

    /// Drops the oldest history until `cost` of the plan fits `budget`.
    /// Returns nil when even the instructions and prompt alone do not fit.
    static func fit(_ plan: Plan, budget: Int,
                    cost: @Sendable (Plan) async throws -> Int) async rethrows -> Plan? {
        var plan = plan
        while try await cost(plan) > budget {
            guard !plan.history.isEmpty else { return nil }
            // Drop a whole exchange where possible, so the history never
            // starts with an answer.
            plan.history.removeFirst()
            if plan.history.first?.role == .assistant { plan.history.removeFirst() }
            plan.trimmed = true
        }
        return plan
    }

    /// Tokens kept for the answer: a quarter of the context, at most what
    /// the request allows.
    static func answerReserve(contextSize: Int, requested: Int) -> Int {
        max(64, min(requested, contextSize / 4))
    }

    /// The text to stream next: what `snapshot` adds after `sent`.
    static func delta(sent: String, snapshot: String) -> String? {
        guard snapshot.hasPrefix(sent), snapshot.count > sent.count else { return nil }
        return String(snapshot.dropFirst(sent.count))
    }
}

/// A scripted Apple model: tests and previews.
public final class ScriptedAppleChatModel: AppleChatModeling, @unchecked Sendable {
    private let lock = NSLock()
    private var currentInfo: AppleModelInfo
    private var answer: Result<GenerationSummary, AssistantModelError>
    private var received: [[ChatTurn]] = []

    public init(info: AppleModelInfo = AppleModelInfo(state: .ready),
                answer: Result<GenerationSummary, AssistantModelError> = .success(
                    GenerationSummary(content: "Hello from Apple.", reasoning: nil, toolCalls: [], finish: .stop,
                                      promptTokens: 0, completionTokens: 0))) {
        self.currentInfo = info
        self.answer = answer
    }

    /// The turns of each request so far.
    public var requests: [[ChatTurn]] { lock.withLock { received } }

    public func setInfo(_ info: AppleModelInfo) { lock.withLock { currentInfo = info } }

    public func info() async -> AppleModelInfo { lock.withLock { currentInfo } }

    public func generate(turns: [ChatTurn], options: GenerationOptions) -> GenerationStream {
        let answer = lock.withLock {
            received.append(turns)
            return self.answer
        }
        return GenerationStream { continuation in
            Task {
                switch answer {
                case .success(let summary):
                    continuation.yield(.prepared(promptTokens: summary.promptTokens))
                    if !summary.content.isEmpty { continuation.yield(.content(summary.content)) }
                    continuation.yield(.completed(summary))
                    continuation.finish()
                case .failure(let error):
                    continuation.finish(throwing: error)
                }
            }
        }
    }
}

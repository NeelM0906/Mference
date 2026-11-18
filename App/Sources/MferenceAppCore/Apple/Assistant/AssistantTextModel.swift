import Foundation

// The small text jobs of decision L12 (titles, short summaries, mail sorting,
// scan field filling) run on a text model behind this protocol: Apple's
// on-device model first, Quick through `EngineServicing` when Apple
// Intelligence is off. Tests use `ScriptedAssistantTextModel`.

/// Which model did a job, for the small label under its result.
public enum AssistantModelKind: String, Sendable, Hashable, Codable {
    /// Apple's on-device model (FoundationModels).
    case appleOnDevice
    /// Quick 4B through the in-process engine.
    case quick

    /// The label shown with the result ("Apple on-device", "Quick 4B").
    public var label: String {
        switch self {
        case .appleOnDevice: "Apple on-device"
        case .quick: AppModel.quick.shortName
        }
    }
}

/// Whether a text model can run now.
public enum AssistantAvailability: Sendable, Hashable {
    case available
    /// `reason` is one user-facing sentence.
    case unavailable(reason: String)

    public var isAvailable: Bool { self == .available }
}

/// One request: fixed instructions plus the prompt, no history.
public struct AssistantRequest: Sendable, Hashable {
    public var instructions: String
    public var prompt: String
    public var maximumTokens: Int
    /// Low by default: these jobs want the same answer for the same input.
    public var temperature: Double

    public init(instructions: String, prompt: String, maximumTokens: Int = 512, temperature: Double = 0) {
        self.instructions = instructions
        self.prompt = prompt
        self.maximumTokens = maximumTokens
        self.temperature = temperature
    }
}

/// Errors from the assistant jobs, each with one user-facing sentence.
public enum AssistantModelError: Error, Sendable, Hashable {
    /// Neither Apple's model nor Quick can run; `reason` says why.
    case noModelAvailable(reason: String)
    /// The model answered, but not in the shape the job needs.
    case unreadableAnswer
    /// The model declined or could not handle the input.
    case declined(String)
    /// The input is too long for the model.
    case inputTooLong

    var isUnavailability: Bool {
        if case .noModelAvailable = self { return true }
        return false
    }

    public var userMessage: String {
        switch self {
        case .noModelAvailable(let reason): reason
        case .unreadableAnswer: "The model's answer could not be read. Try again."
        case .declined(let message): message
        case .inputTooLong: "This is too long for the on-device model."
        }
    }
}

/// A text model that answers one request at a time.
public protocol AssistantTextModel: Sendable {
    var kind: AssistantModelKind { get }
    /// Whether it can run now, without loading anything.
    func availability() async -> AssistantAvailability
    /// The model's whole answer as text.
    func complete(_ request: AssistantRequest) async throws -> String
}

/// A text model that replays scripted answers: tests and previews.
public final class ScriptedAssistantTextModel: AssistantTextModel, @unchecked Sendable {
    public let kind: AssistantModelKind
    private let lock = NSLock()
    private var currentAvailability: AssistantAvailability
    private var answers: [Result<String, AssistantModelError>]
    private var received: [AssistantRequest] = []

    /// Answers are returned in order; the last one repeats.
    public init(kind: AssistantModelKind, availability: AssistantAvailability = .available,
                answers: [Result<String, AssistantModelError>]) {
        self.kind = kind
        self.currentAvailability = availability
        self.answers = answers
    }

    public convenience init(kind: AssistantModelKind, availability: AssistantAvailability = .available,
                            answer: String) {
        self.init(kind: kind, availability: availability, answers: [.success(answer)])
    }

    /// The requests seen so far.
    public var requests: [AssistantRequest] { lock.withLock { received } }

    public func setAvailability(_ availability: AssistantAvailability) {
        lock.withLock { currentAvailability = availability }
    }

    public func availability() async -> AssistantAvailability {
        lock.withLock { currentAvailability }
    }

    public func complete(_ request: AssistantRequest) async throws -> String {
        let next: Result<String, AssistantModelError> = lock.withLock {
            received.append(request)
            guard let first = answers.first else { return .failure(.unreadableAnswer) }
            if answers.count > 1 { answers.removeFirst() }
            return first
        }
        return try next.get()
    }
}

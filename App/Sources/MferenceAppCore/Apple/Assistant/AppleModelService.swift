import Foundation

/// A job's result and the model that produced it.
public struct AssistantResult<Value: Sendable>: Sendable {
    public var value: Value
    public var model: AssistantModelKind

    public init(value: Value, model: AssistantModelKind) {
        self.value = value
        self.model = model
    }
}

extension AssistantResult: Equatable where Value: Equatable {}

/// The small text jobs of decision L12, used by Mail, Scan and chat titles.
public protocol AppleModelServicing: Sendable {
    /// The model the next job will use, or why none can run.
    func activeModel() async -> Result<AssistantModelKind, AssistantModelError>
    /// A title of at most six words.
    func title(for text: String) async throws -> AssistantResult<String>
    /// One or two sentences.
    func summary(of text: String) async throws -> AssistantResult<String>
    /// Message-ID to bucket key. Messages the model left out or put in an
    /// unknown bucket are missing from the map.
    func sortMail(_ messages: [MailSortItem], into buckets: [MailSortBucket]) async throws
        -> AssistantResult<[String: String]>
    /// Receipt fields read from recognized text; the person confirms them.
    func fillReceipt(from recognizedText: String) async throws -> AssistantResult<ReceiptDraft>
    /// A meeting the mail asks for, if any (for the event card in Mail).
    func suggestEvent(subject: String, body: String, received: Date, timeZone: TimeZone) async throws
        -> AssistantResult<MailEventSuggestion?>
}

/// Apple's on-device model first; Quick through the engine when Apple
/// Intelligence is off, not ready, declines, or answers unreadably (L12).
public struct AppleModelService: AppleModelServicing {
    /// Messages per sorting request, so the prompt fits Apple's context.
    public static let mailBatchSize = 8
    /// Characters of recognized text sent for one receipt or summary.
    static let textLimit = 6_000

    private let primary: any AssistantTextModel
    private let fallback: (any AssistantTextModel)?

    public init(primary: any AssistantTextModel, fallback: (any AssistantTextModel)?) {
        self.primary = primary
        self.fallback = fallback
    }

    /// The production service over the app's engine.
    public static func live(engine: any EngineServicing) -> AppleModelService {
        AppleModelService(primary: FoundationTextModel(), fallback: QuickTextModel(engine: engine))
    }

    public func activeModel() async -> Result<AssistantModelKind, AssistantModelError> {
        var reasons: [String] = []
        for model in candidates {
            switch await model.availability() {
            case .available: return .success(model.kind)
            case .unavailable(let reason): reasons.append(reason)
            }
        }
        return .failure(.noModelAvailable(reason: reasons.last ?? "No on-device model is available."))
    }

    public func title(for text: String) async throws -> AssistantResult<String> {
        let request = AssistantRequest(instructions: AssistantPrompts.title,
                                       prompt: AssistantPrompts.clip(text, to: 2_000), maximumTokens: 32)
        return try await run(request) { AssistantAnswerParser.singleLine($0, maxWords: 6) }
    }

    public func summary(of text: String) async throws -> AssistantResult<String> {
        let request = AssistantRequest(instructions: AssistantPrompts.summary,
                                       prompt: AssistantPrompts.clip(text, to: Self.textLimit), maximumTokens: 120)
        return try await run(request, parse: AssistantAnswerParser.paragraph)
    }

    public func sortMail(_ messages: [MailSortItem], into buckets: [MailSortBucket]) async throws
        -> AssistantResult<[String: String]> {
        let keys = Set(buckets.map(\.key))
        var assignments: [String: String] = [:]
        var usedModel: AssistantModelKind?
        var start = 0
        while start < messages.count {
            try Task.checkCancellation()
            let batch = Array(messages[start..<min(start + Self.mailBatchSize, messages.count)])
            start += Self.mailBatchSize
            let request = AssistantRequest(instructions: AssistantPrompts.mailSort,
                                           prompt: AssistantPrompts.mailSortPrompt(messages: batch, buckets: buckets),
                                           maximumTokens: 24 * batch.count + 32)
            let result = try await run(request) { answer -> [String: String]? in
                guard let object = AssistantAnswerParser.jsonObject(in: answer) else { return nil }
                var mapped: [String: String] = [:]
                for (number, value) in object {
                    guard let index = Int(number.trimmingCharacters(in: CharacterSet(charactersIn: "[] "))),
                          batch.indices.contains(index - 1),
                          let key = AssistantAnswerParser.string(value), keys.contains(key) else { continue }
                    mapped[batch[index - 1].messageID] = key
                }
                return mapped.isEmpty ? nil : mapped
            }
            assignments.merge(result.value) { _, new in new }
            usedModel = usedModel ?? result.model
        }
        if let usedModel {
            return AssistantResult(value: assignments, model: usedModel)
        }
        return AssistantResult(value: [:], model: try await activeModel().get())
    }

    public func fillReceipt(from recognizedText: String) async throws -> AssistantResult<ReceiptDraft> {
        let request = AssistantRequest(instructions: AssistantPrompts.receiptFields,
                                       prompt: AssistantPrompts.clip(recognizedText, to: Self.textLimit),
                                       maximumTokens: 600)
        return try await run(request) { answer in
            AssistantAnswerParser.jsonObject(in: answer).map(ReceiptDraft.init(json:))
        }
    }

    public func suggestEvent(subject: String, body: String, received: Date, timeZone: TimeZone) async throws
        -> AssistantResult<MailEventSuggestion?> {
        let prompt = AssistantPrompts.eventPrompt(subject: subject, body: body, received: received, timeZone: timeZone)
        let request = AssistantRequest(instructions: AssistantPrompts.mailEvent, prompt: prompt, maximumTokens: 120)
        return try await run(request) { answer -> MailEventSuggestion?? in
            guard let object = AssistantAnswerParser.jsonObject(in: answer) else { return nil }
            return .some(MailEventSuggestion(json: object, timeZone: timeZone))
        }
    }

    // MARK: Running a job

    private var candidates: [any AssistantTextModel] {
        [primary] + (fallback.map { [$0] } ?? [])
    }

    /// Tries each available model in order until one gives a readable answer.
    private func run<Value: Sendable>(_ request: AssistantRequest,
                                      parse: (String) -> Value?) async throws -> AssistantResult<Value> {
        var lastError: AssistantModelError?
        for model in candidates {
            try Task.checkCancellation()
            if case .unavailable(let reason) = await model.availability() {
                // A later model's reason says more ("Apple Intelligence is off
                // and Quick is not installed"); a real failure says more still.
                if lastError == nil || lastError?.isUnavailability == true {
                    lastError = .noModelAvailable(reason: reason)
                }
                continue
            }
            do {
                let answer = try await model.complete(request)
                if let value = parse(answer) {
                    return AssistantResult(value: value, model: model.kind)
                }
                lastError = .unreadableAnswer
            } catch let error as AssistantModelError {
                lastError = error
            } catch is CancellationError {
                throw CancellationError()
            } catch {
                try Task.checkCancellation()
                lastError = .noModelAvailable(reason: "The on-device model could not answer.")
            }
        }
        throw lastError ?? AssistantModelError.noModelAvailable(reason: "No on-device model is available.")
    }
}

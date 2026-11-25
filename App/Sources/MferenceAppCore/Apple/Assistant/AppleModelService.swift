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
    /// unknown bucket, or whose batch failed, are missing from the map.
    /// `allowFallback` false sorts with Apple's model only, so background
    /// sorting never loads Quick over the person's chat model.
    func sortMail(_ messages: [MailSortItem], into buckets: [MailSortBucket], allowFallback: Bool) async throws
        -> AssistantResult<[String: String]>
    /// Receipt fields read from recognized text; the person confirms them.
    func fillReceipt(from recognizedText: String) async throws -> AssistantResult<ReceiptDraft>
    /// A meeting the mail asks for, if any (for the event card in Mail).
    /// `allowFallback` false asks Apple's model only (automatic, on open).
    func suggestEvent(subject: String, body: String, received: Date, timeZone: TimeZone, allowFallback: Bool)
        async throws -> AssistantResult<MailEventSuggestion?>
    /// What a scan is, when the text alone does not say. `allowFallback`
    /// false (automatic, while reading a scan) asks Apple's model only.
    func classifyScan(_ recognizedText: String, allowFallback: Bool) async throws -> AssistantResult<ScanKind>
    /// A business card's name, company and job title; the person checks them.
    func fillContact(from recognizedText: String, allowFallback: Bool) async throws -> AssistantResult<ContactDraft>
}

/// Apple's on-device model first; Quick through the engine when Apple
/// Intelligence is off, not ready, declines, or answers unreadably (L12).
public struct AppleModelService: AppleModelServicing {
    /// Messages per sorting request, so the prompt fits Apple's context.
    public static let mailBatchSize = 8
    /// Characters of recognized text sent for one receipt or summary.
    static let textLimit = 6_000

    /// Why no job runs when the person switched Apple's model off and no
    /// fallback may run (background mail sorting, automatic event cards).
    static let primaryOffReason = "The Apple Intelligence model is off in Settings."

    private let primary: any AssistantTextModel
    private let fallback: (any AssistantTextModel)?
    /// Settings > Models > Apple Intelligence model, read before every job so
    /// a change applies at once. Off skips `primary`; Quick still runs where
    /// a fallback is allowed.
    private let isPrimaryEnabled: @Sendable () async -> Bool

    public init(primary: any AssistantTextModel, fallback: (any AssistantTextModel)?,
                isPrimaryEnabled: @escaping @Sendable () async -> Bool = { true }) {
        self.primary = primary
        self.fallback = fallback
        self.isPrimaryEnabled = isPrimaryEnabled
    }

    /// The production service over the app's engine. `useAppleModel` is the
    /// person's Settings switch for Apple's on-device model.
    public static func live(engine: any EngineServicing,
                            useAppleModel: @escaping @Sendable () async -> Bool) -> AppleModelService {
        AppleModelService(primary: FoundationTextModel(), fallback: QuickTextModel(engine: engine),
                          isPrimaryEnabled: useAppleModel)
    }

    public func activeModel() async -> Result<AssistantModelKind, AssistantModelError> {
        var reasons: [String] = []
        let models = await candidates(allowFallback: true)
        if models.isEmpty { return .failure(.noModelAvailable(reason: Self.primaryOffReason)) }
        for model in models {
            switch await model.availability() {
            case .available: return .success(model.kind)
            case .unavailable(let reason): reasons.append(reason)
            }
        }
        return .failure(.noModelAvailable(reason: reasons.last ?? "No on-device model is available."))
    }

    public func title(for text: String) async throws -> AssistantResult<String> {
        let request = AssistantRequest(instructions: AssistantPrompts.instructions(AssistantPrompts.title),
                                       prompt: AssistantPrompts.textPrompt("Write the title for this text.", text,
                                                                           limit: 2_000),
                                       maximumTokens: 32)
        return try await run(request) { AssistantAnswerParser.singleLine($0, maxWords: 6) }
    }

    public func summary(of text: String) async throws -> AssistantResult<String> {
        let request = AssistantRequest(instructions: AssistantPrompts.instructions(AssistantPrompts.summary),
                                       prompt: AssistantPrompts.textPrompt("Summarize this text.", text,
                                                                           limit: Self.textLimit),
                                       maximumTokens: 120)
        return try await run(request, parse: AssistantAnswerParser.paragraph)
    }

    public func sortMail(_ messages: [MailSortItem], into buckets: [MailSortBucket], allowFallback: Bool) async throws
        -> AssistantResult<[String: String]> {
        let keys = Set(buckets.map(\.key))
        var assignments: [String: String] = [:]
        var usedModel: AssistantModelKind?
        var lastError: (any Error)?
        var start = 0
        while start < messages.count {
            try Task.checkCancellation()
            let batch = Array(messages[start..<min(start + Self.mailBatchSize, messages.count)])
            start += Self.mailBatchSize
            let request = AssistantRequest(instructions: AssistantPrompts.instructions(AssistantPrompts.mailSort),
                                           prompt: AssistantPrompts.mailSortPrompt(messages: batch, buckets: buckets),
                                           maximumTokens: 24 * batch.count + 32)
            // Each batch stands alone: one failed batch keeps the others.
            let parse = { (answer: String) -> [String: String]? in
                guard let object = AssistantAnswerParser.jsonObject(in: answer) else { return nil }
                var mapped: [String: String] = [:]
                for (number, value) in object {
                    guard let index = Int(number.trimmingCharacters(in: CharacterSet(charactersIn: "[] "))),
                          batch.indices.contains(index - 1),
                          let key = AssistantAnswerParser.string(value), keys.contains(key) else { continue }
                    mapped[batch[index - 1].messageID] = key
                }
                // An answer naming only unknown buckets is still an answer:
                // those messages are left out, as documented.
                return mapped
            }
            do {
                let result = try await run(request, models: await candidates(allowFallback: allowFallback), parse: parse)
                assignments.merge(result.value) { _, new in new }
                usedModel = usedModel ?? result.model
            } catch is CancellationError {
                throw CancellationError()
            } catch {
                lastError = error
            }
        }
        if let usedModel {
            return AssistantResult(value: assignments, model: usedModel)
        }
        if let lastError { throw lastError }
        return AssistantResult(value: [:], model: try await activeModel().get())
    }

    public func fillReceipt(from recognizedText: String) async throws -> AssistantResult<ReceiptDraft> {
        let request = AssistantRequest(instructions: AssistantPrompts.instructions(AssistantPrompts.receiptFields),
                                       prompt: AssistantPrompts.textPrompt("Fill the fields from this receipt.",
                                                                           recognizedText, limit: Self.textLimit),
                                       maximumTokens: 600)
        return try await run(request) { answer in
            AssistantAnswerParser.jsonObject(in: answer).map(ReceiptDraft.init(json:))
        }
    }

    public func suggestEvent(subject: String, body: String, received: Date, timeZone: TimeZone,
                             allowFallback: Bool) async throws -> AssistantResult<MailEventSuggestion?> {
        let prompt = AssistantPrompts.eventPrompt(subject: subject, body: body, received: received, timeZone: timeZone)
        let request = AssistantRequest(instructions: AssistantPrompts.instructions(AssistantPrompts.mailEvent),
                                       prompt: prompt, maximumTokens: 120)
        return try await run(request, models: await candidates(allowFallback: allowFallback)) { answer -> MailEventSuggestion?? in
            guard let object = AssistantAnswerParser.jsonObject(in: answer) else { return nil }
            // Only an explicit "no meeting" (no title) is a valid empty
            // answer; a meeting whose times do not parse is unreadable, so
            // the next model gets a try.
            guard AssistantAnswerParser.string(object["title"]) != nil else { return .some(nil) }
            guard let suggestion = MailEventSuggestion(json: object, timeZone: timeZone) else { return nil }
            return .some(suggestion)
        }
    }

    public func classifyScan(_ recognizedText: String, allowFallback: Bool) async throws -> AssistantResult<ScanKind> {
        let request = AssistantRequest(instructions: AssistantPrompts.instructions(AssistantPrompts.scanKind),
                                       prompt: AssistantPrompts.textPrompt("Say what this scanned document is.",
                                                                           recognizedText, limit: 2_000),
                                       maximumTokens: 24)
        return try await run(request, models: await candidates(allowFallback: allowFallback)) { answer in
            guard let object = AssistantAnswerParser.jsonObject(in: answer),
                  let value = AssistantAnswerParser.string(object["kind"])?.lowercased() else { return nil }
            return ScanKind(rawValue: value)
        }
    }

    public func fillContact(from recognizedText: String, allowFallback: Bool) async throws
        -> AssistantResult<ContactDraft> {
        let request = AssistantRequest(instructions: AssistantPrompts.instructions(AssistantPrompts.contactFields),
                                       prompt: AssistantPrompts.textPrompt("Fill the fields from this business card.",
                                                                           recognizedText, limit: 1_500),
                                       maximumTokens: 80)
        return try await run(request, models: await candidates(allowFallback: allowFallback)) { answer in
            AssistantAnswerParser.jsonObject(in: answer).map(ContactDraft.init(json:))
        }
    }

    // MARK: Running a job

    /// The models a job may try, in order: Apple's unless the person
    /// switched it off, then Quick when a fallback is allowed.
    private func candidates(allowFallback: Bool) async -> [any AssistantTextModel] {
        var models: [any AssistantTextModel] = []
        if await isPrimaryEnabled() { models.append(primary) }
        if allowFallback, let fallback { models.append(fallback) }
        return models
    }

    /// Tries each available model in order until one gives a readable answer.
    private func run<Value: Sendable>(_ request: AssistantRequest,
                                      parse: (String) -> Value?) async throws -> AssistantResult<Value> {
        try await run(request, models: await candidates(allowFallback: true), parse: parse)
    }

    private func run<Value: Sendable>(_ request: AssistantRequest, models: [any AssistantTextModel],
                                      parse: (String) -> Value?) async throws -> AssistantResult<Value> {
        var lastError: AssistantModelError?
        for model in models {
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
                try Task.checkCancellation()
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
        throw lastError ?? AssistantModelError.noModelAvailable(
            reason: models.isEmpty ? Self.primaryOffReason : "No on-device model is available.")
    }
}

extension AppleModelServicing {
    /// Asks any available model (Apple's, else Quick).
    public func suggestEvent(subject: String, body: String, received: Date, timeZone: TimeZone) async throws
        -> AssistantResult<MailEventSuggestion?> {
        try await suggestEvent(subject: subject, body: body, received: received, timeZone: timeZone, allowFallback: true)
    }

    /// Sorts with any available model (Apple's, else Quick).
    public func sortMail(_ messages: [MailSortItem], into buckets: [MailSortBucket]) async throws
        -> AssistantResult<[String: String]> {
        try await sortMail(messages, into: buckets, allowFallback: true)
    }
}

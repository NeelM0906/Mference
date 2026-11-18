import Foundation
import Mference
import Synchronization

// A scripted backend for tests of the engine and of its clients. It loads
// nothing and allocates nothing: a "model" is a script of events. Its journal
// and counters let a test check ordering (release before load), exclusivity
// (one generation at a time, one model resident), and cancellation, and its
// latches let a test hold a load or a generation at a known point.

/// A one-way gate a test opens. `wait()` returns once it is open, and throws
/// `CancellationError` if the waiting task is cancelled first.
public final class FakeEngineLatch: Sendable {
    private struct State {
        var isOpen: Bool
        var nextID: UInt64 = 0
        var waiters: [UInt64: CheckedContinuation<Void, Error>] = [:]
    }

    private let state: Mutex<State>

    public init(isOpen: Bool = false) {
        state = Mutex(State(isOpen: isOpen))
    }

    public var isOpen: Bool { state.withLock { $0.isOpen } }

    public func open() {
        let waiting = state.withLock { state -> [CheckedContinuation<Void, Error>] in
            state.isOpen = true
            let waiting = Array(state.waiters.values)
            state.waiters.removeAll()
            return waiting
        }
        for continuation in waiting { continuation.resume() }
    }

    public func wait() async throws {
        let id = state.withLock { state -> UInt64 in
            defer { state.nextID &+= 1 }
            return state.nextID
        }
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
                let immediate = state.withLock { state -> Result<Void, Error>? in
                    if state.isOpen { return .success(()) }
                    if Task.isCancelled { return .failure(CancellationError()) }
                    state.waiters[id] = continuation
                    return nil
                }
                if let immediate { continuation.resume(with: immediate) }
            }
        } onCancel: {
            let continuation = state.withLock { $0.waiters.removeValue(forKey: id) }
            continuation?.resume(throwing: CancellationError())
        }
    }
}

public enum FakeEngineJournalEntry: Sendable, Hashable {
    case loadStarted(String)
    case loadFinished(String)
    case loadFailed(String)
    case unloaded(String)
    /// The model object itself was freed.
    case deinitialized(String)
    case generationStarted(String)
    case generationFinished(String)
    case generationCancelled(String)
}

/// Shared record of what fake backends and their models did, in order.
public final class FakeEngineJournal: Sendable {
    private struct State {
        var entries: [FakeEngineJournalEntry] = []
        var liveModels = 0
        var maxLiveModels = 0
        var generations = 0
        var maxGenerations = 0
    }

    private let state = Mutex(State())

    public init() {}

    public var entries: [FakeEngineJournalEntry] { state.withLock { $0.entries } }
    /// Models between load start and unload (or failed load) right now.
    public var liveModels: Int { state.withLock { $0.liveModels } }
    /// The most models ever live at once. 1 means no swap held two.
    public var maxLiveModels: Int { state.withLock { $0.maxLiveModels } }
    /// The most generations ever running at once.
    public var maxConcurrentGenerations: Int { state.withLock { $0.maxGenerations } }

    func record(_ entry: FakeEngineJournalEntry) {
        state.withLock { state in
            state.entries.append(entry)
            switch entry {
            case .loadStarted:
                state.liveModels += 1
                state.maxLiveModels = max(state.maxLiveModels, state.liveModels)
            case .loadFailed, .unloaded:
                state.liveModels -= 1
            case .generationStarted:
                state.generations += 1
                state.maxGenerations = max(state.maxGenerations, state.generations)
            case .generationFinished, .generationCancelled:
                state.generations -= 1
            case .loadFinished, .deinitialized:
                break
            }
        }
    }
}

/// How one fake model behaves.
public struct FakeEngineScript: Sendable {
    /// Nil: ChatML, default sampling, and the descriptor's resolved context.
    public var traits: EngineModelTraits?
    /// Streamed in order by every generation; a completion is their sum.
    public var events: [EngineEvent]
    /// Reported when every event fits in the request's token budget.
    public var finishReason: EngineFinishReason
    /// Thrown by `load` after `loadGate` opens.
    public var loadError: (any Error)?
    /// Opened by the fake when its load begins.
    public var loadStarted: FakeEngineLatch?
    /// `load` waits for this before finishing.
    public var loadGate: FakeEngineLatch?
    /// A generation pauses after streaming this many events...
    public var holdAfterEvents: Int?
    /// ...opens this to say it has paused...
    public var holdReached: FakeEngineLatch?
    /// ...and resumes once this opens.
    public var holdGate: FakeEngineLatch?

    public init(traits: EngineModelTraits? = nil,
                events: [EngineEvent] = [.content("Hello"), .content(" world")],
                finishReason: EngineFinishReason = .stop,
                loadError: (any Error)? = nil,
                loadStarted: FakeEngineLatch? = nil,
                loadGate: FakeEngineLatch? = nil,
                holdAfterEvents: Int? = nil,
                holdReached: FakeEngineLatch? = nil,
                holdGate: FakeEngineLatch? = nil) {
        self.traits = traits
        self.events = events
        self.finishReason = finishReason
        self.loadError = loadError
        self.loadStarted = loadStarted
        self.loadGate = loadGate
        self.holdAfterEvents = holdAfterEvents
        self.holdReached = holdReached
        self.holdGate = holdGate
    }
}

/// A backend whose models are scripts. Register one per `EngineKind` the
/// catalog uses; several may share one journal.
public final class FakeEngineBackend: EngineBackend {
    public let engine: EngineKind
    public let journal: FakeEngineJournal
    private let scripts: Mutex<(byID: [String: FakeEngineScript], fallback: FakeEngineScript)>

    public init(engine: EngineKind = .mference,
                journal: FakeEngineJournal = FakeEngineJournal(),
                scripts: [String: FakeEngineScript] = [:],
                defaultScript: FakeEngineScript = FakeEngineScript()) {
        self.engine = engine
        self.journal = journal
        self.scripts = Mutex((byID: scripts, fallback: defaultScript))
    }

    /// Replaces the script for `modelID`; takes effect at its next load.
    public func setScript(_ script: FakeEngineScript, for modelID: String) {
        scripts.withLock { $0.byID[modelID] = script }
    }

    public func load(_ descriptor: EngineModelDescriptor,
                     options: EngineLoadOptions) async throws -> any EngineModel {
        let script = scripts.withLock { $0.byID[descriptor.id] ?? $0.fallback }
        journal.record(.loadStarted(descriptor.id))
        script.loadStarted?.open()
        do {
            try await script.loadGate?.wait()
            if let loadError = script.loadError { throw loadError }
        } catch {
            journal.record(.loadFailed(descriptor.id))
            throw error
        }
        var traits = script.traits ?? EngineModelTraits(
            chatDialect: .chatml,
            maxContext: descriptor.resolvedContext(options: options))
        if script.traits != nil {
            traits.maxContext = min(traits.maxContext, descriptor.resolvedContext(options: options))
        }
        let model = FakeEngineModel(descriptor: descriptor, traits: traits,
                                    script: script, journal: journal)
        journal.record(.loadFinished(descriptor.id))
        return model
    }

    /// A descriptor for a model that exists only in tests.
    public static func descriptor(id: String,
                                  engine: EngineKind = .mference,
                                  contextLength: Int = 4_096,
                                  product: EngineProductSlot? = nil) -> EngineModelDescriptor {
        let format: EngineModelFormat = switch engine {
        case .mference: .gturbo(family: .qwen36, checkpointID: nil)
        case .llamaCpp: .gguf(EngineGGUFMetadata(architecture: "fake", contextLength: contextLength,
                                                 tensorCount: 0, nextNTensorCount: 0))
        }
        return EngineModelDescriptor(
            id: id,
            displayName: id,
            engine: engine,
            format: format,
            location: URL(fileURLWithPath: "/nonexistent/fake-engine/\(id)"),
            sizeBytes: 0,
            contextLength: contextLength,
            product: product)
    }
}

/// One loaded fake. Prompt tokens are one per whitespace-separated word plus
/// one per message; each event costs one completion token.
public final class FakeEngineModel: EngineModel {
    public let descriptor: EngineModelDescriptor
    public let traits: EngineModelTraits
    private let script: FakeEngineScript
    private let journal: FakeEngineJournal
    private let unloaded = Mutex(false)

    init(descriptor: EngineModelDescriptor,
         traits: EngineModelTraits,
         script: FakeEngineScript,
         journal: FakeEngineJournal) {
        self.descriptor = descriptor
        self.traits = traits
        self.script = script
        self.journal = journal
    }

    deinit {
        journal.record(.deinitialized(descriptor.id))
    }

    public var isUnloaded: Bool { unloaded.withLock { $0 } }

    public func prepare(_ request: EngineRequest) async throws -> EnginePreparedPrompt {
        guard !isUnloaded else { throw EngineError.modelUnloaded(descriptor.id) }
        var count = 0
        for message in request.messages {
            count += 1 + (message.content ?? "").split(whereSeparator: \.isWhitespace).count
        }
        guard count < traits.maxContext else {
            throw EngineError.invalidRequest(message: "prompt exceeds the configured context",
                                             param: "messages", code: "context_length_exceeded")
        }
        return EnginePreparedPrompt(request: request,
                                    promptTokens: (0..<count).map { Int32($0) },
                                    needsToolTemplate: request.needsToolTemplate)
    }

    public func generate(_ prepared: EnginePreparedPrompt,
                         onEvent: @escaping @Sendable (EngineEvent) -> Void) async throws -> EngineCompletion {
        guard !isUnloaded else { throw EngineError.modelUnloaded(descriptor.id) }
        journal.record(.generationStarted(descriptor.id))
        let budget = max(0, min(prepared.request.maximumCompletionTokens,
                                traits.maxContext - prepared.promptTokens.count))
        var content = ""
        var reasoning = ""
        var calls: [ParsedToolCall] = []
        var emitted = 0
        do {
            for event in script.events.prefix(budget) {
                if emitted == script.holdAfterEvents {
                    script.holdReached?.open()
                    try await script.holdGate?.wait()
                }
                try Task.checkCancellation()
                switch event {
                case .content(let text): content += text
                case .reasoning(let text): reasoning += text
                case .toolCall(let call): calls.append(call)
                }
                onEvent(event)
                emitted += 1
                await Task.yield()
            }
            try Task.checkCancellation()
        } catch {
            journal.record(.generationCancelled(descriptor.id))
            throw error
        }
        journal.record(.generationFinished(descriptor.id))
        let finish: EngineFinishReason = emitted < script.events.count ? .length
            : (calls.isEmpty ? script.finishReason : .toolCalls)
        return EngineCompletion(
            content: content,
            reasoningContent: reasoning.isEmpty ? nil : reasoning,
            toolCalls: calls,
            finishReason: finish,
            usage: EngineUsage(promptTokens: prepared.promptTokens.count, completionTokens: emitted))
    }

    public func unload() async {
        let first = unloaded.withLock { flag -> Bool in
            defer { flag = true }
            return !flag
        }
        if first { journal.record(.unloaded(descriptor.id)) }
    }
}

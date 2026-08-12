import Darwin
import Foundation
import Synchronization
import Mference
import MferenceDecodeProtocol

public final class DecodeServiceInferenceClient: AppModelLifecycleClient,
    AppGenerationContextReporting, AppInferenceMemoryReporting,
    AppInferenceTranscriptReporting, @unchecked Sendable {
    private struct Connection {
        var input: FileHandle?
        var output: FileHandle?
        var loadedDirectory: URL?
        var process: Process?
    }

    private let connection = Mutex(Connection())
    private let commandWrites = Mutex(())
    private let processCreation = Mutex(())
    private let serviceURL: URL
    private let inferenceMemory = Mutex<UInt64?>(nil)
    public let generationTranscriptMailbox = GenerationTranscriptMailbox()

    public var currentInferenceMemoryBytes: UInt64? {
        inferenceMemory.withLock { $0 }
    }

    public init(serviceURL: URL? = nil) {
        self.serviceURL = serviceURL ?? Self.defaultServiceURL()
    }

    public func ensureLoaded(modelDirectory: URL, maxContextTokens: Int,
                             options: AppRuntimeOptions, forceLogitsHead: Bool,
                             onState: @escaping @Sendable (AppModelLoadState) -> Void) async throws {
        onState(.loading(.validatingDirectory))
        let handles = try await Task.detached(priority: .userInitiated) { [self] in
            try ensureProcess()
        }.value
        async let localTokenizer = MFTokenizer.load(
            forModelDirectory: modelDirectory)
        let request = DecodeLoadRequest(
            modelPath: modelDirectory.path, maxContextTokens: maxContextTokens,
            runtimeOptions: Self.decodeRuntimeOptions(options),
            forceLogitsHead: forceLogitsHead)
        try write(DecodeServiceCommand.load(request), to: handles.input)
        let event = try await readEvent(from: handles.output)
        guard event.generationID == request.requestID, event.kind == .ready else {
            throw AppInferenceError.modelLoadFailed(
                event.error ?? "decode service load failed")
        }
        do {
            _ = try await localTokenizer
        } catch {
            shutdown()
            throw AppInferenceError.tokenizerUnavailable("\(error)")
        }
        inferenceMemory.withLock { $0 = event.currentMemoryBytes }
        connection.withLock { $0.loadedDirectory = modelDirectory.standardizedFileURL }
        onState(.ready(modelDirectory: modelDirectory, loadSeconds: 0))
    }

    public func unload() async {
        guard let handles = currentHandles() else { return }
        let requestID = UUID()
        try? write(DecodeServiceCommand.unload(requestID), to: handles.input)
        _ = try? await readEvent(from: handles.output)
        connection.withLock { $0.loadedDirectory = nil }
        inferenceMemory.withLock { $0 = nil }
    }

    public func generate(_ request: AppGenerationRequest)
        -> AsyncThrowingStream<AppInferenceEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task.detached(priority: .userInitiated) { [self] in
                do {
                    try request.validate()
                    guard let handles = handles(forLoadedDirectory: request.modelDirectory) else {
                        throw AppInferenceError.modelNotLoaded
                    }
                    let generationID = UUID()
                    generationTranscriptMailbox.reset()
                    let command = DecodeGenerationRequest(
                        messages: request.messages.map(Self.decodeGenerationMessage),
                        maxNewTokens: request.maxNewTokens,
                        maxContextTokens: request.maxContextTokens,
                        temperature: request.temperature,
                        topK: request.topK,
                        topP: request.topP,
                        repetitionPenalty: request.repetitionPenalty,
                        runtimeOptions: Self.decodeRuntimeOptions(request.runtimeOptions),
                        generationID: generationID)
                    try write(DecodeServiceCommand.generate(command), to: handles.input)

                    var expectedSequence: UInt64 = 1
                    var lastMetricYield = Date.distantPast
                    var hasYieldedVisibleText = false
                    while true {
                        let event = try DecodeFrameCodec.read(
                            DecodeServiceEvent.self, from: handles.output)
                        inferenceMemory.withLock { $0 = event.currentMemoryBytes }
                        guard event.generationID == generationID else { continue }

                        if event.kind == .prefill || event.kind == .snapshot {
                            guard event.sequence == expectedSequence else {
                                throw AppInferenceError.unknown(
                                    "decode service event sequence changed from \(expectedSequence) to \(event.sequence)")
                            }
                            expectedSequence &+= 1
                        }
                        if event.kind == .prefill,
                           let done = event.prefillDone,
                           let total = event.prefillTotal {
                            continuation.yield(.prefillProgress(done: done, total: total))
                            continue
                        }
                        if event.kind == .snapshot {
                            generationTranscriptMailbox.append(event.textDelta)
                            let now = Date()
                            let beginsVisibleText = !hasYieldedVisibleText
                                && event.textDelta.contains { !$0.isWhitespace }
                            if beginsVisibleText
                                || now.timeIntervalSince(lastMetricYield) >= 0.5 {
                                lastMetricYield = now
                                hasYieldedVisibleText = hasYieldedVisibleText || beginsVisibleText
                                continuation.yield(.token(AppTokenEvent(
                                    index: max(0, event.tokenCount - 1),
                                    textDelta: beginsVisibleText ? event.textDelta : "",
                                    elapsedDecodeSeconds: event.decodeSeconds)))
                            }
                            continue
                        }

                        let diagnostics = Self.diagnostics(
                            event, options: request.runtimeOptions)
                        switch event.kind {
                        case .finished:
                            continuation.yield(.finished(diagnostics))
                            continuation.finish()
                        case .cancelled:
                            continuation.yield(.cancelled(diagnostics))
                            continuation.finish()
                        case .failed:
                            let error = AppInferenceError.unknown(
                                event.error ?? "decode service failed")
                            continuation.yield(.failed(error, partial: diagnostics))
                            continuation.finish(throwing: error)
                        default:
                            continue
                        }
                        return
                    }
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { [weak self] _ in
                task.cancel()
                self?.cancel()
            }
        }
    }

    public func prepare(_ request: AppGenerationRequest) async throws
        -> AppGenerationRequest {
        try await AppGenerationContextWindow.prepareUsingModelTokenizer(request)
    }

    public func prepareWithContextReport(_ request: AppGenerationRequest) async throws
        -> AppPreparedGenerationRequest {
        try await AppGenerationContextWindow
            .prepareUsingModelTokenizerWithReport(request)
    }

    public func cancel() {
        guard let input = currentHandles()?.input else { return }
        try? write(DecodeServiceCommand.cancel, to: input)
    }

    public func shutdown() {
        processCreation.withLock { _ in
            let state = connection.withLock { value -> Connection in
                defer { value = Connection() }
                return value
            }
            if let input = state.input {
                try? input.close()
            }
            if let output = state.output { try? output.close() }
            if let process = state.process, process.isRunning {
                if !Self.waitForExit(process, milliseconds: 250), process.isRunning {
                    process.terminate()
                    if !Self.waitForExit(process, milliseconds: 250), process.isRunning {
                        _ = Darwin.kill(process.processIdentifier, SIGKILL)
                        _ = Self.waitForExit(process, milliseconds: 250)
                    }
                }
            }
        }
        inferenceMemory.withLock { $0 = nil }
    }

    deinit {
        shutdown()
    }

    private func ensureProcess() throws -> (input: FileHandle, output: FileHandle) {
        try processCreation.withLock { _ in
            if let handles = currentHandles() { return handles }
            return try launchIndependentService()
        }
    }

    private func launchIndependentService() throws
        -> (input: FileHandle, output: FileHandle) {
        guard FileManager.default.isExecutableFile(atPath: serviceURL.path) else {
            throw AppInferenceError.modelLoadFailed(
                "decode service executable is missing at \(serviceURL.path); run swift build -c release before launching the app")
        }
        let input = Pipe()
        let output = Pipe()
        let process = Process()
        process.executableURL = serviceURL
        process.standardInput = input.fileHandleForReading
        process.standardOutput = output.fileHandleForWriting
        process.standardError = FileHandle.nullDevice
        try process.run()
        let handles = (input.fileHandleForWriting, output.fileHandleForReading)
        connection.withLock {
            $0.input = handles.0
            $0.output = handles.1
            $0.process = process
        }
        return handles
    }

    private func currentHandles() -> (input: FileHandle, output: FileHandle)? {
        let result: (
            handles: (input: FileHandle, output: FileHandle)?,
            stale: Connection?
        ) = connection.withLock { state in
            guard state.process?.isRunning == true,
                  let input = state.input, let output = state.output else {
                let stale = state
                state = Connection()
                return (handles: nil, stale: stale)
            }
            return (handles: (input, output), stale: nil)
        }
        clearStaleConnection(result.stale)
        return result.handles
    }

    private func handles(forLoadedDirectory directory: URL)
        -> (input: FileHandle, output: FileHandle)? {
        let result: (
            handles: (input: FileHandle, output: FileHandle)?,
            stale: Connection?
        ) = connection.withLock { state in
            guard state.process?.isRunning == true else {
                let stale = state
                state = Connection()
                return (handles: nil, stale: stale)
            }
            guard Self.matchesLoadedModelDirectory(
                directory, loadedDirectory: state.loadedDirectory),
                let input = state.input, let output = state.output else {
                return (handles: nil, stale: nil)
            }
            return (handles: (input, output), stale: nil)
        }
        clearStaleConnection(result.stale)
        return result.handles
    }

    private func write(_ command: DecodeServiceCommand, to input: FileHandle) throws {
        try commandWrites.withLock { _ in
            try input.write(contentsOf: DecodeFrameCodec.encode(command))
        }
    }

    private func clearStaleConnection(_ stale: Connection?) {
        guard let stale else { return }
        try? stale.input?.close()
        try? stale.output?.close()
        if connection.withLock({ $0.process == nil }) {
            inferenceMemory.withLock { $0 = nil }
        }
    }

    private static func waitForExit(_ process: Process, milliseconds: UInt64) -> Bool {
        let deadline = Date().addingTimeInterval(Double(milliseconds) / 1_000)
        while process.isRunning, Date() < deadline {
            Thread.sleep(forTimeInterval: 0.01)
        }
        return !process.isRunning
    }

    static func matchesLoadedModelDirectory(
        _ directory: URL, loadedDirectory: URL?
    ) -> Bool {
        directory.standardizedFileURL == loadedDirectory
    }

    private func readEvent(from output: FileHandle) async throws
        -> DecodeServiceEvent {
        try await Task.detached(priority: .userInitiated) {
            try DecodeFrameCodec.read(DecodeServiceEvent.self, from: output)
        }.value
    }

    private static func diagnostics(_ event: DecodeServiceEvent,
                                    options: AppRuntimeOptions) -> AppDiagnostics {
        let stop = AppStopReason(rawValue: event.stopReason ?? "")
            ?? (event.kind == .cancelled
                ? .cancelled
                : event.kind == .failed ? .failed : .maxTokens)
        return AppDiagnostics(
            generatedTokens: event.tokenCount,
            stopReason: stop,
            promptTokenCount: event.promptTokenCount,
            prefillSeconds: event.prefillSeconds,
            timeToFirstTokenSeconds: event.timeToFirstTokenSeconds,
            decodeSeconds: event.decodeSeconds,
            tokensPerSecond: event.tokensPerSecond,
            peakMemoryBytes: event.peakMemoryBytes,
            runtimeOptions: options,
            prefill: prefillDiagnostics(event.prefill, options: options),
            runner: event.runner.map(runnerDiagnostics))
    }

    private static func prefillDiagnostics(
        _ value: DecodePrefillDiagnostics?, options: AppRuntimeOptions
    ) -> PrefillExecutionDiagnostics? {
        guard let value,
              let executedMode = PrefillExecutedMode(rawValue: value.executedMode),
              let completeness = PrefillChunkCompleteness(
                rawValue: value.chunkCompleteness) else { return nil }
        let kvStorage = value.kvStorageMode.flatMap(PrefillKVStorageMode.init(rawValue:))
        return PrefillExecutionDiagnostics(
            config: options.prefillConfig,
            executedMode: executedMode,
            kvStorageMode: kvStorage,
            chunkCompleteness: completeness,
            unsupportedReason: value.unsupportedReason)
    }

    private static func runnerDiagnostics(_ value: DecodeRunnerDiagnostics)
        -> AppRunnerDiagnostics {
        AppRunnerDiagnostics(
            cb1MillisecondsPerToken: value.cb1MillisecondsPerToken,
            ioMillisecondsPerToken: value.ioMillisecondsPerToken,
            cb2MillisecondsPerToken: value.cb2MillisecondsPerToken,
            headMillisecondsPerToken: value.headMillisecondsPerToken,
            rdadviseMillisecondsPerToken: value.rdadviseMillisecondsPerToken,
            rdadviseCallsPerToken: value.rdadviseCallsPerToken,
            rdadviseMegabytesPerToken: value.rdadviseMegabytesPerToken,
            rdadviseSkippedPerToken: value.rdadviseSkippedPerToken,
            rdadviseFailures: value.rdadviseFailures)
    }

    private static func decodeRuntimeOptions(_ options: AppRuntimeOptions)
        -> DecodeRuntimeOptions {
        DecodeRuntimeOptions(
            expertCacheSlots: options.expertCacheSlots,
            expertCachePolicy: options.expertCachePolicy.rawValue,
            prefillEnabled: options.prefillEnabled,
            prefillChunkTokens: options.prefillChunkTokens,
            rdadvisePolicy: options.rdadvisePolicy.rawValue,
            modelVerification: options.modelVerification.rawValue)
    }

    private static func decodeGenerationMessage(
        _ message: AppGenerationMessage
    ) -> DecodeGenerationMessage {
        let role: DecodeGenerationMessage.Role = switch message.role {
        case .system: .system
        case .user: .user
        case .assistant: .assistant
        }
        return DecodeGenerationMessage(role: role, content: message.content)
    }

    private static func defaultServiceURL() -> URL {
        return Bundle.main.executableURL!
            .deletingLastPathComponent()
            .appendingPathComponent("MferenceDecodeService")
    }
}

import Foundation
import Metal
import Mference

private struct MessageJSON: Decodable {
    let role: String
    let content: String
}

public struct RunResult: Equatable, Sendable {
    public let exitCode: Int32
    public init(exitCode: Int32) { self.exitCode = exitCode }
}

public func run(args: Args,
                stdout: FileHandle = .standardOutput,
                stderr: FileHandle = .standardError) async -> RunResult {
    if args.chat {
        return await runChat(args: args, stdout: stdout, stderr: stderr)
    }
    do {
        let modelURL = URL(fileURLWithPath: args.model)
        let expertStreaming = try resolveExpertStreaming(args.expertCacheSlots,
                                                         modelURL: modelURL)
        let tokenizer = try await MFTokenizer.load(forModelDirectory: modelURL)
        let promptIds: [Int32]
        if let rawPrompt = args.prompt {
            promptIds = tokenizer.encode(rawPrompt, addBOS: true)
        } else if let messagesFile = args.messagesFile {
            let data = try Data(contentsOf: URL(fileURLWithPath: messagesFile),
                                options: [.mappedIfSafe])
            let rows = try JSONDecoder().decode([MessageJSON].self, from: data)
            let messages = try rows.map { row -> MFTokenizer.Message in
                guard let role = MFTokenizer.Role(rawValue: row.role) else {
                    throw MFTokenizerError.invalidChatTemplate("unsupported role \(row.role)")
                }
                return MFTokenizer.Message(role: role, content: row.content)
            }
            let rendered = try tokenizer.applyChatTemplate(messages)
            promptIds = tokenizer.encode(rendered, addBOS: false)
        } else {
            return errored(stderr, "one of --prompt or --messages-file is required", 2)
        }
        guard !promptIds.isEmpty else { return errored(stderr, "empty prompt", 2) }
        guard promptIds.count < args.maxContext else {
            return errored(
                stderr,
                "context overflow: prompt \(promptIds.count) reaches maxContext \(args.maxContext)",
                2)
        }
        let effectiveMaxNew = min(args.maxNew, args.maxContext - promptIds.count)
        let config = GenerationConfig(
            maxNewTokens: effectiveMaxNew,
            temperature: args.temperature,
            topK: args.topK,
            topP: args.topP,
            repetitionPenalty: args.repetitionPenalty,
            seed: args.seed,
            stopStrings: args.stops,
            extraStopTokens: [])
        let prefillChunkTokens: Int
        switch args.prefillChunk {
        case .fixed(let n):
            prefillChunkTokens = n
        case .auto:
            // Smallest allowed chunk that covers the prompt: one chunk per
            // prompt when it fits, which reads each layer's routed experts
            // exactly once during prefill.
            prefillChunkTokens = RuntimeConfiguration.allowedPrefillChunkTokens
                .first(where: { $0 >= promptIds.count })
                ?? RuntimeConfiguration.allowedPrefillChunkTokens.last!
        }
        let runtime = RuntimeConfiguration(
            expertCacheSlots: expertStreaming.configSlots,
            rdadvisePolicy: RDAdvicePolicyMode.parse(args.rdadvise),
            prefillChunkTokens: prefillChunkTokens,
            forceLogitsHead: !config.isPureGreedy,
            useMapleFlashHead: args.flashHead,
            kvPagedPolicy: kvPagedPolicy(for: args),
            kvTopKPages: args.kvTopKPages,
            kvPoolPagesPerLayer: args.kvPoolPages)

        guard MTLCreateSystemDefaultDevice() != nil else {
            return errored(stderr, "no Metal device", 1)
        }
        let context = try MetalContext()
        let model = try Model.load(
            directoryURL: modelURL,
            device: context.device,
            streamingMode: expertStreaming.mode,
            expertCachePolicy: runtime.modelExpertCachePolicy,
            integrityPolicy: args.verification)
        let forwardRuntime = try ForwardRunnerFactory.make(
            model: model,
            context: context,
            maxContext: args.maxContext,
            runtimeConfiguration: runtime)
        let runner = forwardRuntime.producer
        let scratch = try RawCompletionScratch(context: context,
                                               vocab: model.config.vocabSize,
                                               logitSoftcap: Float(model.config.finalLogitSoftcap))
        let decoder = args.messagesFile != nil && tokenizer.generationPromptStartsInThinking
            ? StructuredAssistantDecoder(tokenizer: tokenizer,
                                         allowedTools: [],
                                         startsInThought: true)
            : nil
        var completionConfig = config
        var stopMatcher = StreamingStopMatcher(stops: decoder == nil ? [] : config.stopStrings)
        if decoder != nil { completionConfig.stopStrings = [] }
        var decodingError: Error?
        var shouldStop = false
        let stats = try await runRawCompletion(
            producer: runner,
            tokenizer: tokenizer,
            promptIds: promptIds,
            config: completionConfig,
            context: context,
            scratch: scratch,
            prefillConfig: forwardRuntime.prefillConfig,
            shouldStop: { shouldStop }) { progress in
                guard decodingError == nil else { return }
                do {
                switch progress {
                case .prefill:
                    break
                case .token(_, let tokenID, let delta):
                    let events = try structuredEvents(decoder, tokenID: tokenID, text: delta)
                    let visible = try visibleAssistantText(events)
                    let emitted = decoder == nil ? visible : stopMatcher.push(visible)
                    if !emitted.isEmpty { stdout.write(Data(emitted.utf8)) }
                    if decoder != nil, stopMatcher.isStopped { shouldStop = true }
                case .tail(let tail):
                    let events = try structuredTailEvents(decoder, text: tail)
                    let visible = try visibleAssistantText(events)
                    let emitted = decoder == nil ? visible : stopMatcher.push(visible)
                    if !emitted.isEmpty { stdout.write(Data(emitted.utf8)) }
                }
                } catch {
                    decodingError = error
                    shouldStop = true
                }
            }
        if let decodingError { throw decodingError }
        if let decoder {
            let visible = try visibleAssistantText(decoder.finish())
            let emitted = stopMatcher.push(visible) + stopMatcher.finish()
            if !emitted.isEmpty { stdout.write(Data(emitted.utf8)) }
        }

        if ProcessInfo.processInfo.environment["MFERENCE_PREFILL_BREAKDOWN"] == "1",
           runner is RealForwardRunner {
            RealForwardRunner.dumpPrefillBreakdown()
        }
        if ProcessInfo.processInfo.environment["MFERENCE_PHASES"] == "1",
           let runner = runner as? RealForwardRunner {
            let ms = { (n: UInt64) in String(format: "%.1f", Double(n) / 1e6) }
            let total = stats.decodeSeconds * 1000
            let accounted = Double(runner.totalCb1Nanos + runner.totalIoNanos
                                   + runner.totalCb2Nanos) / 1e6
            var lines = "\n[phases over \(stats.newTokens) tokens, decode "
            lines += String(format: "%.0f", total) + " ms]\n"
            lines += "  cb1 encode+commit: " + ms(runner.totalCb1Nanos) + " ms\n"
            lines += "  expert io await:   " + ms(runner.totalIoNanos) + " ms\n"
            lines += "    io overlapped w/ GPU: "
            lines += ms(runner.totalIoOverlappedNanos) + " ms\n"
            lines += "    io exposed (GPU idle): "
            lines += ms(runner.totalIoExposedNanos) + " ms\n"
            lines += "  cb2 encode+commit: " + ms(runner.totalCb2Nanos) + " ms\n"
            if runner.totalRoutedLayerSteps > 0 {
                let rate = 100.0 * Double(runner.totalAllHitLayerSteps)
                    / Double(runner.totalRoutedLayerSteps)
                lines += "  all-hit layer steps: \(runner.totalAllHitLayerSteps)"
                lines += "/\(runner.totalRoutedLayerSteps) ("
                lines += String(format: "%.1f", rate) + "%)\n"
            }
            let gpuBusy = runner.totalGpuBusyNanos
            let gpuSpan = runner.totalGpuSpanNanos
            lines += "  gpu busy: " + ms(gpuBusy) + " ms, span: "
            lines += ms(gpuSpan) + " ms, gap: "
            lines += ms(gpuSpan > gpuBusy ? gpuSpan - gpuBusy : 0) + " ms\n"
            let predicted = runner.totalSpecPrefetchPredicted
            let recall = predicted > 0
                ? Double(runner.totalSpecPrefetchConfirmed) / Double(predicted)
                : 0
            lines += "  spec prefetch: predicted \(predicted)"
            lines += ", issued \(runner.totalSpecPrefetchIssued)"
            lines += ", confirmed \(runner.totalSpecPrefetchConfirmed)"
            lines += String(format: " (recall %.1f%%, %.1f MB)\n",
                            recall * 100,
                            Double(runner.totalSpecPrefetchBytes) / 1_048_576)
            lines += "  unaccounted (GPU waits): "
            lines += String(format: "%.1f", total - accounted) + " ms\n"
            stderr.write(Data(lines.utf8))
        }
        if ProcessInfo.processInfo.environment["MFERENCE_PHASES"] == "1",
           let q38 = runner as? Qwen38ForwardRunner,
           let spec = q38.mtpSpecStats, spec.rounds > 0 {
            let ms = { (n: UInt64) in String(format: "%.1f", Double(n) / 1e6) }
            let acceptRate = spec.draftedTokens > 0
                ? Double(spec.acceptedTokens) / Double(spec.draftedTokens) : 0
            var lines = "\n[mtp spec over \(spec.rounds) rounds]\n"
            lines += "  drafted \(spec.draftedTokens), accepted \(spec.acceptedTokens)"
            lines += String(format: " (accept rate %.1f%%)", acceptRate * 100)
            lines += ", emitted \(spec.emittedTokens), rollbacks \(spec.rollbacks)\n"
            lines += "  draft: " + ms(spec.draftNanos) + " ms, verify: "
            lines += ms(spec.verifyNanos) + " ms, accept: "
            lines += ms(spec.acceptNanos) + " ms\n"
            var profile: [String] = []
            for (accepts, trials) in zip(spec.positionAccepts, spec.positionTrials)
                where trials > 0 {
                let rate: Double = 100.0 * Double(accepts) / Double(trials)
                profile.append(String(format: "%.0f%%", rate))
            }
            if !profile.isEmpty {
                lines += "  accept by draft position: "
                lines += profile.joined(separator: " ") + "\n"
            }
            stderr.write(Data(lines.utf8))
        }
        if !args.quiet {
            let tokensPerSecond = stats.decodeSeconds > 0
                ? Double(stats.newTokens) / stats.decodeSeconds
                : 0
            let footer = "\n[stop=\(String(describing: stats.reason)) prefill=\(stats.prefillTokens)tok/\(String(format: "%.2f", stats.prefillSeconds))s new=\(stats.newTokens)tok decode=\(String(format: "%.2f", stats.decodeSeconds))s tok/s=\(String(format: "%.3f", tokensPerSecond))]\n"
            stderr.write(Data(footer.utf8))
        }
        return RunResult(exitCode: 0)
    } catch is CancellationError {
        stdout.write(Data("\n".utf8))
        return RunResult(exitCode: 130)
    } catch {
        return errored(stderr, "\(error)", 1)
    }
}

private func errored(_ stderr: FileHandle, _ message: String, _ code: Int32) -> RunResult {
    stderr.write(Data("error: \(message)\n".utf8))
    return RunResult(exitCode: code)
}

/// "auto" enables the paged KV cache above 32k context — the point where the
/// linear FP16 full-attention cache (2 GiB there, growing 64 KiB/token)
/// stops being the sensible default on consumer RAM.
private func kvPagedPolicy(for args: Args) -> RuntimeKVPagedPolicy {
    switch args.kvPaged {
    case "on": return .on
    case "off": return .off
    default: return args.maxContext > 32_768 ? .on : .off
    }
}

private func structuredEvents(_ decoder: StructuredAssistantDecoder?,
                              tokenID: Int32,
                              text: String) throws -> [StructuredAssistantEvent] {
    if let decoder { return try decoder.consume(tokenID: tokenID, delta: text) }
    return text.isEmpty ? [] : [.content(text)]
}

private func structuredTailEvents(_ decoder: StructuredAssistantDecoder?,
                                  text: String) throws -> [StructuredAssistantEvent] {
    if let decoder { return try decoder.consumeFlushedText(text) }
    return text.isEmpty ? [] : [.content(text)]
}

private func visibleAssistantText(_ events: [StructuredAssistantEvent]) throws -> String {
    var visible = ""
    for event in events {
        switch event {
        case .content(let text): visible += text
        case .toolCall: throw ToolCallParserError.malformed
        }
    }
    return visible
}

/// Interactive multi-turn chat. The model is loaded once and every turn
/// re-renders the whole history through the tokenizer's chat template, so the
/// REPL follows whichever dialect the loaded checkpoint uses. Each turn starts
/// from a reset KV cache (`runRawCompletion`'s default), so no state leaks
/// between turns.
private func runChat(args: Args,
                     stdout: FileHandle,
                     stderr: FileHandle) async -> RunResult {
    do {
        let modelURL = URL(fileURLWithPath: args.model)
        let expertStreaming = try resolveExpertStreaming(args.expertCacheSlots,
                                                         modelURL: modelURL)
        let tokenizer = try await MFTokenizer.load(forModelDirectory: modelURL)
        let baseConfig = GenerationConfig(
            maxNewTokens: args.maxNew,
            temperature: args.temperature,
            topK: args.topK,
            topP: args.topP,
            repetitionPenalty: args.repetitionPenalty,
            seed: args.seed,
            stopStrings: args.stops,
            extraStopTokens: [])
        // Interactive chat has no prompt at load time, so `auto` keeps the
        // production default; a fixed size applies to every turn's prefill.
        let prefillChunkTokens: Int
        switch args.prefillChunk {
        case .fixed(let n): prefillChunkTokens = n
        case .auto: prefillChunkTokens = 128
        }
        let runtime = RuntimeConfiguration(
            expertCacheSlots: expertStreaming.configSlots,
            rdadvisePolicy: RDAdvicePolicyMode.parse(args.rdadvise),
            prefillChunkTokens: prefillChunkTokens,
            forceLogitsHead: !baseConfig.isPureGreedy,
            useMapleFlashHead: args.flashHead,
            kvPagedPolicy: kvPagedPolicy(for: args),
            kvTopKPages: args.kvTopKPages,
            kvPoolPagesPerLayer: args.kvPoolPages)

        guard MTLCreateSystemDefaultDevice() != nil else {
            return errored(stderr, "no Metal device", 1)
        }
        let context = try MetalContext()
        let model = try Model.load(
            directoryURL: modelURL,
            device: context.device,
            streamingMode: expertStreaming.mode,
            expertCachePolicy: runtime.modelExpertCachePolicy,
            integrityPolicy: args.verification)
        let forwardRuntime = try ForwardRunnerFactory.make(
            model: model,
            context: context,
            maxContext: args.maxContext,
            runtimeConfiguration: runtime)
        let runner = forwardRuntime.producer
        let scratch = try RawCompletionScratch(context: context,
                                               vocab: model.config.vocabSize,
                                               logitSoftcap: Float(model.config.finalLogitSoftcap))

        let opening: [MFTokenizer.Message] = args.systemPrompt.map {
            [MFTokenizer.Message(role: .system, content: $0)]
        } ?? []
        var history = opening
        let promptTokens = { (messages: [MFTokenizer.Message]) throws -> Int in
            tokenizer.encode(try tokenizer.applyChatTemplate(messages), addBOS: false).count
        }

        stderr.write(Data("Interactive chat. Commands: /clear, /history, /quit.\n".utf8))
        while true {
            stderr.write(Data("\nyou> ".utf8))
            // A nil line is EOF (Ctrl-D): leave the loop and exit cleanly.
            guard let line = readLine(strippingNewline: true) else {
                stderr.write(Data("\n".utf8))
                return RunResult(exitCode: 0)
            }
            let input = line.trimmingCharacters(in: .whitespacesAndNewlines)
            if input.isEmpty { continue }
            if input.hasPrefix("/") {
                switch input {
                case "/quit", "/exit":
                    return RunResult(exitCode: 0)
                case "/clear":
                    history = opening
                    stderr.write(Data("history cleared\n".utf8))
                case "/history":
                    for message in history {
                        stderr.write(Data("[\(message.role.rawValue)] \(message.content ?? "")\n".utf8))
                    }
                default:
                    stderr.write(Data("unknown command \(input); try /clear, /history, or /quit\n".utf8))
                }
                continue
            }

            let turn = history + [MFTokenizer.Message(role: .user, content: input)]
            let fitted = try trimChatHistory(turn, limit: args.maxContext, measure: promptTokens)
            guard fitted.fits else {
                stderr.write(Data(
                    "error: message needs \(fitted.tokens) tokens and does not fit maxContext \(args.maxContext); shorten it or raise --max-context\n".utf8))
                continue
            }
            if fitted.dropped > 0 {
                stderr.write(Data("note: dropped \(fitted.dropped) oldest message(s) to fit the context\n".utf8))
            }
            history = fitted.messages

            let promptIds = tokenizer.encode(try tokenizer.applyChatTemplate(history),
                                             addBOS: false)
            var config = baseConfig
            config.maxNewTokens = min(args.maxNew, args.maxContext - promptIds.count)
            let reply = try await streamChatTurn(promptIds: promptIds,
                                                 config: config,
                                                 tokenizer: tokenizer,
                                                 runner: runner,
                                                 context: context,
                                                 scratch: scratch,
                                                 prefillConfig: forwardRuntime.prefillConfig,
                                                 quiet: args.quiet,
                                                 stdout: stdout,
                                                 stderr: stderr)
            if !reply.isEmpty {
                history.append(MFTokenizer.Message(role: .assistant, content: reply))
            }
        }
    } catch is CancellationError {
        stdout.write(Data("\n".utf8))
        return RunResult(exitCode: 130)
    } catch {
        return errored(stderr, "\(error)", 1)
    }
}

/// Resolved expert-cache choice: the streaming mode `Model.load` uses and the
/// slot count `RuntimeConfiguration` carries. Resident mode has no slot cache,
/// so it carries the largest allowed count for the config's slot-budget
/// consumers, which resident-mode models ignore.
private struct ExpertStreamingResolution {
    let mode: ExpertStreamingMode
    let configSlots: Int
}

private func resolveExpertStreaming(_ choice: ExpertCacheSlotChoice,
                                    modelURL: URL) throws -> ExpertStreamingResolution {
    switch choice {
    case .fixed(let slots):
        return ExpertStreamingResolution(mode: .pread(slotCount: slots),
                                         configSlots: slots)
    case .resident:
        return ExpertStreamingResolution(
            mode: .resident,
            configSlots: RuntimeConfiguration.allowedExpertCacheSlots.max()!)
    case .auto:
        let family = try ManifestReader.peekFamily(directoryURL: modelURL)
        let mode = RuntimeConfiguration.defaultExpertStreamingMode(
            for: family,
            expertPoolBytes: try ExpertPoolInspector.poolByteSize(
                directoryURL: modelURL),
            coreWeightsBytes: try ExpertPoolInspector.coreWeightsByteSize(
                directoryURL: modelURL))
        switch mode {
        case .pread(let slots):
            return ExpertStreamingResolution(mode: mode, configSlots: slots)
        case .resident:
            return ExpertStreamingResolution(
                mode: mode,
                configSlots: RuntimeConfiguration.allowedExpertCacheSlots.max()!)
        }
    }
}

/// Generate one assistant turn, streaming deltas to `stdout`, and return the
/// text that was streamed so the caller can append it to the history.
private func streamChatTurn(promptIds: [Int32],
                            config: GenerationConfig,
                            tokenizer: MFTokenizer,
                            runner: any ContinuableLogitProducer,
                            context: MetalContext,
                            scratch: RawCompletionScratch,
                            prefillConfig: PrefillRuntimeConfig,
                            quiet: Bool,
                            stdout: FileHandle,
                            stderr: FileHandle) async throws -> String {
    var reply = ""
    let decoder = tokenizer.generationPromptStartsInThinking
        ? StructuredAssistantDecoder(tokenizer: tokenizer,
                                     allowedTools: [],
                                     startsInThought: true)
        : nil
    var completionConfig = config
    var stopMatcher = StreamingStopMatcher(stops: decoder == nil ? [] : config.stopStrings)
    if decoder != nil { completionConfig.stopStrings = [] }
    var decodingError: Error?
    var shouldStop = false
    let stats = try await runRawCompletion(
        producer: runner,
        tokenizer: tokenizer,
        promptIds: promptIds,
        config: completionConfig,
        context: context,
        scratch: scratch,
        prefillConfig: prefillConfig,
        shouldStop: { shouldStop }) { progress in
            guard decodingError == nil else { return }
            do {
            switch progress {
            case .prefill:
                break
            case .token(_, let tokenID, let delta):
                let visible = try visibleAssistantText(
                    structuredEvents(decoder, tokenID: tokenID, text: delta))
                let emitted = decoder == nil ? visible : stopMatcher.push(visible)
                if !emitted.isEmpty { stdout.write(Data(emitted.utf8)); reply += emitted }
                if decoder != nil, stopMatcher.isStopped { shouldStop = true }
            case .tail(let tail):
                let visible = try visibleAssistantText(structuredTailEvents(decoder, text: tail))
                let emitted = decoder == nil ? visible : stopMatcher.push(visible)
                if !emitted.isEmpty { stdout.write(Data(emitted.utf8)); reply += emitted }
            }
            } catch {
                decodingError = error
                shouldStop = true
            }
        }
    if let decodingError { throw decodingError }
    if let decoder {
        let visible = try visibleAssistantText(decoder.finish())
        let emitted = stopMatcher.push(visible) + stopMatcher.finish()
        if !emitted.isEmpty { stdout.write(Data(emitted.utf8)); reply += emitted }
    }
    stdout.write(Data("\n".utf8))
    if !quiet {
        let tokensPerSecond = stats.decodeSeconds > 0
            ? Double(stats.newTokens) / stats.decodeSeconds
            : 0
        let footer = "[stop=\(String(describing: stats.reason)) prefill=\(stats.prefillTokens)tok/\(String(format: "%.2f", stats.prefillSeconds))s new=\(stats.newTokens)tok decode=\(String(format: "%.2f", stats.decodeSeconds))s tok/s=\(String(format: "%.3f", tokensPerSecond))]\n"
        stderr.write(Data(footer.utf8))
    }
    return reply
}

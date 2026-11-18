import Darwin
import Foundation
import Mference
import Testing
@testable import MferenceEngine

/// Writes what the llama.cpp backend generates for a fixed prompt set, for the
/// engine program's token-identity gate (G6): each case's rendered prompt and
/// the token ids sampled at greedy decoding, which a script compares with
/// upstream `llama-completion` built from the same pin and run on the same
/// prompt text and settings. Skipped unless all three variables are set:
///
/// - `MFERENCE_LLAMA_TRACE_GGUF`: the `.gguf` file.
/// - `MFERENCE_LLAMA_TRACE_CASES`: a directory with `cases.tsv` (name,
///   messages file, tools file or `-`, `true`/`false` thinking) and the JSON
///   files it names (OpenAI message and tool shapes).
/// - `MFERENCE_LLAMA_TRACE_OUT`: where to write `<case>/prompt.txt`,
///   `<case>/tokens.txt` and `<case>/completion.json`.
///
/// `MFERENCE_LLAMA_TRACE_MAX_TOKENS` caps each completion (default 256).
@Suite("Llama token identity trace", .serialized)
struct LlamaTokenIdentityTraceTests {
    static let environment = ProcessInfo.processInfo.environment
    static let enabled = ["MFERENCE_LLAMA_TRACE_GGUF", "MFERENCE_LLAMA_TRACE_CASES",
                          "MFERENCE_LLAMA_TRACE_OUT"].allSatisfy { environment[$0] != nil }

    @Test(.enabled(if: enabled, "set MFERENCE_LLAMA_TRACE_GGUF, _CASES and _OUT"))
    func writeGreedyTraces() async throws {
        let gguf = URL(fileURLWithPath: Self.environment["MFERENCE_LLAMA_TRACE_GGUF"]!)
        let cases = URL(fileURLWithPath: Self.environment["MFERENCE_LLAMA_TRACE_CASES"]!)
        let out = URL(fileURLWithPath: Self.environment["MFERENCE_LLAMA_TRACE_OUT"]!)
        let maxTokens = Self.environment["MFERENCE_LLAMA_TRACE_MAX_TOKENS"].flatMap(Int.init) ?? 256

        let descriptor = EngineModelDescriptor(
            id: gguf.deletingPathExtension().lastPathComponent,
            displayName: gguf.lastPathComponent,
            engine: .llamaCpp,
            format: .gguf(EngineGGUFMetadata(architecture: "qwen35", contextLength: 262_144,
                                             tensorCount: 0, nextNTensorCount: 0)),
            location: gguf,
            sizeBytes: 0,
            contextLength: 262_144,
            product: nil)
        let clock = ContinuousClock()
        let loadStart = clock.now
        let loaded = try await LlamaBackend().load(descriptor, options: EngineLoadOptions())
        let loadSeconds = (clock.now - loadStart).components
        let model = try #require(loaded as? LlamaModel)
        try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
        try """
        {"load_seconds": \(Double(loadSeconds.seconds) + Double(loadSeconds.attoseconds) / 1e18), \
        "max_context": \(model.traits.maxContext)}
        """.write(to: out.appendingPathComponent("load.json"), atomically: true, encoding: .utf8)

        let table = try String(contentsOf: cases.appendingPathComponent("cases.tsv"), encoding: .utf8)
        for line in table.split(separator: "\n") where !line.isEmpty {
            let fields = line.split(separator: "\t").map(String.init)
            let (name, messagesFile, toolsFile, thinking) = (fields[0], fields[1], fields[2], fields[3] == "true")
            let messages = try LlamaBackendTests.messages(
                String(contentsOf: cases.appendingPathComponent(messagesFile), encoding: .utf8))
            let tools = toolsFile == "-" ? [] : try LlamaBackendTests.tools(
                String(contentsOf: cases.appendingPathComponent(toolsFile), encoding: .utf8))
            let request = EngineRequest(messages: messages, tools: tools,
                                        generationConfig: GenerationConfig(temperature: 0),
                                        maximumCompletionTokens: maxTokens,
                                        reasoningEffort: thinking ? .xhigh : nil)

            let directory = out.appendingPathComponent(name)
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            try await model.renderedPrompt(request)
                .write(to: directory.appendingPathComponent("prompt.txt"), atomically: true, encoding: .utf8)

            await model.discardCachedPrompt()
            let prepared = try await model.prepare(request)
            let completion = try await model.generate(prepared) { _ in }
            let tokens = await model.lastSampledTokens
            try (prepared.promptTokens.map(String.init).joined(separator: " ") + "\n")
                .write(to: directory.appendingPathComponent("prompt-tokens.txt"), atomically: true, encoding: .utf8)
            try (tokens.map(String.init).joined(separator: " ") + "\n")
                .write(to: directory.appendingPathComponent("tokens.txt"), atomically: true, encoding: .utf8)
            let summary: [String: Any] = [
                "content": completion.content,
                "reasoning": completion.reasoningContent ?? NSNull(),
                "tool_calls": completion.toolCalls.map { ["name": $0.name, "arguments": $0.argumentsJSON] },
                "finish_reason": completion.finishReason.rawValue,
                "prompt_tokens": completion.usage.promptTokens,
                "completion_tokens": completion.usage.completionTokens,
                "cached_prompt_tokens": completion.usage.cachedPromptTokens ?? NSNull(),
                "prefill_seconds": completion.timings.prefillSeconds ?? NSNull(),
                "decode_seconds": completion.timings.decodeSeconds ?? NSNull(),
            ]
            try JSONSerialization.data(withJSONObject: summary, options: [.prettyPrinted, .sortedKeys])
                .write(to: directory.appendingPathComponent("completion.json"))
            #expect(!tokens.isEmpty)
            try "{\"phys_footprint_bytes\": \(Self.footprint().current)}\n"
                .write(to: directory.appendingPathComponent("memory.json"), atomically: true, encoding: .utf8)

            // A tool loop's next prompt extends what the context holds, so it
            // reuses that prefix. Record the reuse and whether the warm result
            // equals a cold one (it may legitimately differ: a different prefill
            // split can change the last bits of a logit).
            if let call = completion.toolCalls.first {
                let followUp = EngineRequest(
                    messages: messages + [
                        MFTokenizer.Message(role: .assistant, content: completion.content,
                                            toolCalls: [MFTokenizer.HistoricalToolCall(
                                                id: call.id, name: call.name, arguments: call.arguments)],
                                            reasoningContent: completion.reasoningContent),
                        MFTokenizer.Message(role: .tool, content: #"{"temperature": 18, "condition": "cloudy"}"#,
                                            toolCallID: call.id),
                    ],
                    tools: tools, generationConfig: GenerationConfig(temperature: 0),
                    maximumCompletionTokens: maxTokens, reasoningEffort: thinking ? .xhigh : nil)
                let warm = try await model.generate(try await model.prepare(followUp)) { _ in }
                let warmTokens = await model.lastSampledTokens
                await model.discardCachedPrompt()
                let cold = try await model.generate(try await model.prepare(followUp)) { _ in }
                let coldTokens = await model.lastSampledTokens
                let reuse: [String: Any] = [
                    "prompt_tokens": warm.usage.promptTokens,
                    "warm_cached_prompt_tokens": warm.usage.cachedPromptTokens ?? 0,
                    "cold_cached_prompt_tokens": cold.usage.cachedPromptTokens ?? 0,
                    "warm_prefill_seconds": warm.timings.prefillSeconds ?? 0,
                    "cold_prefill_seconds": cold.timings.prefillSeconds ?? 0,
                    "warm_equals_cold_tokens": warmTokens == coldTokens,
                    "warm_content": warm.content,
                ]
                try JSONSerialization.data(withJSONObject: reuse, options: [.prettyPrinted, .sortedKeys])
                    .write(to: directory.appendingPathComponent("prefix-reuse.json"))
                #expect((warm.usage.cachedPromptTokens ?? 0) > 0)
            }
        }
        await model.unload()
        let footprint = Self.footprint()
        try """
        {"phys_footprint_after_unload_bytes": \(footprint.current), \
        "phys_footprint_peak_bytes": \(footprint.peak)}
        """.write(to: out.appendingPathComponent("memory.json"), atomically: true, encoding: .utf8)
    }

    /// This process's physical footprint now and at its peak (the figure
    /// Activity Monitor shows as Memory), from the task's VM ledger.
    static func footprint() -> (current: UInt64, peak: UInt64) {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let status = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        guard status == KERN_SUCCESS else { return (0, 0) }
        return (info.phys_footprint, UInt64(max(0, info.ledger_phys_footprint_peak)))
    }
}

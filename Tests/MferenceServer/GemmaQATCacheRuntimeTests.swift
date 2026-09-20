import Foundation
import Metal
import Testing
@testable import Mference
@testable import MferenceServerCore

/// One installed runner, no checkpoint copy. The fresh and recovered paths
/// use the same prefill/decode schedule so every FP16 logit must be bit exact.
@Suite(.serialized, .enabled(if: ProcessInfo.processInfo.environment["MFERENCE_GEMMA_QAT_CACHE_PROOF"] != nil))
struct GemmaQATCacheRuntimeTests {
    @Test func installedSourceHistoryRecoveryMatchesFreshTeacherForcedLogits() async throws {
        let env = ProcessInfo.processInfo.environment
        let directory = URL(fileURLWithPath: try #require(env["MFERENCE_GEMMA_QAT_GTURBO"]))
        let output = URL(fileURLWithPath: try #require(env["MFERENCE_GEMMA_QAT_CACHE_PROOF"]))
        try #require(!FileManager.default.fileExists(atPath: output.path), "preserve existing proof")
        let tok = try await MFTokenizer.load(forModelDirectory: directory)
        let context = try MetalContext()
        let model = try Model.load(directoryURL: directory, device: context.device,
            expecting: .gemma4_26B_A4B, streamingMode: .pread(slotCount: 32))
        let runtime = try ForwardRunnerFactory.make(model: model, context: context, maxContext: 1024,
            runtimeConfiguration: .init(expertCacheSlots: 32, prefillChunkTokens: 128, forceLogitsHead: true))
        let runner = try #require(runtime.producer as? RealForwardRunner)
        let logits = try #require(context.device.makeBuffer(length: tok.vocabSize * 2, options: .storageModeShared))
        let domain = try GemmaQATCacheTests.domain(tok)
        let tools: [MFTokenizer.FunctionDefinition] = [.init(name: "lookup", description: "Return a stored value",
            parameters: .object(["type": .string("object"), "properties": .object([:])]))]
        func row() -> [UInt16] {
            Array(UnsafeBufferPointer(start: logits.contents().assumingMemoryBound(to: UInt16.self), count: tok.vocabSize))
        }
        func prefill(_ ids: ArraySlice<Int32>, at position: Int) async throws {
            guard !ids.isEmpty else { return }
            let result = try await runner.prefillChunked(tokens: ids, startPosition: position,
                outputMode: .logits, config: .production(chunkTokens: 128), into: logits, onProgress: { _ in })
            let execution = try #require(result.execution)
            #expect(execution.replayedTokens == 0)
            #expect(execution.batchedTokens == ids.count)
        }
        func render(_ request: ValidatedChatRequest) throws -> [Int32] {
            try tok.encodeChat(messages: request.messages, tools: request.tools, reasoningEffort: request.reasoningEffort)
        }
        var records: [[String: Any]] = []
        for thinking in [false, true] {
            let reasoning: String? = thinking ? "Check." : nil
            func request(_ messages: [MFTokenizer.Message]) -> ValidatedChatRequest {
                .init(reasoningEffort: thinking ? .medium : .off, messages: messages, tools: tools,
                    stream: false, includeUsage: false, generationConfig: .init(maxNewTokens: 32, temperature: 0),
                    maximumCompletionTokens: 32)
            }
            var history: [MFTokenizer.Message] = [.init(role: .user, content: "Look up the value twice, then reply.")]
            for round in 0..<3 {
                let first = request(history)
                let prompt = try render(first)
                let calls: [ParsedToolCall] = round < 2
                    ? [.init(id: "call_\(round)", name: "lookup", arguments: .object([:]), argumentsJSON: "{}")]
                    : []
                let content = calls.isEmpty ? "Found." : ""
                let thought = thinking ? "<|channel>thought\nCheck.<channel|>" : ""
                let generated = tok.encode(thought + (calls.isEmpty ? content : "<|tool_call>call:lookup{}<tool_call|>"), addBOS: false)
                let kv = prompt + generated
                let assistant = MFTokenizer.Message(role: .assistant, content: calls.isEmpty ? content : nil,
                    toolCalls: calls.map { .init(id: $0.id, name: $0.name, arguments: $0.arguments) }, reasoningContent: reasoning)
                let following: MFTokenizer.Message = calls.isEmpty
                    ? .init(role: .user, content: "Now reply READY.")
                    : .init(role: .tool, content: "Stored value is READY.", toolCallID: calls[0].id)
                let next = request(history + [assistant, following])
                let fresh = try render(next)
                let common = zip(kv, fresh).prefix { $0 == $1 }.count
                try #require(common > 0 && fresh.count < 1000)
                let split = min(prompt.count, common)
                runner.reset()
                try await prefill(prompt[..<split], at: 0)
                #expect(try runner.captureGemmaPrefix())
                try await prefill(prompt[split...], at: split)
                for (offset, id) in generated.enumerated() {
                    try await runner.produce(token: id, position: prompt.count + offset, into: logits)
                }
                var cache = ServerPromptCache()
                cache.publish(domain: domain, request: first, content: content, calls: calls,
                    result: RawDecodeResult(prefillTokens: prompt.count, cachedPromptTokens: 0,
                        computedPrefillTokens: prompt.count, prefillSeconds: 0, newTokens: generated.count,
                        decodeSeconds: 0, reason: calls.isEmpty ? .endOfTurn : .toolCalls, kvPosition: kv.count,
                        kvBackedTokenIDs: kv, uncommittedBoundaryTokenIDs: [calls.isEmpty ? tok.endOfTurnID : tok.toolResponseID],
                        prefillExecution: nil), reasoningContent: reasoning)
                let match = cache.match(domain: domain, request: next, renderedPromptIDs: fresh, tokenizer: tok,
                    gemmaRecoverablePrefix: { runner.gemmaRecoverablePrefix(upTo: $0) })
                guard case .hit(let effective, let cached) = match else {
                    Issue.record("installed QAT history did not reuse its available source prefix")
                    continue
                }
                #expect(effective == fresh && cached == common)
                if cached != runner.continuationPosition { _ = try runner.recoverGemmaPrefix(to: cached) }
                try runner.prepareForContinuation(expectedPosition: cached)
                try await prefill(effective[cached...], at: cached)
                var recovered = [row()]
                let followup = Array(tok.encode(" READY", addBOS: false).prefix(3))
                for (offset, id) in followup.enumerated() {
                    try await runner.produce(token: id, position: fresh.count + offset, into: logits)
                    recovered.append(row())
                }
                // Fresh KV and no recovery image, with identical operation
                // boundaries for the shared prefix and recomputed suffix.
                runner.reset()
                #expect(runner.gemmaRecoveryBytes == 0)
                try await prefill(fresh[..<split], at: 0)
                if cached > split {
                    for position in split..<cached {
                        try await runner.produce(token: fresh[position], position: position, into: logits)
                    }
                }
                try await prefill(fresh[cached...], at: cached)
                var actual = [row()]
                for (offset, id) in followup.enumerated() {
                    try await runner.produce(token: id, position: fresh.count + offset, into: logits)
                    actual.append(row())
                }
                let different = zip(actual, recovered).reduce(0) { count, rows in
                    count + zip(rows.0, rows.1).filter { $0 != $1 }.count
                }
                #expect(different == 0, "thinking=\(thinking), round=\(round): every prompt/decode logit must match")
                #expect(actual.allSatisfy { $0.allSatisfy { Float16(bitPattern: $0).isFinite } })
                records.append(["thinking": thinking, "round": round, "prompt_tokens": prompt.count,
                    "source_prompt_ids": fresh, "resumed_prompt_ids": effective, "cached_tokens": cached,
                    "compared_rows": actual.count, "vocabulary": tok.vocabSize, "different_logits": different,
                    "teacher_forced_tokens": followup])
                print("QAT cache proof thinking=\(thinking) round=\(round) reused=\(cached) rows=\(actual.count) different=\(different)")
                history = next.messages
            }
        }
        try #require(records.count == 6)
        try JSONSerialization.data(withJSONObject: ["cases": records, "template_sha256": domain.templateSHA256,
            "manifest_sha256": Sha256Verifier.hashData(Data(contentsOf: directory.appendingPathComponent("manifest.json"))),
            "schedule": "matched production chunked prefill and scalar teacher-forced decode; reset versus recovered KV"],
            options: [.prettyPrinted, .sortedKeys]).write(to: output)
    }
}

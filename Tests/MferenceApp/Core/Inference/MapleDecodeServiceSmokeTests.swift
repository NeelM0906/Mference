import Foundation
import Testing
import Mference
@testable import MferenceAppCore

@Suite("Maple decode service smoke", .serialized)
struct MapleDecodeServiceSmokeTests {
    @Test func realMapleAppRouteStreamsVisibleTextAndReportsNativeRuntime() async throws {
        let environment = ProcessInfo.processInfo.environment
        let modelPath = environment["MFERENCE_MAPLE_GTURBO"]
        let servicePath = environment["MFERENCE_MAPLE_DECODE_SERVICE"]
        guard modelPath != nil || servicePath != nil else { return }
        guard let modelPath, let servicePath,
              !modelPath.isEmpty, !servicePath.isEmpty else {
            Issue.record("MFERENCE_MAPLE_GTURBO and MFERENCE_MAPLE_DECODE_SERVICE are both required")
            return
        }

        let fileManager = FileManager.default
        guard modelPath.hasPrefix("/"), servicePath.hasPrefix("/") else {
            Issue.record("Maple smoke-test paths must be absolute")
            return
        }
        var isModelDirectory = ObjCBool(false)
        guard fileManager.fileExists(atPath: modelPath, isDirectory: &isModelDirectory),
              isModelDirectory.boolValue else {
            Issue.record("MFERENCE_MAPLE_GTURBO is not a directory: \(modelPath)")
            return
        }
        guard fileManager.fileExists(atPath: servicePath),
              fileManager.isExecutableFile(atPath: servicePath) else {
            Issue.record("MFERENCE_MAPLE_DECODE_SERVICE is not executable: \(servicePath)")
            return
        }

        let modelURL = URL(fileURLWithPath: modelPath, isDirectory: true)
        let options = AppRuntimeOptions()
        let request = AppGenerationRequest(
            modelDirectory: modelURL,
            messages: [.init(role: .user,
                             content: "What is 2 + 2? Answer only 4.")],
            maxNewTokens: 128,
            maxContextTokens: AppContextLengthOption.fourK.tokens,
            temperature: 0,
            runtimeOptions: options)

        let client = DecodeServiceInferenceClient(
            serviceURL: URL(fileURLWithPath: servicePath))
        defer { client.shutdown() }
        try await client.ensureLoaded(
            modelDirectory: modelURL,
            maxContextTokens: AppContextLengthOption.fourK.tokens,
            options: options,
            forceLogitsHead: !request.isPureGreedy,
            onState: { _ in })

        var deliveredVisibleText = false
        var diagnostics: AppDiagnostics?
        for try await event in client.generate(request) {
            switch event {
            case .token(let token):
                deliveredVisibleText = deliveredVisibleText
                    || token.textDelta.contains { !$0.isWhitespace }
            case .finished(let result):
                diagnostics = result
            case .cancelled(let result):
                diagnostics = result
                Issue.record("Maple generation was cancelled")
            case .failed(let error, _):
                throw error
            case .prefillProgress:
                break
            }
        }

        let result = try #require(diagnostics)
        let prefill = try #require(result.prefill)
        let visibleText = client.generationTranscriptMailbox.completeText
        #expect(deliveredVisibleText)
        #expect(!visibleText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        #expect(visibleText.range(of: "<think>", options: .caseInsensitive) == nil)
        #expect(visibleText.range(of: "</think>", options: .caseInsensitive) == nil)
        #expect(result.runtimeOptions == options)
        #expect(prefill.requestedMode == .chunked)
        #expect(prefill.executedMode == .sequential)
        #expect(prefill.kvStorageMode == .bf16)
        #expect(prefill.chunkCompleteness == .complete)
        #expect(prefill.unsupportedReason == nil)

        await client.unload()
        client.shutdown()
        #expect(client.currentInferenceMemoryBytes == nil)
    }
}

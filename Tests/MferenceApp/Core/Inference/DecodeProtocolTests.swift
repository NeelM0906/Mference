import Foundation
import Testing
import MferenceDecodeProtocol
@testable import MferenceAppCore

@Suite struct DecodeProtocolTests {
    @Test func loadRequestRoundTripPreservesEveryPublicRuntimeOption() throws {
        let options = DecodeRuntimeOptions(
            expertCacheSlots: 32,
            expertCachePolicy: "lru",
            prefillEnabled: false,
            prefillChunkTokens: 64,
            rdadvisePolicy: "adaptive",
            modelVerification: "trusted-install")
        let request = DecodeLoadRequest(
            modelPath: "/tmp/model.gturbo",
            maxContextTokens: 8192,
            runtimeOptions: options,
            forceLogitsHead: true)

        let pipe = Pipe()
        try pipe.fileHandleForWriting.write(
            contentsOf: DecodeFrameCodec.encode(request))
        try pipe.fileHandleForWriting.close()
        let decoded = try DecodeFrameCodec.read(
            DecodeLoadRequest.self,
            from: pipe.fileHandleForReading)

        #expect(decoded.modelPath == request.modelPath)
        #expect(decoded.maxContextTokens == 8192)
        #expect(decoded.runtimeOptions == options)
        #expect(decoded.forceLogitsHead)
    }

    @Test func terminalEventRoundTripPreservesDiagnosticsAndMemory() throws {
        let runner = DecodeRunnerDiagnostics(
            cb1MillisecondsPerToken: 0.6,
            ioMillisecondsPerToken: 12,
            cb2MillisecondsPerToken: 0.4,
            headMillisecondsPerToken: 1.7,
            rdadviseMillisecondsPerToken: 0,
            rdadviseCallsPerToken: 0,
            rdadviseMegabytesPerToken: 0,
            rdadviseSkippedPerToken: 0,
            rdadviseFailures: 0)
        let event = DecodeServiceEvent(
            kind: .finished,
            generationID: UUID(),
            tokenCount: 256,
            promptTokenCount: 1_017,
            prefillSeconds: 10.2,
            timeToFirstTokenSeconds: 0.04,
            decodeSeconds: 7.7,
            tokensPerSecond: 33.2,
            currentMemoryBytes: 2_000_000_000,
            peakMemoryBytes: 2_100_000_000,
            runner: runner)
        let pipe = Pipe()
        try pipe.fileHandleForWriting.write(
            contentsOf: DecodeFrameCodec.encode(event))
        try pipe.fileHandleForWriting.close()
        let decoded = try DecodeFrameCodec.read(
            DecodeServiceEvent.self,
            from: pipe.fileHandleForReading)

        #expect(decoded.tokenCount == 256)
        #expect(decoded.promptTokenCount == 1_017)
        #expect(decoded.currentMemoryBytes == 2_000_000_000)
        #expect(decoded.peakMemoryBytes == 2_100_000_000)
        #expect(decoded.runner == runner)
    }

    @Test func prefillEventRoundTripPreservesProgress() throws {
        let event = DecodeServiceEvent(
            kind: .prefill,
            generationID: UUID(),
            sequence: 7,
            prefillDone: 128,
            prefillTotal: 514)
        let pipe = Pipe()
        try pipe.fileHandleForWriting.write(
            contentsOf: DecodeFrameCodec.encode(event))
        try pipe.fileHandleForWriting.close()

        let decoded = try DecodeFrameCodec.read(
            DecodeServiceEvent.self,
            from: pipe.fileHandleForReading)

        #expect(decoded.kind == .prefill)
        #expect(decoded.sequence == 7)
        #expect(decoded.prefillDone == 128)
        #expect(decoded.prefillTotal == 514)
    }

    @Test func terminalEventRoundTripPreservesSequentialBF16PrefillDiagnostics() throws {
        let prefill = DecodePrefillDiagnostics(
            requestedMode: "chunked",
            executedMode: "sequential",
            kvStorageMode: "bf16",
            chunkCompleteness: "complete",
            unsupportedReason: nil)
        let event = DecodeServiceEvent(
            kind: .finished,
            generationID: UUID(),
            prefill: prefill)
        let pipe = Pipe()
        try pipe.fileHandleForWriting.write(
            contentsOf: DecodeFrameCodec.encode(event))
        try pipe.fileHandleForWriting.close()

        let decoded = try DecodeFrameCodec.read(
            DecodeServiceEvent.self,
            from: pipe.fileHandleForReading)

        #expect(decoded.prefill == prefill)
    }

    @Test func generationRequestRoundTripPreservesChatRoles() throws {
        let generationID = UUID()
        let runtimeOptions = DecodeRuntimeOptions(
            expertCacheSlots: 32,
            expertCachePolicy: "lru",
            prefillEnabled: false,
            prefillChunkTokens: 64,
            rdadvisePolicy: "adaptive",
            modelVerification: "trusted-install")
        let request = DecodeGenerationRequest(
            messages: [
                DecodeGenerationMessage(role: .system, content: "Memory"),
                DecodeGenerationMessage(role: .user, content: "Question"),
                DecodeGenerationMessage(role: .assistant, content: "Answer"),
                DecodeGenerationMessage(role: .user, content: "Follow-up"),
            ],
            maxNewTokens: 128,
            maxContextTokens: 8_192,
            temperature: 0.2,
            topK: 32,
            topP: 0.7,
            repetitionPenalty: 1.1,
            runtimeOptions: runtimeOptions,
            generationID: generationID)
        let pipe = Pipe()
        try pipe.fileHandleForWriting.write(
            contentsOf: DecodeFrameCodec.encode(request))
        try pipe.fileHandleForWriting.close()

        let decoded = try DecodeFrameCodec.read(
            DecodeGenerationRequest.self,
            from: pipe.fileHandleForReading)

        #expect(decoded.messages == request.messages)
        #expect(decoded.prompt == "Follow-up")
        #expect(decoded.maxNewTokens == 128)
        #expect(decoded.maxContextTokens == 8_192)
        #expect(decoded.temperature == 0.2)
        #expect(decoded.topK == 32)
        #expect(decoded.topP == 0.7)
        #expect(decoded.repetitionPenalty == 1.1)
        #expect(decoded.runtimeOptions == runtimeOptions)
        #expect(decoded.generationID == generationID)
    }

    @Test func legacyPromptInitializerProducesASingleUserMessage() {
        let request = DecodeGenerationRequest(
            prompt: "Question",
            maxNewTokens: 16,
            maxContextTokens: 4_096,
            temperature: 0)

        #expect(request.messages == [
            DecodeGenerationMessage(role: .user, content: "Question"),
        ])
        #expect(request.prompt == "Question")
    }

    @Test func promptCompatibilityAccessorUsesTheLastUserTurn() {
        let request = DecodeGenerationRequest(
            messages: [
                DecodeGenerationMessage(role: .system, content: "Memory"),
                DecodeGenerationMessage(role: .user, content: "First"),
                DecodeGenerationMessage(role: .assistant, content: "Answer"),
            ],
            maxNewTokens: 16,
            maxContextTokens: 4_096,
            temperature: 0)

        #expect(request.prompt == "First")
    }

    @Test func legacyGenerationRequestDefaultsMissingTruncationControls() throws {
        let generationID = UUID()
        let payload = """
        {
          "messages": [{"role": "user", "content": "Question"}],
          "maxNewTokens": 16,
          "maxContextTokens": 4096,
          "temperature": 0.2,
          "repetitionPenalty": 1,
          "runtimeOptions": {
            "expertCacheSlots": 16,
            "expertCachePolicy": "lfu",
            "prefillEnabled": true,
            "prefillChunkTokens": 128,
            "rdadvisePolicy": "off",
            "modelVerification": "full-sha256"
          },
          "generationID": "\(generationID.uuidString)"
        }
        """

        let decoded = try JSONDecoder().decode(
            DecodeGenerationRequest.self, from: Data(payload.utf8))

        #expect(decoded.topK == 64)
        #expect(decoded.topP == 0.95)
    }

    @Test func generationRequestRoundTripPreservesDisabledTruncationControls() throws {
        let request = DecodeGenerationRequest(
            prompt: "Question",
            maxNewTokens: 16,
            maxContextTokens: 4_096,
            temperature: 0.2,
            topK: nil,
            topP: nil)
        let decoded = try JSONDecoder().decode(
            DecodeGenerationRequest.self, from: JSONEncoder().encode(request))

        #expect(decoded.topK == nil)
        #expect(decoded.topP == nil)
    }

    @Test func decodeServiceClientRequiresItsLoadedModelDirectory() {
        let loaded = URL(fileURLWithPath: "/tmp/loaded.gturbo")
        let other = URL(fileURLWithPath: "/tmp/other.gturbo")

        #expect(DecodeServiceInferenceClient.matchesLoadedModelDirectory(
            loaded, loadedDirectory: loaded))
        #expect(!DecodeServiceInferenceClient.matchesLoadedModelDirectory(
            other, loadedDirectory: loaded))
        #expect(!DecodeServiceInferenceClient.matchesLoadedModelDirectory(
            loaded, loadedDirectory: nil))
    }

    @Test func decodeServiceClientMissingExecutableHasSafeIdleTeardown() async {
        let serviceURL = URL(fileURLWithPath: "/private/tmp", isDirectory: true)
            .appendingPathComponent("mference-missing-service-\(UUID())")
        let client = DecodeServiceInferenceClient(serviceURL: serviceURL)
        do {
            try await client.ensureLoaded(
                modelDirectory: URL(fileURLWithPath: "/tmp/model.gturbo"),
                maxContextTokens: 4_096,
                options: AppRuntimeOptions(),
                forceLogitsHead: false,
                onState: { _ in })
            Issue.record("Missing decode service unexpectedly loaded a model")
        } catch let error as AppInferenceError {
            switch error {
            case .modelLoadFailed:
                break
            default:
                Issue.record("Expected modelLoadFailed, received \(error)")
            }
        } catch {
            Issue.record("Expected AppInferenceError, received \(error)")
        }

        client.shutdown()
        client.shutdown()
        await client.unload()
        #expect(client.currentInferenceMemoryBytes == nil)
    }

    @Test func decoderAcceptsAFrameSplitAcrossSingleByteWrites() throws {
        let event = DecodeServiceEvent(
            kind: .snapshot,
            generationID: UUID(),
            sequence: 1,
            textDelta: "caf\u{00E9}",
            tokenCount: 1)
        let frame = try DecodeFrameCodec.encode(event)
        let pipe = Pipe()
        for byte in frame {
            try pipe.fileHandleForWriting.write(contentsOf: Data([byte]))
        }
        try pipe.fileHandleForWriting.close()

        let decoded = try DecodeFrameCodec.read(
            DecodeServiceEvent.self,
            from: pipe.fileHandleForReading)

        #expect(decoded.sequence == 1)
        #expect(decoded.textDelta == "caf\u{00E9}")
    }

    @Test func oversizedPayloadIsRejectedBeforeEncoding() {
        let request = DecodeGenerationRequest(
            prompt: String(repeating: "x", count: DecodeFrameCodec.maximumPayloadBytes + 1),
            maxNewTokens: 1,
            maxContextTokens: 4_096,
            temperature: 0)

        #expect(throws: DecodeFrameError.self) {
            _ = try DecodeFrameCodec.encode(request)
        }
    }

    @Test func oversizedFrameIsRejectedBeforePayloadRead() throws {
        let pipe = Pipe()
        var count = UInt32(DecodeFrameCodec.maximumPayloadBytes + 1).littleEndian
        try pipe.fileHandleForWriting.write(contentsOf: withUnsafeBytes(of: &count) { Data($0) })
        try pipe.fileHandleForWriting.close()

        #expect(throws: DecodeFrameError.self) {
            _ = try DecodeFrameCodec.read(
                DecodeServiceEvent.self,
                from: pipe.fileHandleForReading)
        }
    }
}

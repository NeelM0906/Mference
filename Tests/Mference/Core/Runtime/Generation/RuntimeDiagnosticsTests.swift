import Foundation
import Metal
import Testing
@testable import Mference

@Suite struct RuntimeDiagnosticsTests {
    @Test func executionReportCountsActualWorkAndResetsPerCall() throws {
        var report = PrefillExecutionReport()
        #expect(report.executedMode == .off)
        report.recordBatch(32)
        report.recordBatch(19)
        report.recordReplay(4, reason: "cutover")
        report.recordReplay(5, reason: "cutover")
        #expect(report.executedMode == .mixed)
        #expect(report.batchedTokens == 51)
        #expect(report.replayedTokens == 9)
        #expect(report.computedTokens == 60)
        #expect(report.batchedChunkSizes == [32, 19])
        #expect(report.replayReasons == ["cutover": 9])
        let json = try #require(JSONSerialization.jsonObject(
            with: JSONEncoder().encode(report)) as? [String: Any])
        #expect(json["executedMode"] as? String == "mixed")
        #expect(json["batchedTokens"] as? Int == 51)
        #expect(json["replayedTokens"] as? Int == 9)
        #expect(json["batchedChunkSizes"] as? [Int] == [32, 19])
        #expect(PrefillExecutionReport().computedTokens == 0)
    }

    @Test func snapshotPreservesUnknownMetricsAndDoesNotInventATotal() throws {
        let snapshot = RuntimeMemorySnapshot(bytes: [
            "mappedWeightsPhysicalResidency": nil,
            "filesystemCache": nil,
            "runnerScratchBuffers": nil,
            "completionScratchBuffers": 1028,
        ])
        let result = RawDecodeResult(
            prefillTokens: 100, cachedPromptTokens: 90, computedPrefillTokens: 10,
            prefillSeconds: 0, newTokens: 1, decodeSeconds: 0, reason: .maxTokens,
            kvPosition: 100, kvBackedTokenIDs: [],
            uncommittedBoundaryTokenIDs: [], prefillExecution: nil)
        let diagnostics = RuntimeDiagnostics(result: result, memory: snapshot)
        let json = try #require(JSONSerialization.jsonObject(
            with: Data(diagnostics.jsonLine().utf8)) as? [String: Any])
        #expect(json["schemaVersion"] as? Int == 1)
        #expect(json["cachedPromptTokens"] as? Int == 90)
        #expect(json["computedPrefillTokens"] as? Int == 10)
        #expect(json["prefill"] is NSNull)
        let memory = try #require(json["memory"] as? [String: Any])
        let bytes = try #require(memory["bytes"] as? [String: Any])
        #expect(bytes["mappedWeightsPhysicalResidency"] is NSNull)
        #expect(bytes["filesystemCache"] is NSNull)
        #expect(bytes["runnerScratchBuffers"] is NSNull)
        #expect(bytes["completionScratchBuffers"] as? Int == 1028)
        #expect(bytes["total"] == nil)
        #expect((memory["scope"] as? String)?.contains("not_peak") == true)
    }

    @Test func sharedBuffersAreCountedOnce() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let first = try #require(device.makeBuffer(length: 1024, options: .storageModeShared))
        let second = try #require(device.makeBuffer(length: 2048, options: .storageModeShared))
        #expect(uniqueBufferBytes([]) == 0)
        #expect(uniqueBufferBytes([first, first, second, first])
                == UInt64(first.length + second.length))
    }

    @Test func unreportedConstructionDoesNotClaimCompleteExecution() {
        let diagnostics = PrefillExecutionDiagnostics(
            config: .defaultChunked, executedMode: .unreported)
        #expect(diagnostics.chunkCompleteness == .unreported)
    }
}

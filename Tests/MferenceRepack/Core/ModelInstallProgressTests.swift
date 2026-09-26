import Testing
@testable import MferenceRepackCore

@Suite struct ModelInstallProgressTests {
    @Test func payloadDistinguishesReusedAndNewBytes() {
        let line = ModelInstallProgress.copyingPayload(reusedBytes: 2_000_000_000,
            downloadedThisRunBytes: 1_000_000_000, totalBytes: 10_000_000_000).statusLine
        #expect(line == "Payload 30.0%: 2.00 GB reused + 1.00 GB downloaded this run / 10.00 GB total.")
        #expect(ModelInstallProgress.copyingPayload(reusedBytes: 0,
            downloadedThisRunBytes: 0, totalBytes: 0).statusLine.hasPrefix("Payload 100.0%"))
    }

    @Test func phasesExplainVerificationAndSize() {
        #expect(ModelInstallProgress.downloadingMetadata.statusLine.contains("verifies saved ranges"))
        #expect(ModelInstallProgress.planning(downloadBytes: 5_000_000_000,
            outputBytes: 1_400_000_000).statusLine == "Plan: 5.00 GB source payload; 1.40 GB installed output.")
        #expect(ModelInstallProgress.hashingOutput("model_weights.bin").statusLine == "Verifying output: model_weights.bin")
        #expect(ModelInstallProgress.compactingExperts("packed_experts/layer_00.bin").statusLine
                == "Storing experts without implied biases: packed_experts/layer_00.bin")
        #expect(ModelInstallProgress.finalizing.statusLine == "Finalizing verified install.")
    }
}

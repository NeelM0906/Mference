import Foundation
import Testing
@testable import MferenceRepackCore

extension RemotePayloadCopyTests {
    @Test func swiftQwenOriginalInstallCarriesOwnMTPAndFoldsOnlyZeroCenteredNorms() async throws {
        let input = tmpDirForRemote("swift-qwen-source")
        let output = tmpPathForRemote("swift-qwen-installed")
        defer { cleanUpRemote([input, output]) }
        let snapshot = try SyntheticSnapshot.buildQwen38(at: input, originalBF16: true)
        resetFakeHF()
        FakeHFURLProtocol.files = try remoteFiles(snapshotDir: input, snap: snapshot,
                                                  includeRequiredTokenizer: true, includeOptionalTokenizer: true)
        let result = try await RemoteStreamingRepacker(options: remoteOptions(
            outputDir: output, session: fakeHFSession(), repoID: SupportedModelSource.swiftQwen38.repoID)).run()
        #expect(result.plan.arch.family == .qwen38)
        #expect(result.plan.quantizedAtInstall)
        #expect(result.excludedMultimodalTensorCount == 2)
        let entries = result.plan.resident.entries
        #expect(entries.filter { $0.name.hasPrefix("mtp.") }.count == 15)
        #expect(entries.filter { $0.weightTransform == .addOneBF16 }.count == 18) // 11 trunk + 7 MTP
        #expect(entries.filter { $0.quantSpec != nil }.allSatisfy { $0.quantSpec?.bits == 4 })
        let destination = try Data(contentsOf: URL(fileURLWithPath: output + "/model_weights.bin"))
        let source = try Data(contentsOf: URL(fileURLWithPath: snapshot.shardPath))
        for entry in entries where entry.quantSpec == nil {
            let tensor = entry.sourceWeight
            var expected = Data(source[Int(tensor.absoluteOffset)..<Int(tensor.absoluteOffset + tensor.sizeBytes)])
            if entry.weightTransform == .addOneBF16 {
                for i in stride(from: 0, to: expected.count, by: 2) {
                    let value = UInt16(expected[i]) | UInt16(expected[i + 1]) << 8
                    let folded = Int4AffineEncoder.bf16Bits(Int4AffineEncoder.bf16ToFloat(value) + 1)
                    expected[i] = UInt8(truncatingIfNeeded: folded)
                    expected[i + 1] = UInt8(truncatingIfNeeded: folded >> 8)
                }
            }
            #expect(Data(destination[Int(entry.fileOffset)..<Int(entry.fileOffset + entry.sizeBytes)]) == expected,
                    "\(entry.name)")
            if entry.name.hasSuffix(".conv1d.weight") {
                #expect(entry.logicalShape4 == [256, 4, 1, 0])
            }
        }
        let verified = try VerifiedInstallTool.run(options: VerifyInstallOptions(inputGTurbo: output))
        #expect(verified.unexpectedEntries.isEmpty)
        #expect(!FileManager.default.fileExists(atPath: output + "/packed_experts/layer_00.bin"))
    }
}

@Suite struct SwiftQwenSourceTests {
    @Test func pinnedSeparateFromBase() throws {
        let source = try #require(SupportedModelSource.named("swiftqwen38"))
        #expect(source.repoID == "ukisai/Swift-Qwen3.8-27b")
        #expect(source.revision == "1b30aaaf753fe5c1cb51ada2ea0367a53445359c")
        #expect(source.kind == .originalRepoQuantize)
        #expect(SourceFingerprint.modelID(forIndexSha256: source.sourceIndexSHA256!) == source.modelID)
        #expect(source.modelID != SupportedModelSource.qwen38.modelID)
    }

    @Test func convolutionAxisMoveIsMetadataOnlyAndRejectsUnexpectedLayouts() throws {
        let name = "model.language_model.layers.0.linear_attn.conv1d.weight"
        #expect(try FlashNextPlanner.residentShape([8192, 1, 4], name: name, family: .qwen38) == [8192, 4, 1])
        #expect(throws: RepackError.self) {
            try FlashNextPlanner.residentShape([8192, 4, 1], name: name, family: .qwen38)
        }
        #expect(!FlashNextPlanner.foldsNormBias("model.language_model.layers.0.linear_attn.norm.weight", family: .qwen38))
        #expect(FlashNextPlanner.foldsNormBias("mtp.pre_fc_norm_hidden.weight", family: .qwen38))
    }
}

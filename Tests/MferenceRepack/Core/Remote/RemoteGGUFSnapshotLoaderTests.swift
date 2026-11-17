import CryptoKit
import Foundation
import Testing
@testable import MferenceRepackCore

/// Shares the serialized `RemotePayloadCopyTests` suite because it touches
/// `FakeHFURLProtocol` statics.
extension RemotePayloadCopyTests {
    private func ggufShards(bigHeader: Bool) -> [(String, Data)] {
        // llama.cpp writes split.no / split.count as u16 and split.tensors.count as i32.
        var s0 = SyntheticGGUF()
        s0.kv = [("split.no", 2, SyntheticGGUF.le(UInt16(0))), ("split.count", 2, SyntheticGGUF.le(UInt16(2))),
                 ("split.tensors.count", 5, SyntheticGGUF.le(Int32(2)))]
        if bigHeader {
            // ~2.3 MB of metadata forces more than one header read.
            s0.addStringArray("tokenizer.ggml.tokens", (0..<200_000).map { "tok\($0)" })
        }
        var s1 = SyntheticGGUF()
        s1.kv = [("split.no", 2, SyntheticGGUF.le(UInt16(1))), ("split.count", 2, SyntheticGGUF.le(UInt16(2))),
                 ("split.tensors.count", 5, SyntheticGGUF.le(Int32(2)))]
        s1.tensors = [("blk.0.ffn_gate_exps.weight", [256, 4, 2], 22, 0),
                      ("blk.0.ffn_down_exps.weight", [64, 256, 2], 20, 704)]
        return [("shard-1.gguf", s0.file(dataBytes: 0)),
                // Tail padding makes the data shard larger than the first 1 MiB header read.
                ("shard-2.gguf", s1.file(dataBytes: 704 + 2 * 256 * 36 + (2 << 20)))]
    }

    private func installFake(_ shards: [(String, Data)]) -> GGUFSourcePin {
        resetFakeHF()
        var pinShards: [GGUFSourcePin.Shard] = []
        for (name, data) in shards {
            FakeHFURLProtocol.files[name] = data
            let sha = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
            FakeHFURLProtocol.etagOverrides[name] = "\"\(sha)\""
            pinShards.append(.init(path: name, sha256: sha, size: UInt64(data.count)))
        }
        return GGUFSourcePin(commit: FakeHFURLProtocol.commit, shards: pinShards)
    }

    private func ggufTemporaryRoot(_ tag: String) -> String {
        let path = (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("mference-gguf-\(tag)-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(atPath: path, withIntermediateDirectories: true)
        return path
    }

    private func ggufRemote(_ root: String) -> HuggingFaceRemoteSource {
        HuggingFaceRemoteSource(repoID: "owner/gguf", requestedRevision: FakeHFURLProtocol.commit,
                                downloadSession: fakeHFSession(), baseURL: URL(string: "https://hf.test")!,
                                tempDirectory: root, retryPolicy: RemoteRetryPolicy(attempts: 1, baseDelayNs: 0))
    }

    @Test func ggufLoaderParsesPinnedShardsWithGrowingHeaderReads() async throws {
        let root = ggufTemporaryRoot("gguf-loader")
        defer { try? FileManager.default.removeItem(atPath: root) }
        let pin = installFake(ggufShards(bigHeader: true))
        let snap = try await RemoteGGUFSnapshotLoader.load(remote: ggufRemote(root), pin: pin)
        #expect(snap.headers.count == 2)
        #expect(snap.headers[0].tensors.isEmpty)
        #expect(snap.headers[1].tensors.map { $0.type } == [GGMLType.iq2_s, .iq4_nl])
        #expect(snap.headers[1].tensors[0].sizeBytes == 82 * 4 * 2)
        #expect((FakeHFURLProtocol.requestedRanges["shard-1.gguf"]?.count ?? 0) >= 2)
        // The data shard's header fits in one read; its tail is never fetched.
        #expect(FakeHFURLProtocol.requestedRanges["shard-2.gguf"] == ["bytes=0-\((1 << 20) - 1)"])
        #expect(snap.metadataSourceBytes < UInt64(pin.totalBytes))
        #expect(pin.fingerprint.count == 64)
    }

    @Test func ggufLoaderFailsClosedOnShaSizeOrCommitMismatch() async throws {
        let root = ggufTemporaryRoot("gguf-loader-pin")
        defer { try? FileManager.default.removeItem(atPath: root) }
        let pin = installFake(ggufShards(bigHeader: false))

        var wrongSha = pin.shards
        wrongSha[1] = .init(path: wrongSha[1].path, sha256: String(repeating: "0", count: 64), size: wrongSha[1].size)
        await #expect(throws: RepackError.self) {
            _ = try await RemoteGGUFSnapshotLoader.load(remote: ggufRemote(root),
                                                        pin: GGUFSourcePin(commit: pin.commit, shards: wrongSha))
        }
        var wrongSize = pin.shards
        wrongSize[0] = .init(path: wrongSize[0].path, sha256: wrongSize[0].sha256, size: wrongSize[0].size + 1)
        await #expect(throws: RepackError.self) {
            _ = try await RemoteGGUFSnapshotLoader.load(remote: ggufRemote(root),
                                                        pin: GGUFSourcePin(commit: pin.commit, shards: wrongSize))
        }
        await #expect(throws: RepackError.self) {
            _ = try await RemoteGGUFSnapshotLoader.load(
                remote: ggufRemote(root),
                pin: GGUFSourcePin(commit: String(repeating: "f", count: 40), shards: pin.shards))
        }
        // Shards out of split order.
        await #expect(throws: RepackError.self) {
            _ = try await RemoteGGUFSnapshotLoader.load(
                remote: ggufRemote(root),
                pin: GGUFSourcePin(commit: pin.commit, shards: pin.shards.reversed()))
        }
    }
}

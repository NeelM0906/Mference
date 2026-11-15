import Foundation
import Testing
@testable import MferenceRepackCore

/// Env-gated (`MFERENCE_WS4_REFERENCE_GGUF` = directory holding the four pinned
/// Unsloth UD-IQ3_XXS shards): parses the real headers with the installer's
/// own parser, growing the prefix as the remote loader does, then maps the
/// arch and plans the install. Reads headers only (about 11 MB), never tensor
/// data. Skips without the gate.
@Suite struct GGUFReferenceHeaderTests {
    private static let dir = ProcessInfo.processInfo.environment["MFERENCE_WS4_REFERENCE_GGUF"]

    @Test(.enabled(if: dir != nil)) func realShardHeadersParseAndPlan() throws {
        let pin = try #require(SupportedModelSource.named("qwen38flashnextudiq3xxs")?.ggufPin)
        var headers: [GGUF.Header] = []
        for shard in pin.shards {
            let path = (Self.dir! as NSString).appendingPathComponent((shard.path as NSString).lastPathComponent)
            let handle = try #require(FileHandle(forReadingAtPath: path))
            defer { try? handle.close() }
            let size = try handle.seekToEnd()
            #expect(size == shard.size)
            var want = min(RemoteGGUFSnapshotLoader.initialHeaderRead, Int(size))
            while true {
                try handle.seek(toOffset: 0)
                let prefix = try handle.read(upToCount: want) ?? Data()
                switch try GGUF.parseHeader(path: shard.path, fileSize: size, prefix: prefix) {
                case .complete(let h): headers.append(h)
                case .needBytes(let n): want = n; continue
                }
                break
            }
        }
        try RemoteGGUFSnapshotLoader.validateSplit(headers: headers, pin: pin)
        #expect(headers.map(\.tensors.count) == [0, 1224 - headers[2].tensors.count - headers[3].tensors.count,
                                                 headers[2].tensors.count, headers[3].tensors.count])
        let all = headers.flatMap(\.tensors)
        #expect(all.count == 1224)
        let arch = try ArchInfo.fromGGUFQwen4Exp(metadata: headers[0].metadata, tensors: all, path: "shard1")
        #expect(arch.family == .qwen38flashnextudiq3xxs)
        _ = try GGUFTokenizer.synthesize(metadata: headers[0].metadata, path: "shard1")
        _ = try GTurboJSON.GGUFPleHash.from(metadata: headers[0].metadata, path: "shard1")
        let plan = try FlashNextGGUFPlanner.plan(headers: headers, shardPaths: pin.shards.map(\.path), arch: arch,
                                                 outputDir: "/out", matchedModelID: nil)
        #expect(plan.layers.count == 48 && plan.plePools.count == 1)
    }
}

import Foundation
import Metal
import Testing
@testable import Mference

@Suite(.serialized, .enabled(if:
    ProcessInfo.processInfo.environment["MFERENCE_GEMMA_QAT_GREEDY_REFERENCE"] != nil))
struct GemmaQATGreedyHeadTests {
    struct Capture: Decodable {
        let manifest_sha256: String
        let vocab: Int
        let items: [Item]
        struct Item: Decodable {
            let name: String
            let sequence: [Int32]
            let positions: [Int]
            let file: String
        }
    }

    @Test func productionGreedyHeadMatchesFrozenReferenceWinners() async throws {
        let environment = ProcessInfo.processInfo.environment
        let directory = URL(fileURLWithPath: try #require(environment["MFERENCE_GEMMA_QAT_GTURBO"]))
        let reference = URL(fileURLWithPath: try #require(environment["MFERENCE_GEMMA_QAT_GREEDY_REFERENCE"]))
        // These exact v16 logits were independently checked against the pinned
        // source at all 158 positions, with zero error. No new native output is
        // allowed to replace this frozen oracle. The winner-gap limit is the
        // original reference gate's 0.15 after softcap, not a new tolerance.
        let hashes = [
            "meta.json": "73391f891d45212a528886389af6fa99085a85ce81df2195eee21e26615647e5",
            "reference.jsonl": "f79ee8e1202421ad1b3df3a3eb762524d9a225b6a8ef1bf9abd381db55738c81",
            "capital-scalar.f16": "df48b7962a017536b53e964906a444dda1bed707a257f7eeecc3f8f09106a3ae",
            "capital-chunked.f16": "4a0ed3e29e7eec7c5f615bb973b8a0c76572ebe3015c20fc8c4b8b1298d29b71",
            "arithmetic-scalar.f16": "697457bf4a30c01ca3b6d1889b8e2eb8887630e4c1e8901cd0639c886cb4cf91",
            "arithmetic-chunked.f16": "0b2532e990cba67d6b82e8a6e21499a0744651b1856b95687275a90fcd5e2a19",
            "multi-chunk-scalar.f16": "5e952330343bc406b64e23e26371ef871743d07d4b4a636583355704e02238d7",
            "multi-chunk-chunked.f16": "3c920c4db627d8c81478b25d11c9c44795c5c21295fb98240a89b6d3e1bfc8d9",
        ]
        for (file, hash) in hashes {
            try Sha256Verifier.verifyFile(at: reference.appendingPathComponent(file), named: file, expectedHex: hash)
        }
        let capture = try JSONDecoder().decode(Capture.self,
            from: Data(contentsOf: reference.appendingPathComponent("meta.json")))
        try #require(capture.vocab == 262_144)
        #expect(Sha256Verifier.hashData(try Data(contentsOf: directory.appendingPathComponent("manifest.json")))
                == capture.manifest_sha256)
        let context = try MetalContext()
        let model = try Model.load(directoryURL: directory, device: context.device,
                                   expecting: .gemma4_26B_A4B)
        let runtime = try ForwardRunnerFactory.make(model: model, context: context, maxContext: 256)
        let runner = try #require(runtime.producer as? RealForwardRunner)
        try #require(runner.usesFusedGreedyHead)
        let logits = try #require(context.device.makeBuffer(length: capture.vocab * 2, options: .storageModeShared))
        var results: [[String: Any]] = []
        var maximumGap = 0.0
        var differentWinners = 0
        for item in capture.items {
            runner.reset()
            let data = try Data(contentsOf: reference.appendingPathComponent(item.file), options: .mappedIfSafe)
            try #require(data.count == item.positions.count * capture.vocab * 2)
            func compare(_ row: Int, position: Int, token: UInt32) throws {
                let actual = Int(token)
                try #require(actual < capture.vocab)
                let (expected, gap) = data.withUnsafeBytes { bytes -> (Int, Double) in
                    let values = bytes.bindMemory(to: Float16.self)
                    let offset = row * capture.vocab
                    var best = 0
                    for candidate in 1..<capture.vocab where values[offset + candidate] > values[offset + best] {
                        best = candidate
                    }
                    let gap = 30 * (tanh(Double(values[offset + best]) / 30)
                        - tanh(Double(values[offset + actual]) / 30))
                    return (best, gap)
                }
                maximumGap = max(maximumGap, gap)
                if actual != expected { differentWinners += 1 }
                results.append(["name": item.name, "position": position,
                    "expected": expected, "actual": actual, "winner_reference_gap": gap])
                #expect(gap.isFinite && gap <= 0.15,
                    "\(item.name) position \(position): source winner \(expected), native \(actual), gap \(gap)")
            }
            if item.name.hasSuffix("-scalar") {
                for (position, token) in item.sequence.enumerated() {
                    try await runner.produce(token: token, position: position, into: logits)
                    try compare(position, position: position, token: runner.lastGreedyToken)
                }
            } else {
                let prefixCount = item.sequence.count - 2
                let result = try await runner.prefillChunked(tokens: item.sequence[..<prefixCount], startPosition: 0,
                    outputMode: .greedyIfAvailable, config: runtime.prefillConfig, into: logits, onProgress: { _ in })
                guard case .greedyToken(let token) = result.seed else {
                    Issue.record("normal greedy prefill did not use the fused head")
                    continue
                }
                let execution = try #require(result.execution)
                #expect(execution.replayedTokens == 0)
                #expect(execution.batchedChunkSizes == (prefixCount > 128 ? [128, prefixCount - 128] : [prefixCount]))
                try compare(0, position: prefixCount - 1, token: token)
                for position in prefixCount..<item.sequence.count {
                    try await runner.produce(token: item.sequence[position], position: position, into: logits)
                    try compare(position - prefixCount + 1, position: position, token: runner.lastGreedyToken)
                }
            }
        }
        #expect(results.count == 158)
        let report: [String: Any] = ["positions": results.count, "different_winners": differentWinners,
            "maximum_winner_reference_gap": maximumGap, "limit": 0.15, "items": results]
        try JSONSerialization.data(withJSONObject: report, options: [.sortedKeys, .prettyPrinted])
            .write(to: reference.appendingPathComponent("greedy-head.json"))
        print("[qat-greedy] \(results.count) positions, \(differentWinners) different winners, max source gap \(maximumGap)")
    }
}

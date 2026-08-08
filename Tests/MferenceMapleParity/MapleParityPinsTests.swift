import CryptoKit
import Foundation
import Testing

@testable import MferenceMapleParityCore

@Suite struct MapleParityPinsTests {
    @Test func sourceAndFixturePinsStayAligned() throws {
        let source = MapleParityPins.source
        #expect(source.name == "maple")
        #expect(source.repoID == "deepgrove/maple-preview-2bit-mlx")
        #expect(source.revision == "361db5da5e74ff6fcdd852d478e1f266ce11013a")
        #expect(source.sourceIndexSHA256 == "56000110535c5023b43209a5c142035e12c1cde7b1118759cc9f86335d46ef95")
        #expect(source.modelID == "maple-preview-2bit-mlx")

        let snapshotData = try fixtureData("maple-snapshot")
        #expect(sha256(snapshotData) == MapleParityPins.sourceSnapshotManifestSHA256)
        let snapshot = try fixture(snapshotData)
        #expect(snapshot["schema"] as? String == "mference.maple.snapshot.v1")
        #expect(snapshot["repo_id"] as? String == source.repoID)
        #expect(snapshot["revision"] as? String == source.revision)
        let files = try #require(snapshot["files"] as? [[String: Any]])
        let byPath = Dictionary(uniqueKeysWithValues: files.compactMap { file in
            (file["path"] as? String).map { ($0, file) }
        })
        #expect(byPath["model.safetensors.index.json"]?["size"] as? Int == 40_054)
        #expect(byPath["model.safetensors.index.json"]?["sha256"] as? String == source.sourceIndexSHA256)
        #expect(byPath["config.json"]?["sha256"] as? String == MapleParityPins.configSHA256)
        #expect(byPath["tokenizer.json"]?["sha256"] as? String == MapleParityPins.tokenizerSHA256)
    }

    @Test func corpusFixturePinsStayAligned() throws {
        let corpusData = try fixtureData("RavenNormalization")
        #expect(sha256(corpusData) == MapleParityPins.corpusManifestSHA256)
        let corpus = try fixture(corpusData)
        #expect(corpus["schema"] as? String == "mference.maple.corpus.v1")
        #expect(corpus["source_url"] as? String == MapleParityPins.corpusSourceURL)
        let expected = try #require(corpus["expected"] as? [String: Any])
        #expect(expected["sha256"] as? String == MapleParityPins.corpusSHA256)
        #expect(expected["utf8_bytes"] as? Int == 6_877)
    }

    private func fixtureData(_ name: String) throws -> Data {
        let url = try #require(Bundle.module.url(forResource: name, withExtension: "json", subdirectory: "Fixtures"))
        return try Data(contentsOf: url)
    }

    private func fixture(_ data: Data) throws -> [String: Any] {
        try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
    }

    private func sha256(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
}

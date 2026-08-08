import Foundation

@testable import MferenceMapleParityCore

func testMetadata(
    engine: String = MapleParityPins.engine,
    engineVersion: String = "candidate",
    engineSourceRevision: String = String(repeating: "c", count: 40),
    engineSourceDirty: Bool = false,
    runtimeVersions: [String: String] = [
        "architecture": "arm64", "harness": String(repeating: "c", count: 40),
        "macos": "15.0.0", "swift": "6.1.0",
    ],
    modelID: String = MapleParityPins.source.modelID,
    positions: Int = 2,
    corpusTokenCount: Int = 2,
    tokenIDsSHA256: String = "49a64717d5d4cb19952e6eac2946415cf6879adacf9908e7d872332d32c6e684",
    topK: Int = 2,
    vocabSize: Int = 8,
    logitDType: String = MapleParityPins.candidateLogitDType,
    expertCacheSlots: Int = MapleParityPins.expertCacheSlots,
    integrityPolicy: String = "full-sha256",
    acceptanceEligible: Bool = true
) -> MapleParityMetadata {
    MapleParityMetadata(
        schema: MapleParityPins.schema,
        engine: engine,
        engineVersion: engineVersion,
        engineSourceRevision: engineSourceRevision,
        engineSourceDirty: engineSourceDirty,
        engineBinarySHA256: String(repeating: "b", count: 64),
        command: ["test"],
        runtimeVersions: runtimeVersions,
        modelID: modelID,
        sourceRepoID: MapleParityPins.source.repoID,
        modelRevision: MapleParityPins.modelRevision,
        sourceIndexSHA256: MapleParityPins.sourceIndexSHA256,
        sourceSnapshotManifestSHA256: MapleParityPins.sourceSnapshotManifestSHA256,
        configSHA256: MapleParityPins.configSHA256,
        tokenizerSHA256: MapleParityPins.tokenizerSHA256,
        corpusSourceURL: MapleParityPins.corpusSourceURL,
        corpusSHA256: MapleParityPins.corpusSHA256,
        corpusTokenCount: corpusTokenCount,
        positions: positions,
        tokenIDsSHA256: tokenIDsSHA256,
        tokenPolicy: MapleParityPins.tokenPolicy,
        topK: topK,
        topKTieBreak: MapleParityPins.topKTieBreak,
        vocabSize: vocabSize,
        logitDType: logitDType,
        exactLMHead: true,
        useFlashHead: false,
        expertCacheSlots: expertCacheSlots,
        integrityPolicy: integrityPolicy,
        acceptanceEligible: acceptanceEligible)
}

func testPosition(_ position: Int, inputID: Int, targetID: Int?, top: [MapleParityTopLogit] = [
    MapleParityTopLogit(id: 3, logit: 1), MapleParityTopLogit(id: 4, logit: 0.5),
]) -> MapleParityPosition {
    MapleParityPosition(position: position, inputID: inputID, targetID: targetID, top: top)
}

func testTrace(metadata: MapleParityMetadata = testMetadata(), positions: [MapleParityPosition]? = nil) -> MapleParityTrace {
    let records = positions ?? [testPosition(0, inputID: 1, targetID: 2), testPosition(1, inputID: 2, targetID: nil)]
    return MapleParityTrace(
        metadata: metadata,
        positions: records,
        summary: MapleParitySummary(positions: records.count, loadSeconds: 0, elapsedSeconds: 1, positionsPerSecond: 2, exitCode: 0))
}

func temporaryTraceURL() -> URL {
    FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString).appendingPathExtension("jsonl")
}

func jsonLine<T: Encodable>(_ value: T, type: String) throws -> String {
    let data = try JSONEncoder().encode(value)
    var object = try JSONSerialization.jsonObject(with: data) as! [String: Any]
    object["record_type"] = type
    return String(decoding: try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys]), as: UTF8.self)
}

func traceLines(_ trace: MapleParityTrace) throws -> [String] {
    try [jsonLine(trace.metadata, type: "metadata")]
        + trace.positions.map { try jsonLine($0, type: "position") }
        + [jsonLine(trace.summary, type: "summary")]
}

func replacingObject(_ line: String, _ update: (inout [String: Any]) -> Void) throws -> String {
    var object = try JSONSerialization.jsonObject(with: Data(line.utf8)) as! [String: Any]
    update(&object)
    return String(decoding: try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys]), as: UTF8.self)
}

func writeRawTrace(_ lines: [String]) throws -> URL {
    let url = temporaryTraceURL()
    try (lines.joined(separator: "\n") + "\n").write(to: url, atomically: true, encoding: .utf8)
    return url
}

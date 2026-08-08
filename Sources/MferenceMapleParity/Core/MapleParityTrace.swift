import Foundation
import Mference

public struct MapleParityMetadata: Codable, Equatable, Sendable {
    public let schema: String
    public let engine: String
    public let engineVersion: String
    public let engineSourceRevision: String
    public let engineSourceDirty: Bool
    public let engineBinarySHA256: String
    public let command: [String]
    public let runtimeVersions: [String: String]
    public let modelID: String
    public let sourceRepoID: String
    public let modelRevision: String
    public let sourceIndexSHA256: String
    public let sourceSnapshotManifestSHA256: String
    public let configSHA256: String
    public let tokenizerSHA256: String
    public let corpusSourceURL: String
    public let corpusSHA256: String
    public let corpusTokenCount: Int
    public let positions: Int
    public let tokenIDsSHA256: String
    public let tokenPolicy: String
    public let topK: Int
    public let topKTieBreak: String
    public let vocabSize: Int
    public let logitDType: String
    public let exactLMHead: Bool
    public let useFlashHead: Bool
    public let expertCacheSlots: Int
    public let integrityPolicy: String
    public let acceptanceEligible: Bool

    enum CodingKeys: String, CodingKey {
        case schema, engine
        case engineVersion = "engine_version"
        case engineSourceRevision = "engine_source_revision"
        case engineSourceDirty = "engine_source_dirty"
        case engineBinarySHA256 = "engine_binary_sha256"
        case command
        case runtimeVersions = "runtime_versions"
        case modelID = "model_id"
        case sourceRepoID = "source_repo_id"
        case modelRevision = "model_revision"
        case sourceIndexSHA256 = "source_index_sha256"
        case sourceSnapshotManifestSHA256 = "source_snapshot_manifest_sha256"
        case configSHA256 = "config_sha256"
        case tokenizerSHA256 = "tokenizer_sha256"
        case corpusSourceURL = "corpus_source_url"
        case corpusSHA256 = "corpus_sha256"
        case corpusTokenCount = "corpus_token_count"
        case positions
        case tokenIDsSHA256 = "token_ids_sha256"
        case tokenPolicy = "token_policy"
        case topK = "top_k"
        case topKTieBreak = "top_k_tie_break"
        case vocabSize = "vocab_size"
        case logitDType = "logit_dtype"
        case exactLMHead = "exact_lm_head"
        case useFlashHead = "use_flash_head"
        case expertCacheSlots = "expert_cache_slots"
        case integrityPolicy = "integrity_policy"
        case acceptanceEligible = "acceptance_eligible"
    }
}

public struct MapleParityTopLogit: Codable, Equatable, Sendable {
    public let id: Int
    public let logit: Float
}

public struct MapleParityPosition: Codable, Equatable, Sendable {
    public let position: Int
    public let inputID: Int
    public let targetID: Int?
    public let top: [MapleParityTopLogit]

    enum CodingKeys: String, CodingKey {
        case position, top
        case inputID = "input_id"
        case targetID = "target_id"
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(position, forKey: .position)
        try container.encode(inputID, forKey: .inputID)
        if let targetID {
            try container.encode(targetID, forKey: .targetID)
        } else {
            try container.encodeNil(forKey: .targetID)
        }
        try container.encode(top, forKey: .top)
    }
}

public struct MapleParitySummary: Codable, Equatable, Sendable {
    public let positions: Int
    public let loadSeconds: Double
    public let elapsedSeconds: Double
    public let positionsPerSecond: Double
    public let exitCode: Int

    enum CodingKeys: String, CodingKey {
        case positions
        case loadSeconds = "load_seconds"
        case elapsedSeconds = "elapsed_seconds"
        case positionsPerSecond = "positions_per_second"
        case exitCode = "exit_code"
    }
}

public struct MapleParityTrace: Equatable, Sendable {
    public let metadata: MapleParityMetadata
    public let positions: [MapleParityPosition]
    public let summary: MapleParitySummary
}

public enum MapleParityTraceWriter {
    public static func write(_ trace: MapleParityTrace, to output: URL) throws {
        guard !FileManager.default.fileExists(atPath: output.path) else {
            throw MapleParityError.io("refusing to overwrite \(output.path)")
        }
        let directory = output.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        guard FileManager.default.createFile(atPath: output.path, contents: nil),
              let handle = FileHandle(forWritingAtPath: output.path) else {
            throw MapleParityError.io("unable to create \(output.path)")
        }
        defer { try? handle.close() }
        try writeRecord(trace.metadata, type: "metadata", to: handle)
        for position in trace.positions {
            try writeRecord(position, type: "position", to: handle)
        }
        try writeRecord(trace.summary, type: "summary", to: handle)
    }

    private static func writeRecord<T: Encodable>(_ value: T, type: String, to handle: FileHandle) throws {
        let data = try JSONEncoder().encode(value)
        guard var object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw MapleParityError.io("unable to encode JSONL record")
        }
        object["record_type"] = type
        let line = try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys, .withoutEscapingSlashes])
        handle.write(line)
        handle.write(Data([0x0A]))
    }
}

public enum MapleParityTraceReader {
    public static func read(from input: URL) throws -> MapleParityTrace {
        let text: String
        do {
            text = try String(contentsOf: input, encoding: .utf8)
        } catch {
            throw MapleParityError.io("unable to read \(input.path): \(error.localizedDescription)")
        }
        let lines = text.split(separator: "\n", omittingEmptySubsequences: false)
        var metadata: MapleParityMetadata?
        var positions: [MapleParityPosition] = []
        var summary: MapleParitySummary?
        for (offset, line) in lines.enumerated() {
            guard !line.isEmpty else {
                if offset + 1 == lines.count { continue }
                throw MapleParityError.invalid("\(input.path):\(offset + 1): blank JSONL record")
            }
            let data = Data(line.utf8)
            let object: [String: Any]
            do {
                guard let decoded = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                    throw MapleParityError.invalid("record is not an object")
                }
                object = decoded
            } catch let error as MapleParityError {
                throw MapleParityError.invalid("\(input.path):\(offset + 1): \(error)")
            } catch {
                throw MapleParityError.invalid("\(input.path):\(offset + 1): invalid JSON")
            }
            guard let type = object["record_type"] as? String else {
                throw MapleParityError.invalid("\(input.path):\(offset + 1): record_type is required")
            }
            switch type {
            case "metadata":
                guard metadata == nil, positions.isEmpty, summary == nil else {
                    throw MapleParityError.invalid("\(input.path):\(offset + 1): metadata must be first and unique")
                }
                try requireKeys(object, exactly: metadataKeys, path: input.path, line: offset + 1)
                metadata = try decode(MapleParityMetadata.self, data: data, path: input.path, line: offset + 1)
            case "position":
                guard metadata != nil, summary == nil else {
                    throw MapleParityError.invalid("\(input.path):\(offset + 1): position is out of order")
                }
                try requireKeys(object, exactly: positionKeys, path: input.path, line: offset + 1)
                guard let top = object["top"] as? [[String: Any]],
                      top.allSatisfy({ Set($0.keys) == Set(["id", "logit"]) }) else {
                    throw MapleParityError.invalid("\(input.path):\(offset + 1): malformed top-k object")
                }
                let position = try decode(MapleParityPosition.self, data: data, path: input.path, line: offset + 1)
                positions.append(position)
            case "summary":
                guard metadata != nil, summary == nil else {
                    throw MapleParityError.invalid("\(input.path):\(offset + 1): summary must be last and unique")
                }
                try requireKeys(object, exactly: summaryKeys, path: input.path, line: offset + 1)
                summary = try decode(MapleParitySummary.self, data: data, path: input.path, line: offset + 1)
            default:
                throw MapleParityError.invalid("\(input.path):\(offset + 1): unknown record_type \(type)")
            }
        }
        guard let metadata, let summary else {
            throw MapleParityError.invalid("\(input.path): incomplete trace")
        }
        let trace = MapleParityTrace(metadata: metadata, positions: positions, summary: summary)
        try validate(trace, path: input.path)
        return trace
    }

    private static let metadataKeys: Set<String> = [
        "record_type", "schema", "engine", "engine_version", "engine_source_revision",
        "engine_source_dirty", "engine_binary_sha256", "command", "runtime_versions", "model_id",
        "source_repo_id", "model_revision", "source_index_sha256", "source_snapshot_manifest_sha256",
        "config_sha256", "tokenizer_sha256", "corpus_source_url", "corpus_sha256", "corpus_token_count",
        "positions", "token_ids_sha256", "token_policy", "top_k", "top_k_tie_break",
        "vocab_size", "logit_dtype", "exact_lm_head", "use_flash_head", "expert_cache_slots",
        "integrity_policy", "acceptance_eligible",
    ]
    private static let positionKeys: Set<String> = ["record_type", "position", "input_id", "target_id", "top"]
    private static let summaryKeys: Set<String> = ["record_type", "positions", "load_seconds", "elapsed_seconds", "positions_per_second", "exit_code"]

    private static func requireKeys(_ object: [String: Any], exactly keys: Set<String>, path: String, line: Int) throws {
        guard Set(object.keys) == keys else {
            throw MapleParityError.invalid("\(path):\(line): unexpected or missing JSON keys")
        }
    }

    private static func decode<T: Decodable>(_ type: T.Type, data: Data, path: String, line: Int) throws -> T {
        do { return try JSONDecoder().decode(T.self, from: data) }
        catch { throw MapleParityError.invalid("\(path):\(line): \(error.localizedDescription)") }
    }

    private static func validate(_ trace: MapleParityTrace, path: String) throws {
        let metadata = trace.metadata
        guard metadata.schema == MapleParityPins.schema, metadata.vocabSize > 0,
              metadata.topK > 0, metadata.topK <= metadata.vocabSize,
              metadata.positions > 0, metadata.corpusTokenCount >= metadata.positions,
              !metadata.command.isEmpty,
              MapleParityPreflight.isLowerHex(metadata.engineBinarySHA256, count: 64),
              trace.summary.loadSeconds.isFinite, trace.summary.loadSeconds >= 0,
              trace.summary.elapsedSeconds.isFinite, trace.summary.elapsedSeconds >= 0,
              trace.summary.positionsPerSecond.isFinite, trace.summary.positionsPerSecond >= 0,
              trace.summary.exitCode >= 0 else {
            throw MapleParityError.invalid("\(path): invalid trace metadata")
        }
        guard trace.positions.count == metadata.positions, trace.summary.positions == metadata.positions else {
            throw MapleParityError.invalid("\(path): trace position count does not match metadata")
        }
        for (expected, record) in trace.positions.enumerated() {
            guard record.position == expected,
                  record.inputID >= 0, record.inputID < metadata.vocabSize,
                  record.top.count == metadata.topK,
                  Set(record.top.map(\.id)).count == metadata.topK,
                  record.top.allSatisfy({ $0.id >= 0 && $0.id < metadata.vocabSize && $0.logit.isFinite }) else {
                throw MapleParityError.invalid("\(path): malformed position \(expected)")
            }
            let canonical = record.top.sorted { left, right in
                left.logit == right.logit ? left.id < right.id : left.logit > right.logit
            }
            guard canonical == record.top else {
                throw MapleParityError.invalid("\(path): non-canonical top-k at position \(expected)")
            }
            let target = expected + 1 < trace.positions.count ? trace.positions[expected + 1].inputID : nil
            guard target == nil ? (record.targetID == nil || (metadata.positions < metadata.corpusTokenCount && record.targetID! >= 0 && record.targetID! < metadata.vocabSize)) : record.targetID == target else {
                throw MapleParityError.invalid("\(path): target mismatch at position \(expected)")
            }
        }
        let tokenJSON = "[" + trace.positions.map { String($0.inputID) }.joined(separator: ",") + "]"
        guard Sha256Verifier.hashData(Data(tokenJSON.utf8)) == metadata.tokenIDsSHA256 else {
            throw MapleParityError.invalid("\(path): token_ids_sha256 does not match inputs")
        }
    }

}

import Foundation

public struct MapleParityDivergence: Equatable, Sendable {
    public let position: Int
    public let reason: String
}

public struct MapleParityComparison: Equatable, Sendable {
    public let positions: Int
    public let top1Matches: Int
    public let orderedMatches: Int
    public let firstDivergence: MapleParityDivergence?
    public let metadataErrors: [String]

    public var accepted: Bool { metadataErrors.isEmpty && firstDivergence == nil }
}

public enum MapleParityComparator {
    /// Compares the pinned full trace contract with no numerical tolerance.
    public static func compare(reference: MapleParityTrace,
                               candidate: MapleParityTrace) -> MapleParityComparison {
        let metadataErrors = acceptanceMetadataErrors(reference: reference,
                                                       candidate: candidate)
        let count = min(reference.positions.count, candidate.positions.count)
        var top1Matches = 0
        var orderedMatches = 0
        var first: MapleParityDivergence?
        for position in 0..<count {
            let expected = reference.positions[position]
            let actual = candidate.positions[position]
            if expected.top.first?.id == actual.top.first?.id { top1Matches += 1 }
            if expected.top.map(\.id) == actual.top.map(\.id) { orderedMatches += 1 }
            guard first == nil else { continue }
            if expected.inputID != actual.inputID || expected.targetID != actual.targetID {
                first = MapleParityDivergence(position: position, reason: "token sequence")
            } else if expected.top.map(\.id) != actual.top.map(\.id) {
                first = MapleParityDivergence(position: position, reason: "ordered top-k IDs")
            } else if zip(expected.top, actual.top).contains(where: { $0.logit != $1.logit }) {
                first = MapleParityDivergence(position: position, reason: "top-k logits")
            }
        }
        if first == nil, reference.positions.count != candidate.positions.count {
            first = MapleParityDivergence(position: count, reason: "position count")
        }
        return MapleParityComparison(positions: count, top1Matches: top1Matches,
                                     orderedMatches: orderedMatches, firstDivergence: first,
                                     metadataErrors: metadataErrors)
    }

    private static func acceptanceMetadataErrors(reference: MapleParityTrace,
                                                 candidate: MapleParityTrace) -> [String] {
        let ref = reference.metadata
        let got = candidate.metadata
        var errors: [String] = []
        if ref.engine != "mlx" { errors.append("reference engine must be mlx") }
        if got.engine != MapleParityPins.engine { errors.append("candidate engine must be mference") }
        let commonStrings: [(String, String, String)] = [
            ("schema", ref.schema, got.schema),
            ("model_id", ref.modelID, got.modelID),
            ("source_repo_id", ref.sourceRepoID, got.sourceRepoID),
            ("model_revision", ref.modelRevision, got.modelRevision),
            ("source_index_sha256", ref.sourceIndexSHA256, got.sourceIndexSHA256),
            ("source_snapshot_manifest_sha256", ref.sourceSnapshotManifestSHA256, got.sourceSnapshotManifestSHA256),
            ("config_sha256", ref.configSHA256, got.configSHA256),
            ("tokenizer_sha256", ref.tokenizerSHA256, got.tokenizerSHA256),
            ("corpus_source_url", ref.corpusSourceURL, got.corpusSourceURL),
            ("corpus_sha256", ref.corpusSHA256, got.corpusSHA256),
            ("token_ids_sha256", ref.tokenIDsSHA256, got.tokenIDsSHA256),
            ("token_policy", ref.tokenPolicy, got.tokenPolicy),
            ("top_k_tie_break", ref.topKTieBreak, got.topKTieBreak),
        ]
        for (name, left, right) in commonStrings where left != right { errors.append("metadata \(name) differs") }
        let commonIntegers: [(String, Int, Int)] = [
            ("corpus_token_count", ref.corpusTokenCount, got.corpusTokenCount),
            ("positions", ref.positions, got.positions),
            ("top_k", ref.topK, got.topK),
            ("vocab_size", ref.vocabSize, got.vocabSize),
        ]
        for (name, left, right) in commonIntegers where left != right { errors.append("metadata \(name) differs") }
        if ref.exactLMHead != got.exactLMHead || ref.useFlashHead != got.useFlashHead {
            errors.append("metadata head policy differs")
        }
        validatePinned(ref, label: "reference", errors: &errors)
        validatePinned(got, label: "candidate", errors: &errors)
        if ref.engineVersion != MapleParityPins.oracleEngineVersion { errors.append("reference engine version is not pinned") }
        if ref.engineSourceRevision != MapleParityPins.oracleSourceRevision { errors.append("reference source revision is not pinned") }
        if ref.engineSourceDirty { errors.append("reference source checkout is dirty") }
        let referencePinnedRuntime = ref.runtimeVersions.filter { $0.key != "harness" }
        if referencePinnedRuntime != MapleParityPins.oracleRuntimeVersions
            || Set(ref.runtimeVersions.keys) != Set(MapleParityPins.oracleRuntimeVersions.keys).union(["harness"])
            || !MapleParityPreflight.isLowerHex(ref.runtimeVersions["harness"] ?? "", count: 40) {
            errors.append("reference runtime versions are not pinned")
        }
        if ref.expertCacheSlots != 0 || ref.integrityPolicy != MapleParityPins.oracleIntegrityPolicy { errors.append("reference snapshot policy is not pinned") }
        if ref.logitDType != MapleParityPins.oracleLogitDType { errors.append("reference logit dtype is not pinned") }
        if got.engineVersion.isEmpty || !MapleParityPreflight.isLowerHex(got.engineSourceRevision, count: 40) {
            errors.append("candidate engine provenance is invalid")
        }
        if got.engineSourceDirty { errors.append("candidate source checkout is dirty") }
        if got.expertCacheSlots != MapleParityPins.expertCacheSlots || got.integrityPolicy != "full-sha256" {
            errors.append("candidate integrity or cache policy is not pinned")
        }
        if got.logitDType != MapleParityPins.candidateLogitDType { errors.append("candidate logit dtype is not pinned") }
        if Set(got.runtimeVersions.keys) != Set(["architecture", "harness", "macos", "swift"])
            || got.runtimeVersions["architecture"] != "arm64"
            || got.runtimeVersions["harness"] != got.engineSourceRevision
            || got.runtimeVersions["macos"]?.isEmpty != false
            || got.runtimeVersions["swift"]?.isEmpty != false {
            errors.append("candidate runtime versions are invalid")
        }
        if ref.runtimeVersions["harness"] != got.engineSourceRevision {
            errors.append("reference and candidate harness revisions differ")
        }
        for (label, metadata, trace) in [("reference", ref, reference), ("candidate", got, candidate)] {
            if trace.positions.count != metadata.positions || trace.summary.positions != metadata.positions {
                errors.append("\(label) trace count does not match metadata")
            }
            if metadata.command.isEmpty { errors.append("\(label) command is missing") }
            if !MapleParityPreflight.isLowerHex(metadata.engineBinarySHA256, count: 64) {
                errors.append("\(label) binary hash is invalid")
            }
            if trace.summary.exitCode != 0 { errors.append("\(label) exporter did not exit successfully") }
        }
        return errors
    }

    private static func validatePinned(_ metadata: MapleParityMetadata,
                                       label: String,
                                       errors: inout [String]) {
        if metadata.schema != MapleParityPins.schema { errors.append("\(label) schema is not pinned") }
        if metadata.modelID != MapleParityPins.source.modelID { errors.append("\(label) model_id is not pinned") }
        if metadata.sourceRepoID != MapleParityPins.source.repoID { errors.append("\(label) source_repo_id is not pinned") }
        if metadata.modelRevision != MapleParityPins.modelRevision { errors.append("\(label) model_revision is not pinned") }
        if metadata.sourceIndexSHA256 != MapleParityPins.sourceIndexSHA256 { errors.append("\(label) source_index_sha256 is not pinned") }
        if metadata.sourceSnapshotManifestSHA256 != MapleParityPins.sourceSnapshotManifestSHA256 { errors.append("\(label) snapshot manifest is not pinned") }
        if metadata.configSHA256 != MapleParityPins.configSHA256 { errors.append("\(label) config is not pinned") }
        if metadata.tokenizerSHA256 != MapleParityPins.tokenizerSHA256 { errors.append("\(label) tokenizer is not pinned") }
        if metadata.corpusSHA256 != MapleParityPins.corpusSHA256 || metadata.corpusSourceURL != MapleParityPins.corpusSourceURL {
            errors.append("\(label) corpus is not pinned")
        }
        if metadata.corpusTokenCount != MapleParityPins.expectedPositions || metadata.positions != MapleParityPins.expectedPositions {
            errors.append("\(label) is not a full \(MapleParityPins.expectedPositions)-position trace")
        }
        if metadata.tokenIDsSHA256 != MapleParityPins.tokenIDsSHA256 { errors.append("\(label) token IDs are not pinned") }
        if metadata.tokenPolicy != MapleParityPins.tokenPolicy || metadata.topKTieBreak != MapleParityPins.topKTieBreak {
            errors.append("\(label) token or tie policy is not pinned")
        }
        if metadata.topK != MapleParityPins.topK || metadata.vocabSize != MapleParityPins.vocabularySize {
            errors.append("\(label) top-k or vocabulary size is not pinned")
        }
        if !metadata.exactLMHead || metadata.useFlashHead || !metadata.acceptanceEligible {
            errors.append("\(label) is not eligible for exact acceptance")
        }
    }
}

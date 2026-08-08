import Foundation
import Testing

@testable import MferenceMapleParityCore

@Suite struct MapleParityComparisonPreflightTests {
    @Test func comparatorReportsMatchingPrevalidatedContentWithNonFullMetadata() {
        var oracleRuntime = MapleParityPins.oracleRuntimeVersions
        oracleRuntime["harness"] = String(repeating: "c", count: 40)
        let reference = testTrace(metadata: testMetadata(
            engine: "mlx",
            engineVersion: MapleParityPins.oracleEngineVersion,
            engineSourceRevision: MapleParityPins.oracleSourceRevision,
            runtimeVersions: oracleRuntime,
            logitDType: MapleParityPins.oracleLogitDType,
            expertCacheSlots: 0,
            integrityPolicy: MapleParityPins.oracleIntegrityPolicy))
        let candidate = testTrace()
        let comparison = MapleParityComparator.compare(reference: reference, candidate: candidate)
        #expect(comparison.firstDivergence == nil)
        #expect(comparison.positions == 2)
        #expect(comparison.top1Matches == 2)
        #expect(comparison.orderedMatches == 2)
        #expect(comparison.metadataErrors.contains("reference is not a full 1639-position trace"))
        #expect(!comparison.accepted)
    }

    @Test func comparatorReportsFirstContentDivergence() {
        let reference = testTrace(metadata: oracleMetadata())
        let token = testTrace(metadata: acceptanceMetadata(), positions: [testPosition(0, inputID: 7, targetID: 2), reference.positions[1]])
        let ordered = testTrace(metadata: acceptanceMetadata(), positions: [testPosition(0, inputID: 1, targetID: 2, top: [
            MapleParityTopLogit(id: 4, logit: 1), MapleParityTopLogit(id: 3, logit: 0.5),
        ]), reference.positions[1]])
        let logits = testTrace(metadata: acceptanceMetadata(), positions: [testPosition(0, inputID: 1, targetID: 2, top: [
            MapleParityTopLogit(id: 3, logit: 2), MapleParityTopLogit(id: 4, logit: 0.5),
        ]), reference.positions[1]])
        let short = testTrace(metadata: acceptanceMetadata(), positions: [reference.positions[0]])
        #expect(MapleParityComparator.compare(reference: reference, candidate: token).firstDivergence?.reason == "token sequence")
        #expect(MapleParityComparator.compare(reference: reference, candidate: ordered).firstDivergence?.reason == "ordered top-k IDs")
        #expect(MapleParityComparator.compare(reference: reference, candidate: logits).firstDivergence?.reason == "top-k logits")
        #expect(MapleParityComparator.compare(reference: reference, candidate: short).firstDivergence?.reason == "position count")
    }

    @Test func comparatorRejectsMetadataDiagnosticAndProvenanceDivergence() {
        let reference = testTrace(metadata: oracleMetadata())
        let metadata = testTrace(metadata: acceptanceMetadata(modelID: "other"))
        let diagnostic = testTrace(metadata: acceptanceMetadata(acceptanceEligible: false))
        let dirty = testTrace(metadata: acceptanceMetadata(engineSourceDirty: true))
        #expect(MapleParityComparator.compare(reference: reference, candidate: metadata).metadataErrors.contains("metadata model_id differs"))
        #expect(MapleParityComparator.compare(reference: reference, candidate: diagnostic).metadataErrors.contains("candidate is not eligible for exact acceptance"))
        #expect(MapleParityComparator.compare(reference: reference, candidate: dirty).metadataErrors.contains("candidate source checkout is dirty"))
    }

    @Test func preflightAcceptsFactsAndRejectsEachRequiredCondition() {
        #expect(MapleParityPreflight.validate(preflightFacts()).isAccepted)
        let cases: [(String, MapleParityPreflightFacts)] = [
            ("architecture", preflightFacts(appleSilicon: false)),
            ("macOS", preflightFacts(macOSMajor: 14)),
            ("Swift", preflightFacts(swiftMajor: 6, swiftMinor: 0)),
            ("memory", preflightFacts(memoryFreePercent: 9.9)),
            ("disk", preflightFacts(freeBytes: 99)),
            ("process", preflightFacts(conflictingProcesses: ["1 MferenceCLI"])),
            ("model", preflightFacts(modelDirectoryExists: false)),
            ("model validation", preflightFacts(modelValidationError: "invalid Maple install")),
            ("repository", preflightFacts(repositoryDirectory: nil)),
            ("dirty", preflightFacts(checkoutClean: false)),
            ("revision", preflightFacts(gitRevision: "ABC")),
        ]
        for (name, facts) in cases {
            #expect(!MapleParityPreflight.validate(facts).isAccepted, "\(name) must reject export")
        }
    }

    private func oracleMetadata() -> MapleParityMetadata {
        var runtimeVersions = MapleParityPins.oracleRuntimeVersions
        runtimeVersions["harness"] = String(repeating: "c", count: 40)
        return testMetadata(
            engine: "mlx",
            engineVersion: MapleParityPins.oracleEngineVersion,
            engineSourceRevision: MapleParityPins.oracleSourceRevision,
            runtimeVersions: runtimeVersions,
            positions: MapleParityPins.expectedPositions,
            corpusTokenCount: MapleParityPins.expectedPositions,
            tokenIDsSHA256: MapleParityPins.tokenIDsSHA256,
            topK: MapleParityPins.topK,
            vocabSize: MapleParityPins.vocabularySize,
            logitDType: MapleParityPins.oracleLogitDType,
            expertCacheSlots: 0,
            integrityPolicy: MapleParityPins.oracleIntegrityPolicy)
    }

    private func acceptanceMetadata(
        modelID: String = MapleParityPins.source.modelID,
        engineSourceDirty: Bool = false,
        acceptanceEligible: Bool = true
    ) -> MapleParityMetadata {
        testMetadata(
            engineSourceDirty: engineSourceDirty,
            modelID: modelID,
            positions: MapleParityPins.expectedPositions,
            corpusTokenCount: MapleParityPins.expectedPositions,
            tokenIDsSHA256: MapleParityPins.tokenIDsSHA256,
            topK: MapleParityPins.topK,
            vocabSize: MapleParityPins.vocabularySize,
            acceptanceEligible: acceptanceEligible)
    }

    private func preflightFacts(
        appleSilicon: Bool = true,
        macOSMajor: Int = 15,
        swiftMajor: Int? = 6,
        swiftMinor: Int? = 1,
        modelDirectoryExists: Bool = true,
        modelValidationError: String? = nil,
        freeBytes: UInt64? = 100,
        memoryFreePercent: Double? = 10,
        conflictingProcesses: [String] = [],
        checkoutClean: Bool = true,
        repositoryDirectory: URL? = URL(fileURLWithPath: "/repo"),
        gitRevision: String? = String(repeating: "a", count: 40)
    ) -> MapleParityPreflightFacts {
        MapleParityPreflightFacts(
            appleSilicon: appleSilicon,
            macOSMajor: macOSMajor,
            macOSVersion: "15.0.0",
            swiftMajor: swiftMajor,
            swiftMinor: swiftMinor,
            swiftVersion: "6.1.0",
            architecture: appleSilicon ? "arm64" : "x86_64",
            modelDirectoryExists: modelDirectoryExists,
            modelValidationError: modelValidationError,
            freeBytes: freeBytes,
            requiredFreeBytes: 100,
            memoryFreePercent: memoryFreePercent,
            conflictingProcesses: conflictingProcesses,
            checkoutClean: checkoutClean,
            repositoryDirectory: repositoryDirectory,
            gitRevision: gitRevision)
    }
}

import Foundation
import Metal
import Mference

public struct MapleParityExportRequest: Sendable {
    public let modelDirectory: URL
    public let corpusURL: URL
    public let outputURL: URL
    public let repositoryDirectory: URL
    public let command: [String]
    public let executableURL: URL
    /// A positive prefix is diagnostic-only and is marked ineligible for acceptance.
    public let diagnosticMaxPositions: Int?

    public init(modelDirectory: URL,
                corpusURL: URL,
                outputURL: URL,
                repositoryDirectory: URL,
                command: [String] = CommandLine.arguments,
                diagnosticMaxPositions: Int? = nil) {
        let fallbackURL = Bundle.main.executableURL
            ?? URL(fileURLWithPath: CommandLine.arguments.first ?? "MferenceMapleParity")
        let fallback = fallbackURL.path
        let normalizedCommand = command.isEmpty ? [fallback] : command
        self.modelDirectory = modelDirectory
        self.corpusURL = corpusURL
        self.outputURL = outputURL
        self.repositoryDirectory = repositoryDirectory
        self.command = normalizedCommand
        self.executableURL = fallbackURL.standardizedFileURL
        self.diagnosticMaxPositions = diagnosticMaxPositions
    }
}

public struct MapleParityExportResult: Equatable, Sendable {
    public let positions: Int
    public let outputURL: URL
    public let preflight: MapleParityPreflightReport
}

public enum MapleParityExporter {
    public static func export(_ request: MapleParityExportRequest) async throws -> MapleParityExportResult {
        guard !FileManager.default.fileExists(atPath: request.outputURL.path) else {
            throw MapleParityError.io("refusing to overwrite \(request.outputURL.path)")
        }
        let preflight = try MapleParityPreflight.validate(
            MapleParityPreflight.inspect(modelDirectory: request.modelDirectory,
                                         repositoryDirectory: request.repositoryDirectory))
        guard preflight.isAccepted else {
            throw MapleParityError.invalid("preflight failed: \(preflight.failures.joined(separator: "; "))")
        }
        guard let revision = preflight.facts.gitRevision else {
            throw MapleParityError.invalid("preflight did not resolve the checkout revision")
        }
        let corpus = try loadCorpus(request.corpusURL)
        let tokenizer = try await MFTokenizer.load(forModelDirectory: request.modelDirectory)
        let tokens = tokenizer.encode(corpus, addBOS: false)
        guard tokens.count == MapleParityPins.expectedPositions,
              canonicalTokenHash(tokens) == MapleParityPins.tokenIDsSHA256 else {
            throw MapleParityError.invalid("corpus tokenization does not match the pinned Maple sequence")
        }
        let positions = try outputPositions(diagnostic: request.diagnosticMaxPositions)
        let loadStarted = Date()
        let context = try MetalContext()
        let model = try Model.load(directoryURL: request.modelDirectory,
                                   device: context.device,
                                   streamingMode: .pread(slotCount: MapleParityPins.expertCacheSlots),
                                   integrityPolicy: .fullSha256)
        guard model.config.family == .maple,
              model.modelID == MapleParityPins.source.modelID,
              model.sourceSnapshotHash == "sha256:\(MapleParityPins.sourceIndexSHA256)",
              model.config.vocabSize == MapleParityPins.vocabularySize else {
            throw MapleParityError.invalid("installed model is not the pinned Maple checkpoint")
        }
        let configuration = RuntimeConfiguration(expertCacheSlots: MapleParityPins.expertCacheSlots,
                                                 rdadvisePolicy: .off,
                                                 prefillEnabled: false,
                                                 prefillChunkTokens: 32,
                                                 forceLogitsHead: true)
        let runtime = try ForwardRunnerFactory.make(model: model, context: context,
                                                    maxContext: max(positions, 32),
                                                    runtimeConfiguration: configuration)
        guard runtime.producer is MapleForwardRunner else {
            throw MapleParityError.invalid("Maple exporter did not select the Maple forward runner")
        }
        let executableHash = try Sha256Verifier.hashFile(at: request.executableURL)
        let tokenizerHash = try Sha256Verifier.hashFile(
            at: request.modelDirectory.appendingPathComponent("tokenizer/tokenizer.json"))
        guard tokenizerHash == MapleParityPins.tokenizerSHA256 else {
            throw MapleParityError.invalid("installed tokenizer does not match the pinned Maple tokenizer")
        }
        let loadSeconds = Date().timeIntervalSince(loadStarted)
        let metadata = MapleParityMetadata(
            schema: MapleParityPins.schema,
            engine: MapleParityPins.engine,
            engineVersion: "1",
            engineSourceRevision: revision,
            engineSourceDirty: false,
            engineBinarySHA256: executableHash,
            command: request.command,
            runtimeVersions: runtimeVersions(preflight.facts, revision: revision),
            modelID: MapleParityPins.source.modelID,
            sourceRepoID: MapleParityPins.source.repoID,
            modelRevision: MapleParityPins.modelRevision,
            sourceIndexSHA256: MapleParityPins.sourceIndexSHA256,
            sourceSnapshotManifestSHA256: MapleParityPins.sourceSnapshotManifestSHA256,
            configSHA256: MapleParityPins.configSHA256,
            tokenizerSHA256: tokenizerHash,
            corpusSourceURL: MapleParityPins.corpusSourceURL,
            corpusSHA256: MapleParityPins.corpusSHA256,
            corpusTokenCount: tokens.count,
            positions: positions,
            tokenIDsSHA256: canonicalTokenHash(Array(tokens.prefix(positions))),
            tokenPolicy: MapleParityPins.tokenPolicy,
            topK: MapleParityPins.topK,
            topKTieBreak: MapleParityPins.topKTieBreak,
            vocabSize: model.config.vocabSize,
            logitDType: MapleParityPins.candidateLogitDType,
            exactLMHead: true,
            useFlashHead: false,
            expertCacheSlots: MapleParityPins.expertCacheSlots,
            integrityPolicy: "full-sha256",
            acceptanceEligible: positions == MapleParityPins.expectedPositions)
        guard let logits = context.device.makeBuffer(length: model.config.vocabSize * MemoryLayout<Float16>.stride,
                                                      options: .storageModeShared) else {
            throw MapleParityError.io("unable to allocate Maple logits buffer")
        }
        runtime.producer.reset()
        let started = Date()
        var records: [MapleParityPosition] = []
        records.reserveCapacity(positions)
        for position in 0..<positions {
            try Task.checkCancellation()
            try await runtime.producer.produce(token: tokens[position], position: position, into: logits)
            records.append(MapleParityPosition(position: position,
                                               inputID: Int(tokens[position]),
                                               targetID: position + 1 < tokens.count ? Int(tokens[position + 1]) : nil,
                                               top: try topLogits(logits, vocabularySize: model.config.vocabSize)))
        }
        let elapsedSeconds = Date().timeIntervalSince(started)
        try MapleParityTraceWriter.write(
            MapleParityTrace(metadata: metadata, positions: records,
                             summary: MapleParitySummary(
                                positions: records.count,
                                loadSeconds: loadSeconds,
                                elapsedSeconds: elapsedSeconds,
                                positionsPerSecond: elapsedSeconds > 0 ? Double(records.count) / elapsedSeconds : 0,
                                exitCode: 0)),
            to: request.outputURL)
        return MapleParityExportResult(positions: positions, outputURL: request.outputURL, preflight: preflight)
    }

    private static func outputPositions(diagnostic: Int?) throws -> Int {
        guard let diagnostic else { return MapleParityPins.expectedPositions }
        guard diagnostic > 0, diagnostic < MapleParityPins.expectedPositions else {
            throw MapleParityError.invalid("--max-positions is diagnostic-only and must be 1..<\(MapleParityPins.expectedPositions)")
        }
        return diagnostic
    }

    private static func loadCorpus(_ url: URL) throws -> String {
        let data: Data
        do { data = try Data(contentsOf: url) }
        catch { throw MapleParityError.io("unable to read corpus \(url.path): \(error.localizedDescription)") }
        guard Sha256Verifier.hashData(data) == MapleParityPins.corpusSHA256,
              let text = String(data: data, encoding: .utf8),
              !text.contains("\r"), text == text.precomposedStringWithCanonicalMapping else {
            throw MapleParityError.invalid("corpus does not match the pinned raw UTF-8/LF/NFC input")
        }
        return text
    }

    private static func canonicalTokenHash(_ tokens: [Int32]) -> String {
        let encoded = "[" + tokens.map(String.init).joined(separator: ",") + "]"
        return Sha256Verifier.hashData(Data(encoded.utf8))
    }

    private static func topLogits(_ buffer: MTLBuffer, vocabularySize: Int) throws -> [MapleParityTopLogit] {
        let values = buffer.contents().bindMemory(to: Float16.self, capacity: vocabularySize)
        return try MapleParityTopK.select(vocabularySize: vocabularySize, topK: MapleParityPins.topK) {
            Float(values[$0])
        }
    }

    private static func runtimeVersions(_ facts: MapleParityPreflightFacts,
                                        revision: String) -> [String: String] {
        [
            "architecture": facts.architecture,
            "harness": revision,
            "macos": facts.macOSVersion,
            "swift": facts.swiftVersion ?? "unknown",
        ]
    }
}

enum MapleParityTopK {
    static func select(vocabularySize: Int,
                       topK: Int,
                       logitAt: (Int) -> Float) throws -> [MapleParityTopLogit] {
        guard vocabularySize > 0, topK > 0, topK <= vocabularySize else {
            throw MapleParityError.invalid("invalid top-k dimensions")
        }
        var top: [MapleParityTopLogit] = []
        top.reserveCapacity(topK)
        for id in 0..<vocabularySize {
            let logit = logitAt(id)
            guard logit.isFinite else {
                throw MapleParityError.invalid("non-finite vocabulary logit at token \(id)")
            }
            let candidate = MapleParityTopLogit(id: id, logit: logit)
            let insertion = top.firstIndex { isBetter(candidate, than: $0) } ?? top.endIndex
            guard insertion < topK else { continue }
            top.insert(candidate, at: insertion)
            if top.count > topK { top.removeLast() }
        }
        return top
    }

    private static func isBetter(_ left: MapleParityTopLogit, than right: MapleParityTopLogit) -> Bool {
        left.logit == right.logit ? left.id < right.id : left.logit > right.logit
    }

}

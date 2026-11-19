import Foundation
@testable import MferenceAppCore
import Testing

/// Records Vision receipt accuracy on the synthetic fixture set (mockup note
/// 7, checkpoint C-B1). Gated: set `MFERENCE_SCAN_ACCURACY_DIR` to the output
/// directory. With `MFERENCE_SCAN_ACCURACY_APPLE_MODEL=1` it also asks Apple's
/// on-device model to fill the fields, when Apple Intelligence is on.
@Suite struct AppleScanAccuracyTests {
    static let outputDirectory = ProcessInfo.processInfo.environment["MFERENCE_SCAN_ACCURACY_DIR"]

    struct FieldResult: Codable {
        var expected: String?
        var found: String?
        var correct: Bool
        var confidence: String?
    }

    struct ReceiptResult: Codable {
        var name: String
        var seconds: Double
        var filledBy: String?
        var fields: [String: FieldResult]
        /// The model's whole answer, kept so a wrong amount can be traced to
        /// the model or to the merge (the fixtures are invented data).
        var modelAnswer: String?
    }

    /// Keeps the last answer of the model it wraps.
    final class AnswerRecorder: AssistantTextModel, @unchecked Sendable {
        let base: any AssistantTextModel
        private let lock = NSLock()
        private var answer: String?
        init(_ base: any AssistantTextModel) { self.base = base }
        var kind: AssistantModelKind { base.kind }
        func availability() async -> AssistantAvailability { await base.availability() }
        func complete(_ request: AssistantRequest) async throws -> String {
            let text = try await base.complete(request)
            lock.withLock { answer = text }
            return text
        }
        func take() -> String? { lock.withLock { defer { answer = nil }; return answer } }
    }

    @Test(.enabled(if: outputDirectory != nil))
    func recordReceiptAccuracy() async throws {
        let output = URL(filePath: try #require(Self.outputDirectory), directoryHint: .isDirectory)
        let images = output.appending(path: "images", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: images, withIntermediateDirectories: true)
        let useAppleModel = ProcessInfo.processInfo.environment["MFERENCE_SCAN_ACCURACY_APPLE_MODEL"] == "1"
        let recorder = AnswerRecorder(FoundationTextModel())
        let assistant = AppleModelService(primary: recorder, fallback: nil)
        let appleAvailability = await FoundationTextModel().availability()

        var results: [ReceiptResult] = []
        for fixture in ReceiptFixtures.all {
            let url = try fixture.write(to: images)
            let start = ContinuousClock.now
            let reading = try await VisionScanner().recognize(imageAt: url)
            let printed = ReceiptHeuristics.read(reading)
            let dump = reading.lines.map {
                String(format: "%@ | x %.3f y %.3f w %.3f h %.3f | slope %.4f", $0.text, $0.box.minX, $0.box.minY,
                       $0.box.width, $0.box.height, $0.slope)
            } + ["", "rows:"] + reading.rows
            try dump.joined(separator: "\n").write(to: images.appending(path: "\(fixture.name).lines.txt"),
                                                   atomically: true, encoding: .utf8)
            var model: ReceiptDraft?
            var filledBy: AssistantModelKind?
            if useAppleModel, appleAvailability.isAvailable,
               let result = try? await assistant.fillReceipt(from: reading.modelText) {
                model = result.value
                filledBy = result.model
            }
            let fields = ReceiptFieldMerger.merge(model: model, text: printed, reading: reading, filledBy: filledBy)
            let elapsed = ContinuousClock.now - start
            let truth: [ScanFieldKey: String?] = [
                .merchant: fixture.merchant, .date: fixture.date, .total: ScanCSVExporter.plain(fixture.total),
                .tax: fixture.tax.map(ScanCSVExporter.plain), .paidWith: fixture.paidWith,
            ]
            var fieldResults: [String: FieldResult] = [:]
            for (key, expected) in truth {
                let found = Self.normalized(key, fields.value(key))
                fieldResults[key.rawValue] = FieldResult(
                    expected: expected, found: found,
                    correct: (expected?.lowercased() ?? "") == (found?.lowercased() ?? ""),
                    confidence: fields[key]?.confidence.rawValue)
            }
            results.append(ReceiptResult(name: fixture.name,
                                         seconds: Double(elapsed.components.attoseconds) / 1e18
                                             + Double(elapsed.components.seconds),
                                         filledBy: filledBy?.rawValue, fields: fieldResults,
                                         modelAnswer: recorder.take()))
        }

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(results).write(to: output.appending(path: "accuracy.json"))
        try Self.markdown(results, appleAvailability: appleAvailability, usedAppleModel: useAppleModel)
            .write(to: output.appending(path: "accuracy.md"), atomically: true, encoding: .utf8)
    }

    static func normalized(_ key: ScanFieldKey, _ value: String?) -> String? {
        guard let value else { return nil }
        switch key {
        case .date: return ScanFormat.isoDate(fromDisplay: value)
        case .total, .tax: return ReceiptValueParser.decimal(value).map(ScanCSVExporter.plain)
        default: return value
        }
    }

    static func markdown(_ results: [ReceiptResult], appleAvailability: AssistantAvailability,
                         usedAppleModel: Bool) -> String {
        let keys = [ScanFieldKey.merchant, .date, .total, .tax, .paidWith]
        var lines = ["# B1 Scan: receipt accuracy (synthetic fixtures)", "",
                     "Generated \(Date.now.formatted(.iso8601)) on this Mac by `AppleScanAccuracyTests`.",
                     "Fixtures: \(results.count) receipts rendered from invented data "
                         + "(`Tests/MferenceAppCoreTests/Fixtures/Apple/ReceiptFixtures.swift`), "
                         + "monospaced, tilts up to 3 degrees. Images in `images/`.",
                     "OCR: Apple Vision `RecognizeDocumentsRequest`. Fields: printed-text reading"
                         + (usedAppleModel ? " merged with Apple's on-device model" : "")
                         + ". Apple model availability: \(appleAvailability).", "",
                     "| Field | Correct | Of | Accuracy | Marked Sure | Sure and wrong |",
                     "| --- | --- | --- | --- | --- | --- |"]
        for key in keys {
            let rows = results.compactMap { $0.fields[key.rawValue] }
            let correct = rows.filter(\.correct).count
            let sure = rows.filter { $0.confidence == "sure" }.count
            let sureWrong = rows.filter { $0.confidence == "sure" && !$0.correct }.count
            lines.append("| \(key.label) | \(correct) | \(rows.count) | "
                + String(format: "%.0f%%", Double(correct) / Double(max(rows.count, 1)) * 100)
                + " | \(sure) | \(sureWrong) |")
        }
        lines += ["", "| Receipt | Seconds | Wrong fields |", "| --- | --- | --- |"]
        for result in results {
            let wrong = result.fields.filter { !$0.value.correct }
                .map { "\($0.key): got \($0.value.found ?? "nil"), want \($0.value.expected ?? "nil")" }
                .sorted().joined(separator: "; ")
            lines.append("| \(result.name) | \(String(format: "%.2f", result.seconds)) | \(wrong.isEmpty ? "none" : wrong) |")
        }
        return lines.joined(separator: "\n") + "\n"
    }
}

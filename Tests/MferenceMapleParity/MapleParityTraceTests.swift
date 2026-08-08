import Foundation
import Testing

@testable import MferenceMapleParityCore

@Suite struct MapleParityTraceTests {
    @Test func topKOrdersLogitsAndBreaksTiesByTokenID() throws {
        let values: [Float] = [0.5, 2, 2, -1, 1]
        #expect(try MapleParityTopK.select(vocabularySize: values.count, topK: 3) { values[$0] } == [
            MapleParityTopLogit(id: 1, logit: 2),
            MapleParityTopLogit(id: 2, logit: 2),
            MapleParityTopLogit(id: 4, logit: 1),
        ])
    }

    @Test func topKRejectsInvalidDimensionsAndNonfiniteLogits() {
        for dimensions in [(0, 1), (2, 0), (2, 3)] {
            #expect(throws: MapleParityError.self) {
                try MapleParityTopK.select(vocabularySize: dimensions.0, topK: dimensions.1) { _ in 0 }
            }
        }
        #expect(throws: MapleParityError.self) {
            try MapleParityTopK.select(vocabularySize: 3, topK: 1) { $0 == 1 ? .nan : 0 }
        }
    }

    @Test func traceRoundTripsStrictlyAndDoesNotOverwrite() throws {
        let url = temporaryTraceURL()
        defer { try? FileManager.default.removeItem(at: url) }
        let trace = testTrace()
        try MapleParityTraceWriter.write(trace, to: url)
        #expect(try MapleParityTraceReader.read(from: url) == trace)
        #expect(throws: MapleParityError.self) { try MapleParityTraceWriter.write(trace, to: url) }
    }

    @Test func readerConsumesCanonicalPythonDiagnosticShape() throws {
        let url = try #require(Bundle.module.url(
            forResource: "OracleDiagnostic", withExtension: "jsonl", subdirectory: "Fixtures"))
        let trace = try MapleParityTraceReader.read(from: url)
        #expect(trace.metadata.engine == "mlx")
        #expect(trace.metadata.runtimeVersions == [
            "harness": String(repeating: "c", count: 40),
            "mlx": "0.32.0", "mlx-lm": "0.31.3", "numpy": "2.5.1",
            "python": "3.12.13", "tokenizers": "0.22.2", "transformers": "5.14.1",
        ])
        #expect(trace.positions.count == 2)
        #expect(trace.positions.last?.targetID == nil)
    }

    @Test func readerRejectsMalformedContractRecords() throws {
        let lines = try traceLines(testTrace())
        let extraKey = try replacingObject(lines[0]) { $0["extra"] = true }
        let wrongCount = try replacingObject(lines[0]) { $0["positions"] = 3 }
        let noncanonical = try replacingObject(lines[1]) { object in
            object["top"] = [
                ["id": 4, "logit": 0.5], ["id": 3, "logit": 1.0],
            ]
        }
        let duplicate = try replacingObject(lines[1]) { object in
            object["top"] = [
                ["id": 3, "logit": 1.0], ["id": 3, "logit": 0.5],
            ]
        }
        let outOfRange = try replacingObject(lines[1]) { $0["input_id"] = 8 }
        let badHash = try replacingObject(lines[0]) { $0["token_ids_sha256"] = String(repeating: "a", count: 64) }
        let nonfinite = lines[1].replacingOccurrences(of: "\"logit\":1", with: "\"logit\":1e999")
        let cases: [(String, [String])] = [
            ("exact keys", [extraKey] + Array(lines.dropFirst())),
            ("record order", [lines[1], lines[0]] + Array(lines.dropFirst(2))),
            ("count", [wrongCount] + Array(lines.dropFirst())),
            ("noncanonical", [lines[0], noncanonical] + Array(lines.dropFirst(2))),
            ("duplicate", [lines[0], duplicate] + Array(lines.dropFirst(2))),
            ("out of range", [lines[0], outOfRange] + Array(lines.dropFirst(2))),
            ("nonfinite", [lines[0], nonfinite] + Array(lines.dropFirst(2))),
            ("token hash", [badHash] + Array(lines.dropFirst())),
        ]
        for (name, malformed) in cases {
            let url = try writeRawTrace(malformed)
            defer { try? FileManager.default.removeItem(at: url) }
            #expect(throws: MapleParityError.self, "\(name) must be rejected") {
                try MapleParityTraceReader.read(from: url)
            }
        }
    }
}

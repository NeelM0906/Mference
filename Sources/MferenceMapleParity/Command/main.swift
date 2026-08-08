import Foundation
import Darwin
import MferenceMapleParityCore

private let usage = """
Usage:
  MferenceMapleParity preflight --model <maple.gturbo> --repository <checkout>
  MferenceMapleParity export --model <maple.gturbo> --corpus <raven.txt> --output <trace.jsonl> --repository <checkout> [--max-positions <n>]
  MferenceMapleParity compare <mlx-trace.jsonl> <mference-trace.jsonl>
"""

@main
struct MferenceMapleParityCommand {
    static func main() async {
        do {
            let code = try await run(Array(CommandLine.arguments.dropFirst()))
            exit(code)
        } catch {
            FileHandle.standardError.write(Data("error: \(error)\n\n\(usage)\n".utf8))
            exit(2)
        }
    }

    private static func run(_ arguments: [String]) async throws -> Int32 {
        guard let command = arguments.first else { throw MapleParityError.invalid("missing command") }
        switch command {
        case "preflight":
            let options = try parseOptions(Array(arguments.dropFirst()), allowed: ["--model", "--repository"])
            let model = try required("--model", from: options)
            let repository = try required("--repository", from: options)
            let report = try MapleParityPreflight.validate(
                MapleParityPreflight.inspect(modelDirectory: URL(fileURLWithPath: model),
                                             repositoryDirectory: URL(fileURLWithPath: repository)))
            print(report.failures.isEmpty ? "preflight passed" : "preflight failed: \(report.failures.joined(separator: "; "))")
            return report.isAccepted ? 0 : 1
        case "export":
            let tail = Array(arguments.dropFirst())
            let options = try parseOptions(tail, allowed: ["--model", "--corpus", "--output", "--repository", "--max-positions"])
            let model = try required("--model", from: options)
            let corpus = try required("--corpus", from: options)
            let output = try required("--output", from: options)
            let repository = try required("--repository", from: options)
            let maximum = try options["--max-positions"].map { value in
                guard let parsed = Int(value) else { throw MapleParityError.invalid("--max-positions must be an integer") }
                return parsed
            }
            let result = try await MapleParityExporter.export(MapleParityExportRequest(
                modelDirectory: URL(fileURLWithPath: model),
                corpusURL: URL(fileURLWithPath: corpus),
                outputURL: URL(fileURLWithPath: output),
                repositoryDirectory: URL(fileURLWithPath: repository),
                diagnosticMaxPositions: maximum))
            print("exported \(result.positions) positions to \(result.outputURL.path)")
            return 0
        case "compare":
            guard arguments.count == 3 else { throw MapleParityError.invalid("compare requires reference and candidate traces") }
            let reference = try MapleParityTraceReader.read(from: URL(fileURLWithPath: arguments[1]))
            let candidate = try MapleParityTraceReader.read(from: URL(fileURLWithPath: arguments[2]))
            let result = MapleParityComparator.compare(reference: reference, candidate: candidate)
            print("positions=\(result.positions) top1=\(result.top1Matches)/\(result.positions) ordered=\(result.orderedMatches)/\(result.positions)")
            for error in result.metadataErrors { print("metadata divergence: \(error)") }
            if let divergence = result.firstDivergence {
                print("first divergence at position \(divergence.position): \(divergence.reason)")
            } else {
                print("no exact trace divergence")
            }
            return result.accepted ? 0 : 1
        case "--help", "help":
            print(usage)
            return 0
        default:
            throw MapleParityError.invalid("unknown command \(command)")
        }
    }

    private static func required(_ name: String, from options: [String: String]) throws -> String {
        guard let value = options[name] else {
            throw MapleParityError.invalid("missing \(name)")
        }
        return value
    }

    private static func parseOptions(_ arguments: [String], allowed: Set<String>) throws -> [String: String] {
        var options: [String: String] = [:]
        var index = 0
        while index < arguments.count {
            guard allowed.contains(arguments[index]), index + 1 < arguments.count else {
                throw MapleParityError.invalid("unknown or incomplete argument \(arguments[index])")
            }
            guard options[arguments[index]] == nil else {
                throw MapleParityError.invalid("duplicate argument \(arguments[index])")
            }
            options[arguments[index]] = arguments[index + 1]
            index += 2
        }
        return options
    }
}

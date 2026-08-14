import Foundation
import MferenceRepackCore

private let usage = """
Usage:
  MferenceRepack [--dry-run] [--model <gemma4|qwen36|qwen38|deepseekv4flash|inklingsmall|maple>] --output <model.gturbo> [--overwrite] [--resume] [--base-url <url>]
  MferenceRepack --attach-mtp <mtp-shard.safetensors> --output <model.gturbo>
  MferenceRepack --discard-partial --output <model.gturbo>
  MferenceRepack --verify-install --input-gturbo <model.gturbo>
  MferenceRepack --help

--attach-mtp appends the Qwen 3.8 MTP draft-layer tensors from a local BF16
safetensors shard to an existing install (projections quantized to INT4
affine group-64), enabling speculative decoding.

The installer streams the selected checkpoint (default: the supported Gemma 4
checkpoint) from Hugging Face and repackages it without materializing the
source checkpoint on disk. Set HF_TOKEN only if Hugging Face requests
authentication. A cancelled or interrupted download can be continued with
--resume or removed with --discard-partial.
"""

private struct Arguments {
    var model = SupportedModelSource.default
    var output: String?
    var overwrite = false
    var resume = false
    var discardPartial = false
    var verifyInstall = false
    var attachMTP: String?
    var inputGTurbo: String?
    var baseURL: URL?
    var dryRun = false

    static func parse(_ values: [String]) throws -> Arguments {
        var parsed = Arguments()
        var index = 1
        while index < values.count {
            let flag = values[index]
            switch flag {
            case "--help":
                throw ParseError.help
            case "--overwrite":
                parsed.overwrite = true
                index += 1
            case "--resume":
                parsed.resume = true
                index += 1
            case "--dry-run":
                parsed.dryRun = true
                index += 1
            case "--discard-partial":
                parsed.discardPartial = true
                index += 1
            case "--verify-install":
                parsed.verifyInstall = true
                index += 1
            case "--attach-mtp":
                guard index + 1 < values.count else {
                    throw ParseError.missingValue(flag)
                }
                parsed.attachMTP = values[index + 1]
                index += 2
            case "--model":
                guard index + 1 < values.count else {
                    throw ParseError.missingValue(flag)
                }
                guard let source = SupportedModelSource.named(values[index + 1]) else {
                    throw ParseError.invalidMode(
                        "unknown model \"\(values[index + 1])\"; supported: "
                        + SupportedModelSource.all.map(\.name).joined(separator: ", "))
                }
                parsed.model = source
                index += 2
            case "--base-url":
                guard index + 1 < values.count else {
                    throw ParseError.missingValue(flag)
                }
                guard let url = URL(string: values[index + 1]),
                      url.scheme == "http" || url.scheme == "https" else {
                    throw ParseError.invalidMode(
                        "--base-url must be an http(s) URL")
                }
                parsed.baseURL = url
                index += 2
            case "--output", "--input-gturbo":
                guard index + 1 < values.count else {
                    throw ParseError.missingValue(flag)
                }
                if flag == "--output" {
                    parsed.output = values[index + 1]
                } else {
                    parsed.inputGTurbo = values[index + 1]
                }
                index += 2
            default:
                throw ParseError.unknown(flag)
            }
        }

        guard !(parsed.resume && parsed.discardPartial) else {
            throw ParseError.invalidMode("--resume and --discard-partial are mutually exclusive")
        }
        if parsed.discardPartial {
            guard parsed.output != nil else {
                throw ParseError.missingRequired("--output")
            }
            guard parsed.inputGTurbo == nil, !parsed.overwrite, !parsed.verifyInstall else {
                throw ParseError.invalidMode("--discard-partial only accepts --output")
            }
            return parsed
        }
        if parsed.attachMTP != nil {
            guard parsed.output != nil else {
                throw ParseError.missingRequired("--output")
            }
            guard parsed.inputGTurbo == nil, !parsed.overwrite, !parsed.resume,
                  !parsed.verifyInstall, !parsed.dryRun else {
                throw ParseError.invalidMode("--attach-mtp only accepts --output")
            }
            return parsed
        }
        if parsed.verifyInstall {
            guard parsed.inputGTurbo != nil else {
                throw ParseError.missingRequired("--input-gturbo")
            }
            guard parsed.output == nil, !parsed.overwrite, !parsed.resume else {
                throw ParseError.invalidMode("verification accepts only --input-gturbo")
            }
        } else {
            guard parsed.output != nil else {
                throw ParseError.missingRequired("--output")
            }
            guard parsed.inputGTurbo == nil else {
                throw ParseError.invalidMode("--input-gturbo requires --verify-install")
            }
        }
        return parsed
    }
}

private enum ParseError: Error, CustomStringConvertible {
    case help
    case unknown(String)
    case missingValue(String)
    case missingRequired(String)
    case invalidMode(String)

    var description: String {
        switch self {
        case .help: return "help"
        case .unknown(let flag): return "unknown argument: \(flag)"
        case .missingValue(let flag): return "missing value for \(flag)"
        case .missingRequired(let flag): return "missing required argument: \(flag)"
        case .invalidMode(let message): return message
        }
    }
}

private func printError(_ message: String) {
    FileHandle.standardError.write(Data((message + "\n").utf8))
}

private func run(_ values: [String]) async -> Int32 {
    let arguments: Arguments
    do {
        arguments = try Arguments.parse(values)
    } catch ParseError.help {
        print(usage)
        return 0
    } catch {
        printError("error: \(error)\n\n\(usage)")
        return 2
    }

    if arguments.discardPartial, let output = arguments.output {
        do {
            try RemoteStreamingRepacker.discardPartial(outputDirectory: output)
            print("Discarded saved download for \(output)")
            return 0
        } catch {
            printError("discard failed: \(error)")
            return 1
        }
    }

    if let shard = arguments.attachMTP, let output = arguments.output {
        do {
            let result = try MTPAttachTool.run(gturboDirectory: output,
                                               shardPath: shard)
            print("Attached \(result.tensorCount) MTP tensors to \(output)")
            print("Appended bytes: \(result.appendedBytes)")
            print("model_weights.bin: \(result.weightsFileBytes) bytes"
                + (result.rewroteWeightsFile ? " (rewritten with grown index)" : " (in-place)"))
            return 0
        } catch {
            printError("attach-mtp failed: \(error)")
            return 1
        }
    }

    if arguments.verifyInstall, let input = arguments.inputGTurbo {
        do {
            let result = try VerifiedInstallTool.run(
                options: VerifyInstallOptions(inputGTurbo: input))
            print("Verified \(result.fileCount) files (\(result.bytesVerified) bytes)")
            print("Receipt: \(result.receiptPath)")
            return 0
        } catch {
            printError("verification failed: \(error)")
            return 1
        }
    }

    guard let output = arguments.output else { return 2 }
    let source = arguments.model
    let options = source.installOptions(
        outputDirectory: URL(fileURLWithPath: output),
        overwrite: arguments.overwrite,
        token: ProcessInfo.processInfo.environment["HF_TOKEN"],
        resume: arguments.resume,
        baseURL: arguments.baseURL,
        dryRunSpaceCheck: arguments.dryRun)
    do {
        let result = try await RemoteStreamingRepacker(options: options).run()
        if result.dryRun {
            print("Dry run for \(source.displayName)")
            print("Source revision: \(result.resolvedCommit)")
            print("Range requests: \(result.rangeRequestCount)")
            print("Source bytes to read: \(result.remoteBytesToDownload)")
            print("Output bytes: \(result.outputBytes)")
            print("Resident entries: \(result.residentEntryCount)")
            print("Expert layers: \(result.expertLayerCount)")
            print("Excluded multimodal tensors: "
                + "\(result.excludedMultimodalTensorCount)")
            return 0
        }
        print("Installed \(source.displayName)")
        print("Source revision: \(result.resolvedCommit)")
        print("Model: \(result.outputDir)")
        return 0
    } catch {
        printError("install failed: \(error)")
        return 1
    }
}

exit(await run(CommandLine.arguments))

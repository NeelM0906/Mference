import Foundation
import MferenceRepackCore

private let usage = """
Usage:
  MferenceRepack [--dry-run] [--model <qwen36|qwen36original|qwen38|maple|minicpm5|minicpm5mlx|minicpm5-1b|inkling2mini>] --output <model.gturbo> [--overwrite] [--resume] [--skip-mtp] [--base-url <url>] [--local-gguf <file>]
  MferenceRepack --attach-mtp <mtp-shard.safetensors> --output <model.gturbo>
  MferenceRepack --discard-partial --output <model.gturbo>
  MferenceRepack --verify-install --input-gturbo <model.gturbo>
  MferenceRepack --verify-gguf-copy --input-gturbo <model.gturbo> --local-gguf <file>
  MferenceRepack --help

--attach-mtp appends the Qwen 3.8 MTP draft-layer tensors from a local BF16
safetensors shard to an existing install (projections quantized to INT4
affine group-64), enabling speculative decoding.

--skip-mtp drops the source checkpoint's own mtp.* draft-layer group instead
of carrying it. Vision towers are always skipped. Either way the decision is
recorded in manifest.json -> sidecars rather than left implicit.

qwen36original reads the model vendor's original BF16 repo and quantizes to
INT4 affine group-64 during the install. It is the same checkpoint qwen36
installs from mlx-community's pre-quantized conversion, put through our own
quantizer instead; installing both and comparing them is the W2.1b
quantizer-quality gate.

minicpm5 reads openbmb/MiniCPM5-2B's BF16 upload and quantizes to INT4 affine
group-64 during the install (~5 GB read, ~1.4 GB written); minicpm5mlx is the
vendor's own MLX INT4 conversion of the same checkpoint, the W2.1b control.
minicpm5-1b installs openbmb/MiniCPM5-1B-MLX, the vendor's MLX INT4 conversion
of MiniCPM5-1B (~0.6 GB read and written, quantized bytes copied unchanged).
inkling2mini installs Unsloth's UD-Q4_K_XL GGUF of Thinking Machines Inkling 2
Mini (~17.6 GB read and written): every tensor is copied unchanged, nothing is
re-quantized. --local-gguf reads a local copy of that pinned file instead of the
Hub; it must match the pin's size and SHA-256, and the install is the same. The
path must name the file itself, not a symbolic link (a Hugging Face cache
snapshot entry is one: pass the blob it points to).
--verify-gguf-copy compares every tensor of a GGUF install with the GGUF it was
copied from, byte for byte.

The installer streams the selected checkpoint (default: the supported Qwen 3.6
checkpoint) from Hugging Face and repackages it without materializing the
source checkpoint on disk. Set HF_TOKEN only if Hugging Face requests
authentication. A cancelled or interrupted download can be continued with
--resume or removed with --discard-partial.
"""

/// Range callbacks may arrive from transfer workers. Throttle payload-only
/// updates while always reporting phase transitions and completed payloads.
private final class InstallProgressPrinter: @unchecked Sendable {
    private let lock = NSLock()
    private var lastPayloadTime = -Double.infinity

    func report(_ progress: ModelInstallProgress) {
        lock.lock()
        defer { lock.unlock() }
        if case let .copyingPayload(reused, downloaded, total) = progress {
            let now = ProcessInfo.processInfo.systemUptime
            let complete = Double(reused) + Double(downloaded) >= Double(total)
            if !complete && now - lastPayloadTime < 2 { return }
            lastPayloadTime = now
        }
        FileHandle.standardError.write(Data(("[install] " + progress.statusLine + "\n").utf8))
    }
}

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
    var skipMTP = false
    var localGGUF: String?
    var verifyGGUFCopy = false
    var modelGiven = false

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
            case "--skip-mtp":
                parsed.skipMTP = true
                index += 1
            case "--discard-partial":
                parsed.discardPartial = true
                index += 1
            case "--verify-gguf-copy":
                parsed.verifyGGUFCopy = true
                index += 1
            case "--verify-install":
                parsed.verifyInstall = true
                index += 1
            case "--local-gguf":
                guard index + 1 < values.count else {
                    throw ParseError.missingValue(flag)
                }
                parsed.localGGUF = values[index + 1]
                index += 2
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
                parsed.modelGiven = true
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
        if parsed.verifyGGUFCopy {
            guard parsed.inputGTurbo != nil, parsed.localGGUF != nil else {
                throw ParseError.missingRequired("--input-gturbo and --local-gguf")
            }
            guard parsed.output == nil, !parsed.verifyInstall, !parsed.resume, !parsed.overwrite,
                  parsed.attachMTP == nil, !parsed.discardPartial, !parsed.modelGiven,
                  parsed.baseURL == nil, !parsed.dryRun, !parsed.skipMTP else {
                throw ParseError.invalidMode("--verify-gguf-copy accepts only --input-gturbo and --local-gguf")
            }
            return parsed
        }
        if parsed.localGGUF != nil {
            guard parsed.model.kind == .gguf else {
                throw ParseError.invalidMode("--local-gguf needs a GGUF model (inkling2mini)")
            }
            guard parsed.baseURL == nil, !parsed.discardPartial, !parsed.verifyInstall,
                  parsed.attachMTP == nil else {
                throw ParseError.invalidMode("--local-gguf only accepts an install")
            }
        }
        if parsed.discardPartial {
            guard parsed.output != nil else {
                throw ParseError.missingRequired("--output")
            }
            guard parsed.inputGTurbo == nil, !parsed.overwrite, !parsed.verifyInstall,
                  !parsed.skipMTP else {
                throw ParseError.invalidMode("--discard-partial only accepts --output")
            }
            return parsed
        }
        if parsed.attachMTP != nil {
            guard parsed.output != nil else {
                throw ParseError.missingRequired("--output")
            }
            guard parsed.inputGTurbo == nil, !parsed.overwrite, !parsed.resume,
                  !parsed.verifyInstall, !parsed.dryRun, !parsed.skipMTP else {
                throw ParseError.invalidMode("--attach-mtp only accepts --output")
            }
            return parsed
        }
        if parsed.verifyInstall {
            guard parsed.inputGTurbo != nil else {
                throw ParseError.missingRequired("--input-gturbo")
            }
            guard parsed.output == nil, !parsed.overwrite, !parsed.resume,
                  !parsed.skipMTP else {
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

    if arguments.verifyGGUFCopy, let input = arguments.inputGTurbo, let gguf = arguments.localGGUF {
        do {
            let result = try GGUFCopyVerifier.run(installDirectory: input, ggufPath: gguf)
            print("Identical: \(result.residentTensors) resident and \(result.expertTensors) expert tensors, "
                + "\(result.bytesCompared) bytes")
            return 0
        } catch {
            printError("GGUF copy verification failed: \(error)")
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
    var options = source.installOptions(
        outputDirectory: URL(fileURLWithPath: output),
        overwrite: arguments.overwrite,
        token: ProcessInfo.processInfo.environment["HF_TOKEN"],
        resume: arguments.resume,
        baseURL: arguments.baseURL,
        dryRunSpaceCheck: arguments.dryRun,
        sidecarPolicy: SidecarPolicy(carryMTP: !arguments.skipMTP))
    options.localGGUFPath = arguments.localGGUF
    do {
        let printer = InstallProgressPrinter()
        let audit = RepackAudit()
        let result = try await RemoteStreamingRepacker(options: options, audit: audit).run { progress in
            printer.report(progress)
        }
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
            // The per-tensor width mixture. Printed as a census of what the
            // plan will actually write rather than as the policy's rule list,
            // so a dry run can be diffed against a control conversion's own
            // mixture before committing to a multi-hour install.
            let byBits = result.residentWeightWidthCensus
            let census = byBits.keys.sorted()
                .map { "INT\($0) \(byBits[$0]!)" }
                .joined(separator: ", ")
            print("Resident weight widths: "
                + (census.isEmpty ? "none quantized" : census)
                + " (unquantized \(result.residentUnquantizedCount))")
            print("Bit-width overrides: \(result.bitWidthOverrideCount)")
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

import Foundation
import MferenceRepackCore

private let usage = """
Usage:
  MferenceRepack [--dry-run] [--model <gemma4|gemma4qat|qwen36|qwen36original|qwen38|swiftqwen38|deepseekv4flash|inklingsmall|maple|qwen38flashnext|minicpm5|minicpm5mlx|glm53flash>] --output <model.gturbo> [--overwrite] [--resume] [--skip-mtp] [--base-url <url>]
  MferenceRepack --attach-mtp <mtp-shard.safetensors> --output <model.gturbo>
  MferenceRepack --discard-partial --output <model.gturbo>
  MferenceRepack --verify-install --input-gturbo <model.gturbo>
  MferenceRepack --implicit-qat-biases --input-gturbo <gemma4qat.gturbo> --output <new.gturbo>
  MferenceRepack --help

--attach-mtp appends the Qwen 3.8 MTP draft-layer tensors from a local BF16
safetensors shard to an existing install (projections quantized to INT4
affine group-64), enabling speculative decoding.

--implicit-qat-biases writes a copy of a Gemma 4 QAT install whose routed
experts are stored without their bias arrays (each is exactly -8 x scale; the
runtime rebuilds them after every read), about 10 % fewer bytes per expert
read. The input is only read; the output must not exist yet. Every bias group
and every input file hash is checked before the output is finished.

--skip-mtp drops the source checkpoint's own mtp.* draft-layer group instead
of carrying it. Vision towers are always skipped. Either way the decision is
recorded in manifest.json -> sidecars rather than left implicit.

qwen38flashnext and qwen36original read the model vendor's original BF16 repo
and quantize to INT4 affine group-64 during the install. qwen38flashnext keeps
its two gating tensors per layer at INT8; its capability gate was lifted on
2026-09-10, so an install loads through the ordinary funnel. qwen36original is
the same checkpoint qwen36 installs from mlx-community's pre-quantized
conversion, put through our own quantizer instead; installing both and
comparing them is the W2.1b quantizer-quality gate
(docs/QUANTIZER_QUALITY.md).

swiftqwen38 reads the pinned ukisai/Swift-Qwen3.8-27b BF16 source and quantizes
its text and MTP weights in flight. It is a qualification candidate, distinct
from base qwen38; see docs/families/SWIFT_QWEN38.md.

minicpm5 reads openbmb/MiniCPM5-2B's BF16 upload and quantizes to INT4 affine
group-64 during the install (~5 GB read, ~1.4 GB written); minicpm5mlx is the
vendor's own MLX INT4 conversion of the same checkpoint, the W2.1b control.

glm53flash reads PipeNetwork's pre-quantized mixed 4/8-bit MLX conversion of
GLM-5.3-Flash (~181 GB; check disk first). Resident and bounded streamed
prefill are implemented; see docs/families/GLM53_FLASH.md and the prefill
qualification matrix for the validated profiles and remaining limits.

gemma4qat preserves the aligned checkpoint's native INT4/group-32 weights and
BF16 routers. Its install is separate from gemma4 and uses the checkpoint's
installed chat template and generation settings. The launcher installs it under
$HOME/llm-models/gemma4qat.gturbo.

The installer streams the selected checkpoint (default: the supported Gemma 4
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
    var implicitQATBiases = false
    var attachMTP: String?
    var inputGTurbo: String?
    var baseURL: URL?
    var dryRun = false
    var skipMTP = false

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
            case "--verify-install":
                parsed.verifyInstall = true
                index += 1
            case "--implicit-qat-biases":
                parsed.implicitQATBiases = true
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
            guard parsed.inputGTurbo == nil, !parsed.overwrite, !parsed.verifyInstall,
                  !parsed.skipMTP else {
                throw ParseError.invalidMode("--discard-partial only accepts --output")
            }
            return parsed
        }
        if parsed.implicitQATBiases {
            guard parsed.inputGTurbo != nil, parsed.output != nil else {
                throw ParseError.missingRequired("--input-gturbo and --output")
            }
            guard parsed.attachMTP == nil, !parsed.verifyInstall, !parsed.overwrite,
                  !parsed.resume, !parsed.dryRun, !parsed.skipMTP, parsed.baseURL == nil else {
                throw ParseError.invalidMode(
                    "--implicit-qat-biases only accepts --input-gturbo and --output")
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

    if arguments.implicitQATBiases, let input = arguments.inputGTurbo, let output = arguments.output {
        do {
            let result = try QATImpliedBiasConverter.run(
                inputGTurbo: input, outputGTurbo: output,
                progress: { print($0) })
            print("Converted \(result.layers) layers: expert \(result.expertStride) -> "
                + "\(result.storedExpertStride) bytes; routed experts \(result.routedBytesBefore) -> "
                + "\(result.routedBytesAfter) bytes; \(result.groupsChecked) bias groups checked")
            print("Output: \(output)")
            return 0
        } catch {
            printError("implicit-qat-biases failed: \(error)")
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
        dryRunSpaceCheck: arguments.dryRun,
        sidecarPolicy: SidecarPolicy(carryMTP: !arguments.skipMTP))
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
            print("Source bytes to read: \(result.remoteBytesToDownload + result.sourceMetadataBytes)")
            print("Output bytes: \(result.outputBytes)")
            if result.supportingFileBytes > 0 {
                print("Source metadata/assets bytes (included, excluding retries): \(result.sourceMetadataBytes)")
                print("Resumable payload range bytes: \(result.remoteBytesToDownload)")
                print("Required assets and layout bytes (included): \(result.supportingFileBytes)")
                print("Additional metadata/staging reserve bytes: \(result.metadataReserveBytes)")
                print("Free-space reserve bytes: \(source.reserveBytes)")
                print("Manifest and receipt final sizes are accounted from the metadata reserve.")
            }
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
        if source == .gemma4QAT {
            print("Installed bytes: \(result.outputBytes)")
            print("Payload bytes reused: \(result.reusedBytes); downloaded this run: \(result.downloadedThisRunBytes)")
            print("Source metadata/assets bytes (excluding retries): \(result.sourceMetadataBytes)")
            print("Largest temporary payload range bytes: \(audit.largestRemoteTransferBytes); writer scratch bytes: \(audit.largestScratchBytes)")
            print("Range retries: \(result.remoteRetryCount)")
            print("Install elapsed seconds: \(String(format: "%.3f", audit.wallTimeSeconds))")
            print("Installation verified; select the separate gemma-4-26b-a4b-it-qat-q4_0-mlx-aligned model.")
        }
        return 0
    } catch {
        printError("install failed: \(error)")
        return 1
    }
}

exit(await run(CommandLine.arguments))

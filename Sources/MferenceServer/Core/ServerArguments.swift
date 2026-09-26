import Foundation
import Mference

public struct ServerArguments: Equatable, Sendable {
    /// Model directory. Required in single-model mode; optional in library
    /// mode, where it preloads one install instead of loading lazily.
    public let model: String?
    public let port: Int
    public let bindMode: ServerBindMode
    /// Explicit --model-id value; nil defers to the loaded model's family
    /// default for the loaded model family.
    public let modelIDOverride: String?
    public var modelID: String { modelIDOverride ?? "gemma-4-26b-a4b-it" }
    /// `--max-context`; nil (`max`) gives each model its native context.
    public let maxContext: Int?
    public let queueLimit: Int
    public let promptCacheMode: ServerPromptCacheMode
    /// `--verify`; see `ModelIntegrityPolicy`.
    public let verification: ModelIntegrityPolicy
    /// `--shadow-budget`; nil keeps the family default.
    public let shadowBudget: Int?
    /// `--prefill-chunk`; nil (`auto`) keeps the family default.
    public let prefillChunk: Int?
    /// nil when `--library` was absent, which keeps single-model mode exactly
    /// as it was.
    public let library: ServerLibraryOption?
    /// `--list-models`: run discovery, print what library mode would serve, and
    /// exit without binding a port or loading a model.
    public let listModels: Bool

    public static let usage = """
    usage: MferenceServer --model <completed .gturbo directory> [options]
           MferenceServer --library [dir] [options]
           MferenceServer --library [dir] --list-models

      --model <dir>          Model directory. Required unless --library is
                             given, where it preloads one install instead.
      --library [dir]        Serve runnable completed installs under <dir>,
                             repeatable. With no value, scans the default
                             roots: the Mference.libraryRoot default (or
                             MFERENCE_LIBRARY_ROOT), ~/llm-models,
                             the package checkout's scratch/, and
                             ~/Library/Application Support/Mference.
                             Starting with nothing installed is fine: the
                             model list is then empty.
                             GET /v1/models lists them all; a request naming a
                             model that is not resident unloads the current one
                             and loads it in place. One model is loaded at a
                             time and no second process is ever started.
                             Gemma QAT has its own model ID, installed chat
                             template and sampling defaults.
      --list-models          With --library: print the installs discovery found
                             — identifier, family, installed bytes, and path —
                             then exit 0 without binding a port or loading a
                             model. This is what `./mference-ui.sh models`
                             reports.
      --port <1...65535>     Listening port (default 8080).
      --bind <mode>          loopback or tailnet (default loopback). tailnet
                             binds only the machine's Tailscale IPv4 address
                             and fails when Tailscale is unavailable.
      --model-id <id>        API model identifier (default derived from the
                             installed model: gemma-4-26b-a4b-it,
                             qwen3.6-35b-a3b, qwen3.8-27b-4bit,
                             deepseek-v4-flash-2bit-dq, inkling-small-4bit,
                             maple-preview-2bit-mlx,
                             qwen3.8-flash-next-int4g64,
                             minicpm5-2b-int4g64, or the manifest's distinct
                             GLM / Swift-Qwen checkpoint ID). Single-model mode only.
      --max-context <tokens|max>
                             Context length in tokens (default 16384), up to
                             the model's native context: 262144 for Gemma 4,
                             Qwen 3.6, Qwen 3.8 and Flash-Next, 131072 for
                             MiniCPM5, 128000 for Maple, 1048576 for
                             DeepSeek-V4, Inkling and GLM-5.3. A model whose
                             native context is shorter refuses to load. max
                             gives each model its own native context.
      --queue-limit <count>  Maximum queued requests (default 4).
      --prompt-cache-mode <off|single-prefix>
                             Prompt KV reuse mode (default single-prefix).
      --shadow-budget <0...8>
                             Speculative expert prefetch during decode: reads
                             issued per layer ahead of the router; 0 turns it off.
                             Default: 4 for Qwen 3.6 and Gemma 4 on hosts with 16
                             to under 24 GiB, 2 for DeepSeek-V4-Flash, off elsewhere.
      --prefill-chunk <n|auto>
                             Prompt tokens per prefill chunk: auto (default) or
                             32, 64, 128, 256, 512, 1024, 2048, 4096. Larger
                             chunks re-read routed experts less often but hold
                             more memory. auto is 2048 for Gemma 4 (QAT
                             included) and Qwen 3.6 on hosts with at least
                             16 GiB, 128 elsewhere; 1024 saves Gemma about
                             350 MB. Overrides MFERENCE_SERVER_PREFILL_CHUNK.
      --verify <mode>        Model integrity on load and model swap: auto
                             (default) checks file sizes against the install
                             receipt when it validates and hashes otherwise;
                             full-sha256 re-hashes every routed-expert file on
                             first touch; trusted-receipt requires the receipt.
      --help                 Show this help.
    """

    public static func parse(_ input: [String]) throws -> ServerArguments {
        var model: String?
        var port = 8080
        var bindMode = ServerBindMode.loopback
        var modelIDOverride: String?
        var maxContext: Int? = 16_384
        var queueLimit = 4
        var promptCacheMode: ServerPromptCacheMode = .singlePrefix
        var verification = ModelIntegrityPolicy.trustedReceiptWhenValid
        var shadowBudget: Int?
        var prefillChunk: Int?
        var libraryRoots: [String] = []
        var wantsDefaultLibraryRoots = false
        var listModels = false
        var index = 0
        while index < input.count {
            let flag = input[index]
            if flag == "--help" || flag == "-h" { throw ServerArgumentError.help }
            if flag == "--list-models" {
                listModels = true
                index += 1
                continue
            }
            // The only flag whose value is optional: bare `--library` means the
            // default roots, so a following `--flag` or the end of the argument
            // list terminates it rather than being eaten as a path.
            if flag == "--library" {
                let next = index + 1 < input.count ? input[index + 1] : nil
                if let next, !next.hasPrefix("--") {
                    guard !next.isEmpty else {
                        throw ServerArgumentError.invalid("--library must not be empty")
                    }
                    libraryRoots.append(next)
                    index += 2
                } else {
                    wantsDefaultLibraryRoots = true
                    index += 1
                }
                continue
            }
            guard index + 1 < input.count else {
                throw ServerArgumentError.invalid("\(flag) requires a value")
            }
            let value = input[index + 1]
            index += 2
            switch flag {
            case "--model":
                model = value
            case "--port":
                guard let parsed = Int(value), (1...65_535).contains(parsed) else {
                    throw ServerArgumentError.invalid("--port must be between 1 and 65535")
                }
                port = parsed
            case "--bind":
                guard let parsed = ServerBindMode(rawValue: value) else {
                    throw ServerArgumentError.invalid("--bind must be loopback or tailnet")
                }
                bindMode = parsed
            case "--model-id":
                guard !value.isEmpty else {
                    throw ServerArgumentError.invalid("--model-id must not be empty")
                }
                modelIDOverride = value
            case "--max-context":
                if value == "max" {
                    maxContext = nil
                } else {
                    guard let parsed = Int(value),
                          (1...ModelFamily.largestMaximumContext).contains(parsed) else {
                        throw ServerArgumentError.invalid(
                            "--max-context must be max or between 1 and \(ModelFamily.largestMaximumContext)")
                    }
                    maxContext = parsed
                }
            case "--queue-limit":
                guard let parsed = Int(value), parsed > 0 else {
                    throw ServerArgumentError.invalid("--queue-limit must be positive")
                }
                queueLimit = parsed
            case "--prompt-cache-mode":
                guard let parsed = ServerPromptCacheMode(rawValue: value) else {
                    throw ServerArgumentError.invalid(
                        "--prompt-cache-mode must be off or single-prefix")
                }
                promptCacheMode = parsed
            case "--shadow-budget":
                guard let parsed = Int(value),
                      RuntimeConfiguration.allowedShadowPrefetchBudgets.contains(parsed) else {
                    throw ServerArgumentError.invalid("--shadow-budget must be 0 through 8")
                }
                shadowBudget = parsed
            case "--prefill-chunk":
                if value == "auto" {
                    prefillChunk = nil
                } else {
                    guard let parsed = Int(value),
                          RuntimeConfiguration.allowedPrefillChunkTokens.contains(parsed) else {
                        throw ServerArgumentError.invalid(
                            "--prefill-chunk must be auto, 32, 64, 128, 256, 512, 1024, 2048 or 4096")
                    }
                    prefillChunk = parsed
                }
            case "--verify":
                guard let parsed = ModelIntegrityPolicy(verifyFlag: value) else {
                    throw ServerArgumentError.invalid(
                        "--verify must be \(ModelIntegrityPolicy.verifyFlagValues)")
                }
                verification = parsed
            default:
                throw ServerArgumentError.invalid("unknown flag: \(flag)")
            }
        }
        let library: ServerLibraryOption?
        if wantsDefaultLibraryRoots || !libraryRoots.isEmpty {
            library = ServerLibraryOption(roots: libraryRoots,
                                          includesDefaultRoots: wantsDefaultLibraryRoots)
        } else {
            library = nil
        }
        if library == nil {
            // Listing is a property of the library, so there is nothing to list
            // without one; single-model mode already names its model on the
            // command line.
            guard !listModels else {
                throw ServerArgumentError.invalid("--list-models requires --library")
            }
            guard model != nil else {
                throw ServerArgumentError.invalid("--model is required")
            }
        } else if modelIDOverride != nil {
            // Library mode derives one identifier per install; a single
            // override could only ever name one of them.
            throw ServerArgumentError.invalid("--model-id cannot be combined with --library")
        }
        return ServerArguments(model: model,
                               port: port,
                               bindMode: bindMode,
                               modelIDOverride: modelIDOverride,
                               maxContext: maxContext,
                               queueLimit: queueLimit,
                               promptCacheMode: promptCacheMode,
                               verification: verification,
                               shadowBudget: shadowBudget,
                               prefillChunk: prefillChunk,
                               library: library,
                               listModels: listModels)
    }
}

/// How `--library` was requested. Explicit roots and the default roots
/// combine: `--library a --library` scans `a` and the defaults.
public struct ServerLibraryOption: Equatable, Sendable {
    public let roots: [String]
    public let includesDefaultRoots: Bool

    public init(roots: [String], includesDefaultRoots: Bool) {
        self.roots = roots
        self.includesDefaultRoots = includesDefaultRoots
    }

    /// Resolved roots, explicit ones first and relative paths made absolute
    /// against `currentDirectoryURL`.
    public func resolvedRoots(
        currentDirectoryURL: URL = URL(
            fileURLWithPath: FileManager.default.currentDirectoryPath,
            isDirectory: true),
        defaultRoots: () -> [URL] = { ServerLibraryDiscovery.defaultRoots() }
    ) -> [URL] {
        var resolved = roots.map { path -> URL in
            if path.hasPrefix("/") {
                return URL(fileURLWithPath: path, isDirectory: true).standardizedFileURL
            }
            return currentDirectoryURL.appendingPathComponent(path, isDirectory: true)
                .standardizedFileURL
        }
        if includesDefaultRoots { resolved.append(contentsOf: defaultRoots()) }
        var seen = Set<String>()
        return resolved.filter { seen.insert($0.path).inserted }
    }
}

/// Interface the server listens on. Resolution fails rather than widening:
/// there is no path from `.tailnet` to a wildcard or LAN address.
public enum ServerBindMode: String, Equatable, Sendable {
    case loopback
    case tailnet

    /// Resolves the listening address. `.tailnet` asks the Tailscale CLI for
    /// this machine's IPv4 address; `tailnetAddresses` is the seam tests use to
    /// supply that output without a Tailscale install.
    public func host(
        tailnetAddresses: () throws -> String = ServerBindMode.tailscaleIPv4Output
    ) throws -> String {
        switch self {
        case .loopback: "127.0.0.1"
        case .tailnet: try Self.tailnetHost(from: tailnetAddresses())
        }
    }

    /// Accepts exactly one Tailscale IPv4 address. Empty, ambiguous, IPv6-only,
    /// malformed, and off-range output all fail; none of them fall back.
    static func tailnetHost(from output: String) throws -> String {
        let fields = output.split(whereSeparator: \.isWhitespace)
        guard let first = fields.first else {
            throw ServerArgumentError.invalid(
                "tailscale reported no IPv4 address; ensure Tailscale is running and connected")
        }
        guard fields.count == 1 else {
            throw ServerArgumentError.invalid(
                "tailscale reported \(fields.count) IPv4 addresses; refusing to guess which to bind")
        }
        guard let address = tailscaleIPv4(String(first)) else {
            throw ServerArgumentError.invalid(
                "tailscale reported \"\(first.prefix(64))\", which is not a Tailnet IPv4 address")
        }
        return address
    }

    /// Returns the address unchanged when it is a dotted-quad IPv4 inside
    /// 100.64.0.0/10, the range Tailscale allocates from. Restricting to that
    /// range keeps a wildcard, loopback, or LAN address from ever being bound.
    private static func tailscaleIPv4(_ text: String) -> String? {
        let parts = text.split(separator: ".", omittingEmptySubsequences: false)
        guard parts.count == 4 else { return nil }
        var octets: [UInt8] = []
        octets.reserveCapacity(4)
        for part in parts {
            guard (1...3).contains(part.count),
                  part.allSatisfy({ $0.isASCII && $0.isNumber }),
                  part.count == 1 || part.first != "0",
                  let octet = UInt8(part) else { return nil }
            octets.append(octet)
        }
        guard octets[0] == 100, (64...127).contains(octets[1]) else { return nil }
        return text
    }

    /// Raw stdout of `tailscale ip -4`. Spawned directly with no shell, so
    /// nothing is interpolated into a command line.
    public static func tailscaleIPv4Output() throws -> String {
        let process = Process()
        let output = Pipe()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        process.arguments = ["tailscale", "ip", "-4"]
        process.standardOutput = output
        process.standardError = FileHandle.nullDevice
        do {
            try process.run()
        } catch {
            throw ServerArgumentError.invalid(
                "could not run tailscale; install its CLI and keep it on PATH")
        }
        let data = output.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        guard process.terminationStatus == 0 else {
            throw ServerArgumentError.invalid(
                "tailscale ip -4 exited with status \(process.terminationStatus); ensure the tailscale CLI is on PATH and Tailscale is running")
        }
        guard let text = String(data: data, encoding: .utf8) else {
            throw ServerArgumentError.invalid("tailscale ip -4 returned non-UTF-8 output")
        }
        return text
    }
}

public enum ServerArgumentError: Error, Equatable, CustomStringConvertible {
    case help
    case invalid(String)

    public var description: String {
        switch self {
        case .help: "help"
        case .invalid(let message): message
        }
    }
}

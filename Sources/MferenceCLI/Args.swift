import Mference

/// Prefill chunk selection. `.fixed` must name an allowed size;
/// `.auto` resolves to the smallest allowed size covering the prompt,
/// which minimizes routed-expert re-reads (expert I/O scales with
/// prompt_tokens / chunk_tokens).
public enum PrefillChunkChoice: Equatable, Sendable {
    case fixed(Int)
    case auto
}

/// Routed-expert cache selection. Auto keeps the 16-slot memory-first default
/// for every family except Qwen 3.6, whose 256 experts per layer measurably
/// benefit from larger rungs: 96 slots on hosts with at least 24 GiB,
/// 32 with at least 16 GiB.
public enum ExpertCacheSlotChoice: Equatable, Sendable {
    case fixed(Int)
    case resident
    case auto
}

public struct Args: Equatable, Sendable {
    public var model: String
    public var prompt: String?
    public var messagesFile: String?
    public var chat: Bool
    public var systemPrompt: String?
    public var maxNew: Int
    public var maxContext: Int
    public var temperature: Float
    public var topK: Int?
    public var topP: Float?
    public var repetitionPenalty: Float
    public var seed: UInt64?
    public var stops: [String]
    public var quiet: Bool
    public var expertCacheSlots: ExpertCacheSlotChoice
    public var rdadvise: String
    public var prefillChunk: PrefillChunkChoice
    /// Enables Maple's approximate sparse singleton-decode head when the
    /// installed checkpoint carries the required FlashHead tensors.
    public var flashHead: Bool
    /// Model-integrity policy. `.fullSha256` re-hashes every routed-expert
    /// file on first touch — 145 GB for Inkling-Small, ~59 s inside the first
    /// prefill. `.sizeCheckTrustedReceipt` checks sizes against the receipt
    /// written at install time instead. Mirrors the Mac app's existing
    /// verification control.
    public var verification: ModelIntegrityPolicy
    /// Paged KV cache with SSD spill + sparse decode (Qwen 3.8):
    /// "on" / "off" / "auto" (auto enables it above 32k context).
    public var kvPaged: String
    /// Sparse decode selection budget in 64-token pages.
    public var kvTopKPages: Int
    /// Resident pool per full-attention layer in pages; nil = auto by RAM.
    public var kvPoolPages: Int?

    public init(model: String,
                prompt: String? = nil,
                messagesFile: String? = nil,
                chat: Bool = false,
                systemPrompt: String? = nil,
                maxNew: Int = 1_024,
                maxContext: Int = 4096,
                temperature: Float = 0.2,
                topK: Int? = 64,
                topP: Float? = 0.95,
                repetitionPenalty: Float = 1.0,
                seed: UInt64? = nil,
                stops: [String] = [],
                quiet: Bool = false,
                expertCacheSlots: ExpertCacheSlotChoice = .auto,
                rdadvise: String = "off",
                prefillChunk: PrefillChunkChoice = .auto,
                flashHead: Bool = false,
                verification: ModelIntegrityPolicy = .fullSha256,
                kvPaged: String = "auto",
                kvTopKPages: Int = 60,
                kvPoolPages: Int? = nil) {
        self.model = model
        self.prompt = prompt
        self.messagesFile = messagesFile
        self.chat = chat
        self.systemPrompt = systemPrompt
        self.maxNew = maxNew
        self.maxContext = maxContext
        self.temperature = temperature
        self.topK = topK
        self.topP = topP
        self.repetitionPenalty = repetitionPenalty
        self.expertCacheSlots = expertCacheSlots
        self.rdadvise = rdadvise
        self.prefillChunk = prefillChunk
        self.flashHead = flashHead
        self.verification = verification
        self.kvPaged = kvPaged
        self.kvTopKPages = kvTopKPages
        self.kvPoolPages = kvPoolPages
        self.seed = seed
        self.stops = stops
        self.quiet = quiet
    }
}

public enum ArgsError: Error, Equatable, CustomStringConvertible {
    case helpRequested
    case unknownFlag(String)
    case missingValue(flag: String)
    case invalidValue(flag: String, value: String)
    case requiredMissing(String)
    case mutuallyExclusive(String, String)
    case modeMissing

    public var description: String {
        switch self {
        case .helpRequested: return "help requested"
        case .unknownFlag(let flag): return "unknown flag: \(flag)"
        case .missingValue(let flag): return "missing value for \(flag)"
        case .invalidValue(let flag, let value): return "invalid value for \(flag): \(value)"
        case .requiredMissing(let flag): return "required flag missing: \(flag)"
        case .mutuallyExclusive(let a, let b): return "\(a) and \(b) are mutually exclusive"
        case .modeMissing: return "one of --prompt, --messages-file, or --chat is required"
        }
    }
}

extension Args {
    public static let usage = """
    MferenceCLI — Gemma 4 / Qwen 3.6 / DeepSeek V4 Flash / Inkling-Small / Maple text generation

    usage: MferenceCLI --model <dir> (--prompt <string> | --messages-file <path> | --chat) [options]

    required:
      --model <dir>             Path to a .gturbo model directory.

    modes (exactly one):
      --prompt <string>         Raw-completion prompt.
      --messages-file <path>    JSON chat messages with role and content fields.
      --chat                    Interactive multi-turn chat on stdin.

    options:
      --system <string>         System message for --chat (repeatable).
      --max-new <int>           Generated-token limit (default 1024).
      --max-context <int>       Context limit in tokens (default 4096).
      --kv-paged <on|off|auto>  Paged KV cache with SSD spill + Quest sparse
                                decode (Qwen 3.8; default auto: on above 32k
                                context). Exact when everything fits RAM.
      --kv-topk <pages>         Sparse decode budget in 64-token pages
                                (default 60 ≈ 3.8k attended tokens/layer).
      --kv-pool-pages <n|auto>  Resident pool per full-attention layer in
                                pages (default auto: sized from RAM).
      --temperature <float>     Sampling temperature (default 0.2; 0 = greedy).
      --top-k <int>             Top-k truncation, 1...256 (default 64; 0 = off).
      --top-p <float>           Nucleus truncation (default 0.95).
      --repetition-penalty <f>  Repetition penalty (default 1.0).
      --seed <uint64>           Deterministic sampling seed (default off).
      --stop <string>           Stop substring (repeatable).
      --rdadvise <mode>         Expert read-ahead advice: off, default,
                                bounded, or adaptive (default off).
      --expert-cache-slots <n|resident|auto>
                                Routed-expert cache slots per layer: 8, 16,
                                24, 32, 64, 96, 128, resident, or auto.
                                resident maps every layer file once and skips
                                the slot cache entirely. auto always uses the
                                slot cache: Qwen gets 96 slots on hosts with
                                at least 24 GiB, 32 with at least 16 GiB,
                                else 16; other families get 16.
                                More slots raise the hit rate but use more RAM.
      --prefill-chunk <n|auto>  Prefill chunk tokens (default auto). Larger
                                chunks cut routed-expert re-reads during
                                prompt processing; auto sizes the chunk to
                                the prompt (--chat resolves auto to 128). Allowed:
                                32, 64, 128, 256, 512, 1024, 2048, 4096.
      --flash-head              Enable Maple's approximate sparse decode head.
                                Prefill remains exact; unsupported models use
                                the exact head.
      --verify <mode>           Model integrity: full-sha256 (default)
                                re-hashes every routed-expert file on first
                                touch, which for a 145 GB expert pool costs
                                ~59 s inside the first prefill;
                                trusted-receipt checks file sizes against the
                                receipt written at install time instead.
      --quiet                   Suppress the timing footer.
      --help                    Show this message.
    """

    public static func parse(_ argv: [String]) throws -> Args {
        var model: String?
        var prompt: String?
        var messagesFile: String?
        var chat = false
        var systemPrompt: String?
        var maxNew = 1_024
        var maxContext = 4096
        var temperature: Float = 0.2
        var topK: Int? = 64
        var topP: Float? = 0.95
        var repetitionPenalty: Float = 1.0
        var seed: UInt64?
        var stops: [String] = []
        var quiet = false
        var expertCacheSlots = ExpertCacheSlotChoice.auto
        var rdadvise = "off"
        var prefillChunk = PrefillChunkChoice.auto
        var flashHead = false
        var verification = ModelIntegrityPolicy.fullSha256
        var kvPaged = "auto"
        var kvTopKPages = 60
        var kvPoolPages: Int? = nil

        var index = 0
        while index < argv.count {
            let flag = argv[index]
            switch flag {
            case "--help":
                throw ArgsError.helpRequested
            case "--quiet":
                quiet = true
                index += 1
            case "--flash-head":
                flashHead = true
                index += 1
            case "--model":
                model = try takeValue(argv, &index, flag: flag)
            case "--prompt":
                prompt = try takeValue(argv, &index, flag: flag)
            case "--messages-file":
                messagesFile = try takeValue(argv, &index, flag: flag)
            case "--chat":
                chat = true
                index += 1
            case "--system":
                let value = try takeValue(argv, &index, flag: flag)
                systemPrompt = systemPrompt.map { $0 + "\n" + value } ?? value
            case "--max-new":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = Int(value), parsed > 0 else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                maxNew = parsed
            case "--max-context":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = Int(value), parsed > 0 else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                maxContext = parsed
            case "--kv-paged":
                let value = try takeValue(argv, &index, flag: flag)
                guard ["on", "off", "auto"].contains(value) else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                kvPaged = value
            case "--kv-topk":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = Int(value), parsed >= 0 else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                kvTopKPages = parsed
            case "--kv-pool-pages":
                let value = try takeValue(argv, &index, flag: flag)
                if value == "auto" {
                    kvPoolPages = nil
                } else {
                    guard let parsed = Int(value), parsed > 0 else {
                        throw ArgsError.invalidValue(flag: flag, value: value)
                    }
                    kvPoolPages = parsed
                }
            case "--temperature":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = Float(value), parsed >= 0 else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                temperature = parsed
            case "--top-k":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = Int(value), (0...256).contains(parsed) else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                topK = parsed == 0 ? nil : parsed
            case "--top-p":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = Float(value), parsed > 0, parsed <= 1 else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                topP = parsed
            case "--repetition-penalty":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = Float(value), parsed > 0 else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                repetitionPenalty = parsed
            case "--seed":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = UInt64(value) else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                seed = parsed
            case "--expert-cache-slots":
                let value = try takeValue(argv, &index, flag: flag)
                if value == "auto" {
                    expertCacheSlots = .auto
                } else if value == "resident" {
                    expertCacheSlots = .resident
                } else if let parsed = Int(value),
                          RuntimeConfiguration.allowedExpertCacheSlots.contains(parsed) {
                    expertCacheSlots = .fixed(parsed)
                } else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
            case "--prefill-chunk":
                let value = try takeValue(argv, &index, flag: flag)
                if value == "auto" {
                    prefillChunk = .auto
                } else if let parsed = Int(value),
                          RuntimeConfiguration.allowedPrefillChunkTokens
                              .contains(parsed) {
                    prefillChunk = .fixed(parsed)
                } else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
            case "--verify":
                let value = try takeValue(argv, &index, flag: flag)
                switch value {
                case "full-sha256": verification = .fullSha256
                case "trusted-receipt": verification = .sizeCheckTrustedReceipt
                default: throw ArgsError.invalidValue(flag: flag, value: value)
                }
            case "--rdadvise":
                let value = try takeValue(argv, &index, flag: flag)
                guard ["off", "default", "bounded", "adaptive"].contains(value) else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                rdadvise = value
            case "--stop":
                stops.append(try takeValue(argv, &index, flag: flag))
            default:
                throw ArgsError.unknownFlag(flag)
            }
        }

        guard let model else { throw ArgsError.requiredMissing("--model") }
        if prompt != nil && messagesFile != nil {
            throw ArgsError.mutuallyExclusive("--prompt", "--messages-file")
        }
        if chat && prompt != nil {
            throw ArgsError.mutuallyExclusive("--prompt", "--chat")
        }
        if chat && messagesFile != nil {
            throw ArgsError.mutuallyExclusive("--messages-file", "--chat")
        }
        if prompt == nil && messagesFile == nil && !chat { throw ArgsError.modeMissing }
        if systemPrompt != nil && !chat {
            throw ArgsError.invalidValue(flag: "--system", value: "requires --chat")
        }
        if temperature > 0, topK == nil, let topP, topP < 1 {
            throw ArgsError.invalidValue(
                flag: "--top-p",
                value: "\(topP) requires --top-k between 1 and 256")
        }
        return Args(model: model,
                    prompt: prompt,
                    messagesFile: messagesFile,
                    chat: chat,
                    systemPrompt: systemPrompt,
                    maxNew: maxNew,
                    maxContext: maxContext,
                    temperature: temperature,
                    topK: topK,
                    topP: topP,
                    repetitionPenalty: repetitionPenalty,
                    seed: seed,
                    stops: stops,
                    quiet: quiet,
                    expertCacheSlots: expertCacheSlots,
                    rdadvise: rdadvise,
                    prefillChunk: prefillChunk,
                    flashHead: flashHead,
                    verification: verification,
                    kvPaged: kvPaged,
                    kvTopKPages: kvTopKPages,
                    kvPoolPages: kvPoolPages)
    }

    private static func takeValue(_ argv: [String],
                                  _ index: inout Int,
                                  flag: String) throws -> String {
        guard index + 1 < argv.count else { throw ArgsError.missingValue(flag: flag) }
        let value = argv[index + 1]
        index += 2
        return value
    }
}

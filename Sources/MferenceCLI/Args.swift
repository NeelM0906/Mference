import Foundation
import Mference

/// Prefill chunk selection. `.fixed` must name an allowed size;
/// `.auto` resolves to the smallest allowed size covering the prompt,
/// which minimizes routed-expert re-reads (expert I/O scales with
/// prompt_tokens / chunk_tokens).
public enum PrefillChunkChoice: Equatable, Sendable {
    case fixed(Int)
    case auto

    /// Interactive chat has no prompt at load time, so `auto` takes the
    /// family's server chunk: both prefill a growing conversation turn after
    /// turn. A fixed size applies to every turn's prefill; the server's
    /// environment override does not reach the CLI, whose control is the flag.
    func chatChunkTokens(
        for family: ModelFamily,
        physicalMemoryBytes: UInt64 = ProcessInfo.processInfo.physicalMemory
    ) -> Int {
        switch self {
        case .fixed(let n): return n
        case .auto:
            return RuntimeConfiguration.defaultServerPrefillChunkTokens(
                for: family, physicalMemoryBytes: physicalMemoryBytes, environment: [:])
        }
    }
}

/// Routed-expert cache selection. Auto keeps the 16-slot memory-first default
/// except where a larger rung has been measured: Qwen 3.6 uses 96 slots on
/// hosts with at least 24 GiB and 32 with at least 16 GiB. Flash-Next maps its
/// routed pool on hosts with at least 192 GiB when 32 GiB of headroom remains.
public enum ExpertCacheSlotChoice: Equatable, Sendable {
    case fixed(Int)
    case resident
    case auto
}

public struct Args: Equatable, Sendable {
    enum SamplingOption: CaseIterable, Sendable {
        case temperature, topK, topP, repetitionPenalty, presencePenalty
        case frequencyPenalty, repeatLastN, minP
    }
    // Parsing records omission until the installed checkpoint is known.
    // Direct initialization supplies concrete values; later property writes
    // are explicit overrides even when they equal a shared default or zero.
    var omittedSamplingOptions: Set<SamplingOption> = []
    public var reasoningEffort: QwenReasoningEffort?
    public var model: String
    public var prompt: String?
    public var messagesFile: String?
    public var chat: Bool
    public var systemPrompt: String?
    public var reusePrefix: Bool
    public var showReasoning: Bool
    public var maxNew: Int
    public var maxContext: Int
    public var temperature: Float { didSet { omittedSamplingOptions.remove(.temperature) } }
    public var topK: Int? { didSet { omittedSamplingOptions.remove(.topK) } }
    public var topP: Float? { didSet { omittedSamplingOptions.remove(.topP) } }
    public var repetitionPenalty: Float { didSet { omittedSamplingOptions.remove(.repetitionPenalty) } }
    public var presencePenalty: Float { didSet { omittedSamplingOptions.remove(.presencePenalty) } }
    public var frequencyPenalty: Float { didSet { omittedSamplingOptions.remove(.frequencyPenalty) } }
    public var repeatLastN: Int { didSet { omittedSamplingOptions.remove(.repeatLastN) } }
    public var minP: Float { didSet { omittedSamplingOptions.remove(.minP) } }
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
    /// written at install time instead. The default uses the receipt when it
    /// validates and hashes otherwise.
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
                reusePrefix: Bool = false,
                showReasoning: Bool = false,
                maxNew: Int = 1_024,
                maxContext: Int = 4096,
                temperature: Float = GenerationConfig.defaults.temperature,
                topK: Int? = GenerationConfig.defaults.topK,
                topP: Float? = GenerationConfig.defaults.topP,
                repetitionPenalty: Float = GenerationConfig.defaults.repetitionPenalty,
                presencePenalty: Float = GenerationConfig.defaults.presencePenalty,
                frequencyPenalty: Float = GenerationConfig.defaults.frequencyPenalty,
                repeatLastN: Int = GenerationConfig.defaults.repeatLastN,
                minP: Float = GenerationConfig.defaults.minP,
                seed: UInt64? = nil,
                stops: [String] = [],
                quiet: Bool = false,
                expertCacheSlots: ExpertCacheSlotChoice = .auto,
                rdadvise: String = "off",
                prefillChunk: PrefillChunkChoice = .auto,
                flashHead: Bool = false,
                verification: ModelIntegrityPolicy = .trustedReceiptWhenValid,
                kvPaged: String = "auto",
                kvTopKPages: Int = 60,
                kvPoolPages: Int? = nil,
                reasoningEffort: QwenReasoningEffort? = nil) {
        self.reasoningEffort = reasoningEffort
        self.model = model
        self.prompt = prompt
        self.messagesFile = messagesFile
        self.chat = chat
        self.systemPrompt = systemPrompt
        self.reusePrefix = reusePrefix
        self.showReasoning = showReasoning
        self.maxNew = maxNew
        self.maxContext = maxContext
        self.temperature = temperature
        self.topK = topK
        self.topP = topP
        self.repetitionPenalty = repetitionPenalty
        self.presencePenalty = presencePenalty
        self.frequencyPenalty = frequencyPenalty
        self.repeatLastN = repeatLastN
        self.minP = minP
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
    private static let samplingDefaults = GenerationConfig.defaults

    public static let usage = """
    MferenceCLI — Gemma 4 / Qwen 3.6 / DeepSeek V4 Flash / Inkling-Small / Maple / GLM-5.3 Flash text generation

    usage: MferenceCLI --model <dir> (--prompt <string> | --messages-file <path> | --chat) [options]

    required:
      --model <dir>             Path to a .gturbo model directory.

    modes (exactly one):
      --prompt <string>         Raw-completion prompt.
      --messages-file <path>    JSON chat messages with role and content fields.
      --chat                    Interactive multi-turn chat on stdin.

    Sampling defaults below apply to existing checkpoints. Gemma QAT
    uses its verified local generation_config.json: temperature 1, top-k 64,
    top-p 0.95, min-p 0 and neutral penalties. Explicit flags take precedence.
    QAT chat uses its installed checkpoint template; preserve_thinking=true is unsupported.

    options:
      --system <string>         System message for --chat (repeatable).
      --reuse-prefix            Keep the KV cache between --chat turns and
                                prefill only what a turn adds. Off by default:
                                every turn re-prefills from a reset cache.
      --show-reasoning          Stream the thoughts of --chat turns to standard
                                error; standard output stays the answer only.
      --max-new <int>           Generated-token limit (default 1024).
      --max-context <int>       Context limit in tokens (default 4096).
      --kv-paged <on|off|auto>  Paged KV cache with SSD spill + Quest sparse
                                decode (Qwen 3.8; default auto: on above 32k
                                context). Exact when everything fits RAM.
      --kv-topk <pages>         Sparse decode budget in 64-token pages
                                (default 60 ≈ 3.8k attended tokens/layer).
      --kv-pool-pages <n|auto>  Resident pool per full-attention layer in
                                pages (default auto: sized from RAM).
      --temperature <float>     Sampling temperature (default \(samplingDefaults.temperature); 0 = greedy).
      --top-k <int>             Top-k truncation, 1...256 (default \(samplingDefaults.topK ?? 0); 0 = off).
      --top-p <float>           Nucleus truncation (default \(samplingDefaults.topP ?? 1)).
      --min-p <float>           Peak-relative cutoff, 0...1 (default \(samplingDefaults.minP)).
      --repetition-penalty <f>  Repetition penalty (default \(samplingDefaults.repetitionPenalty)).
      --repeat-penalty <f>      Alias for --repetition-penalty
      --presence-penalty <f>    Once per seen token, -2...2 (default \(samplingDefaults.presencePenalty)).
      --frequency-penalty <f>   Per token occurrence, -2...2 (default \(samplingDefaults.frequencyPenalty)).
      --repeat-last-n <int>     Penalty window (default \(samplingDefaults.repeatLastN); 0 off, -1 all).
      --seed <uint64>           Deterministic sampling seed (default off).
      --stop <string>           Stop substring (repeatable).
      --rdadvise <mode>         Expert read-ahead advice: off, default,
                                bounded, or adaptive (default off).
      --expert-cache-slots <n|resident|auto>
                                Routed-expert cache slots per layer: 8, 16,
                                24, 32, 64, 96, 128, resident, or auto.
                                resident maps every layer file once and skips
                                the slot cache entirely. auto maps Flash-Next's
                                routed pool on 192 GiB+ hosts when 32 GiB of
                                headroom remains; Qwen 3.6 gets 96 slots on
                                hosts with at least 24 GiB, 32 with at least
                                16 GiB, else 16; other cases get 16.
                                More slots raise the hit rate but use more RAM.
      --prefill-chunk <n|auto>  Prefill chunk tokens (default auto). Larger
                                chunks cut routed-expert re-reads during
                                prompt processing; auto sizes the chunk to
                                the prompt (--chat uses the server's chunk
                                for the model family). Allowed:
                                32, 64, 128, 256, 512, 1024, 2048, 4096.
      --flash-head              Enable Maple's approximate sparse decode head.
                                Prefill remains exact; unsupported models use
                                the exact head.
      --verify <mode>           Model integrity: auto (default) checks file
                                sizes against the receipt written at install
                                time when that receipt validates, and hashes
                                otherwise; full-sha256 re-hashes every
                                routed-expert file on first touch, which for a
                                145 GB expert pool costs ~59 s inside the
                                first prefill; trusted-receipt requires the
                                receipt and fails without it.
      --reasoning-effort <mode>  Gemma 4 / Qwen chat: xhigh, medium, low,
                                or none. Gemma 4 and Qwen 3.6 use binary
                                on/off aliases. Not applied to raw prompts.
      --quiet                   Suppress the timing footer.
      --help                    Show this message.
    """

    public static func parse(_ argv: [String]) throws -> Args {
        var model: String?
        var prompt: String?
        var messagesFile: String?
        var chat = false
        var systemPrompt: String?
        var reusePrefix = false
        var showReasoning = false
        var maxNew = 1_024
        var maxContext = 4096
        // Starting values only: each flag below overwrites its own, so an
        // explicit flag always wins over the shared sampling defaults.
        var providedSamplingOptions: Set<SamplingOption> = []
        var temperature = samplingDefaults.temperature
        var topK = samplingDefaults.topK
        var topP = samplingDefaults.topP
        var repetitionPenalty = samplingDefaults.repetitionPenalty
        var presencePenalty = samplingDefaults.presencePenalty
        var frequencyPenalty = samplingDefaults.frequencyPenalty
        var repeatLastN = samplingDefaults.repeatLastN
        var minP = samplingDefaults.minP
        var seed: UInt64?
        var stops: [String] = []
        var quiet = false
        var expertCacheSlots = ExpertCacheSlotChoice.auto
        var rdadvise = "off"
        var prefillChunk = PrefillChunkChoice.auto
        var flashHead = false
        var verification = ModelIntegrityPolicy.trustedReceiptWhenValid
        var kvPaged = "auto"
        var kvTopKPages = 60
        var kvPoolPages: Int? = nil
        var reasoningEffort: QwenReasoningEffort?

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
            case "--reuse-prefix":
                reusePrefix = true
                index += 1
            case "--show-reasoning":
                showReasoning = true
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
                guard let parsed = Float(value), parsed.isFinite, parsed >= 0 else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                providedSamplingOptions.insert(.temperature)
                temperature = parsed
            case "--reasoning-effort":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = QwenReasoningEffort(rawValue: value) else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                reasoningEffort = parsed
            case "--top-k":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = Int(value), (0...256).contains(parsed) else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                providedSamplingOptions.insert(.topK)
                topK = parsed == 0 ? nil : parsed
            case "--top-p":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = Float(value), parsed > 0, parsed <= 1 else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                providedSamplingOptions.insert(.topP)
                topP = parsed
            case "--min-p":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = Float(value), parsed.isFinite, (0...1).contains(parsed) else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                providedSamplingOptions.insert(.minP)
                minP = parsed
            case "--presence-penalty", "--frequency-penalty":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = Float(value), parsed.isFinite, (-2...2).contains(parsed) else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                if flag == "--presence-penalty" {
                    providedSamplingOptions.insert(.presencePenalty)
                    presencePenalty = parsed
                } else {
                    providedSamplingOptions.insert(.frequencyPenalty)
                    frequencyPenalty = parsed
                }
            case "--repeat-last-n":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = Int(value), parsed >= -1 else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                providedSamplingOptions.insert(.repeatLastN)
                repeatLastN = parsed
            case "--repetition-penalty", "--repeat-penalty":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = Float(value), parsed.isFinite, parsed > 0, (1 / parsed).isFinite else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                providedSamplingOptions.insert(.repetitionPenalty)
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
                guard let policy = ModelIntegrityPolicy(verifyFlag: value) else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                verification = policy
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
        if reasoningEffort != nil && prompt != nil {
            throw ArgsError.invalidValue(flag: "--reasoning-effort", value: "requires chat or messages, not raw --prompt")
        }
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
        if reusePrefix && !chat {
            throw ArgsError.invalidValue(flag: "--reuse-prefix", value: "requires --chat")
        }
        if showReasoning && !chat {
            throw ArgsError.invalidValue(flag: "--show-reasoning", value: "requires --chat")
        }
        if temperature > 0, topK == nil, let topP, topP < 1 {
            throw ArgsError.invalidValue(
                flag: "--top-p",
                value: "\(topP) requires --top-k between 1 and 256")
        }
        var result = Args(model: model,
                    prompt: prompt,
                    messagesFile: messagesFile,
                    chat: chat,
                    systemPrompt: systemPrompt,
                    reusePrefix: reusePrefix,
                    showReasoning: showReasoning,
                    maxNew: maxNew,
                    maxContext: maxContext,
                    temperature: temperature,
                    topK: topK,
                    topP: topP,
                    repetitionPenalty: repetitionPenalty,
                    presencePenalty: presencePenalty,
                    frequencyPenalty: frequencyPenalty,
                    repeatLastN: repeatLastN,
                    minP: minP,
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
                    kvPoolPages: kvPoolPages,
                    reasoningEffort: reasoningEffort)
        result.omittedSamplingOptions = Set(SamplingOption.allCases).subtracting(providedSamplingOptions)
        return result
    }

    /// Resolve after tokenizer/asset validation, before runner/head selection.
    func generationConfig(defaults: GenerationConfig,
                          maxNewTokens: Int) throws -> GenerationConfig {
        var config = defaults
        config.maxNewTokens = maxNewTokens
        config.seed = seed
        config.stopStrings = stops
        if !omittedSamplingOptions.contains(.temperature) { config.temperature = temperature }
        if !omittedSamplingOptions.contains(.topK) { config.topK = topK }
        if !omittedSamplingOptions.contains(.topP) { config.topP = topP }
        if !omittedSamplingOptions.contains(.repetitionPenalty) { config.repetitionPenalty = repetitionPenalty }
        if !omittedSamplingOptions.contains(.presencePenalty) { config.presencePenalty = presencePenalty }
        if !omittedSamplingOptions.contains(.frequencyPenalty) { config.frequencyPenalty = frequencyPenalty }
        if !omittedSamplingOptions.contains(.repeatLastN) { config.repeatLastN = repeatLastN }
        if !omittedSamplingOptions.contains(.minP) { config.minP = minP }
        try config.validate()
        return config
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

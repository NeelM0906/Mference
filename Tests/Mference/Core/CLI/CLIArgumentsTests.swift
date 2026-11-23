import Foundation
import Testing
import Mference
@testable import MferenceCLICore

@Suite struct CLIArgumentsTests {
    @Test func llamaSamplingOptionsReachCLIConfiguration() throws {
        let a = try Args.parse(["--model", "m.gturbo", "--prompt", "hi",
            "--min-p", "0.25", "--presence-penalty", "-0.5",
            "--frequency-penalty", "0.75", "--repeat-last-n", "-1",
            "--repeat-penalty", "1.1"])
        #expect(a.minP == 0.25 && a.presencePenalty == -0.5 && a.frequencyPenalty == 0.75)
        #expect(a.repeatLastN == -1 && a.repetitionPenalty == 1.1)
        for (flag, value) in [("--min-p", "1.1"), ("--min-p", "nan"),
                              ("--frequency-penalty", "2.1"), ("--presence-penalty", "-2.1"),
                              ("--repeat-last-n", "-2"), ("--repeat-penalty", "inf")] {
            #expect(throws: ArgsError.self) {
                _ = try Args.parse(["--model", "m.gturbo", "--prompt", "hi", flag, value])
            }
        }
    }
    @Test func emptyTruncationHasActionableNoticeWithoutChangingSuccessOutput() throws {
        let base = try #require(emptyResponseLimitNotice(reason: .maxTokens,
            hasVisibleText: false))
        #expect(base.contains("--max-new"))
        #expect(base.contains("--max-context"))
        #expect(!base.contains("--reasoning-effort"))
        #expect(emptyResponseLimitNotice(reason: .maxTokens,
            hasVisibleText: true) == nil)
        #expect(emptyResponseLimitNotice(reason: .endOfTurn,
            hasVisibleText: false) == nil)
    }
    @Test func usageListsMaple() {
        #expect(Args.usage.contains("Maple"))
    }

    @Test func defaultsUseProductionGenerationValues() throws {
        let arguments = try Args.parse(["--model", "m.gturbo", "--prompt", "hi"])
        #expect(arguments.model == "m.gturbo")
        #expect(arguments.prompt == "hi")
        #expect(arguments.messagesFile == nil)
        #expect(arguments.maxNew == 1_024)
        #expect(arguments.maxContext == 4096)
        #expect(arguments.temperature == 0.8)
        #expect(arguments.topK == 40)
        #expect(arguments.topP == 0.95)
        #expect(arguments.repetitionPenalty == 1)
        #expect(arguments.minP == 0.05)
        #expect(arguments.presencePenalty == 0)
        #expect(arguments.frequencyPenalty == 0)
        #expect(arguments.repeatLastN == 64)
        #expect(arguments.seed == nil)
        #expect(arguments.stops.isEmpty)
        #expect(!arguments.quiet)
    }

    @Test func omittedSamplingFlagsUseSharedDefaultsAndExplicitFlagsWin() throws {
        let d = GenerationConfig.defaults
        let base = ["--model", "m.gturbo", "--prompt", "hi"]
        for omitted in [try Args.parse(base), Args(model: "m.gturbo")] {
            #expect(omitted.temperature == d.temperature && omitted.topK == d.topK && omitted.topP == d.topP)
            #expect(omitted.minP == d.minP && omitted.repetitionPenalty == d.repetitionPenalty)
            #expect(omitted.presencePenalty == d.presencePenalty && omitted.frequencyPenalty == d.frequencyPenalty)
            #expect(omitted.repeatLastN == d.repeatLastN)
        }

        // Every value differs from its default: a default leaking past an
        // explicit flag fails here.
        let explicit = try Args.parse(base + [
            "--temperature", "0.3", "--top-k", "7", "--top-p", "0.6", "--min-p", "0.2",
            "--repetition-penalty", "1.2", "--presence-penalty", "0.4",
            "--frequency-penalty", "-0.3", "--repeat-last-n", "128"])
        #expect(explicit.temperature == 0.3 && explicit.topK == 7 && explicit.topP == 0.6 && explicit.minP == 0.2)
        #expect(explicit.repetitionPenalty == 1.2 && explicit.presencePenalty == 0.4)
        #expect(explicit.frequencyPenalty == -0.3 && explicit.repeatLastN == 128)

        // An explicit zero is a value, not an omission.
        let zeros = try Args.parse(base + ["--temperature", "0", "--min-p", "0", "--repeat-last-n", "0"])
        #expect(zeros.temperature == 0 && zeros.minP == 0 && zeros.repeatLastN == 0)
    }

    @Test func usageAdvertisesTheSharedSamplingDefaults() {
        let d = GenerationConfig.defaults
        for text in ["(default \(d.temperature); 0 = greedy)", "(default \(d.topK ?? 0); 0 = off)",
                     "Nucleus truncation (default \(d.topP ?? 1))", "0...1 (default \(d.minP))",
                     "(default \(d.repeatLastN); 0 off, -1 all)"] {
            #expect(Args.usage.contains(text), "usage is missing: \(text)")
        }
    }

    @Test func generationOptionsParseAndStopsRepeat() throws {
        let arguments = try Args.parse([
            "--model", "m.gturbo", "--prompt", "hi",
            "--max-new", "32", "--max-context", "512",
            "--temperature", "0", "--top-k", "40", "--top-p", "0.95",
            "--repetition-penalty", "1.1", "--seed", "42",
            "--stop", "A", "--stop", "B", "--quiet",
        ])
        #expect(arguments.maxNew == 32)
        #expect(arguments.maxContext == 512)
        #expect(arguments.temperature == 0)
        #expect(arguments.topK == 40)
        #expect(arguments.topP == 0.95)
        #expect(arguments.repetitionPenalty == 1.1)
        #expect(arguments.seed == 42)
        #expect(arguments.stops == ["A", "B"])
        #expect(arguments.quiet)
    }

    @Test func topKZeroRequiresTopPToBeDisabled() throws {
        let disabled = try Args.parse([
            "--model", "m.gturbo", "--prompt", "hi",
            "--top-k", "0", "--top-p", "1",
        ])
        #expect(disabled.topK == nil)
        #expect(disabled.topP == 1)

        #expect(throws: ArgsError.self) {
            _ = try Args.parse([
                "--model", "m.gturbo", "--prompt", "hi", "--top-k", "0",
            ])
        }
    }

    @Test func topKAboveKernelLimitRejected() {
        #expect(throws: ArgsError.invalidValue(flag: "--top-k", value: "257")) {
            _ = try Args.parse([
                "--model", "m.gturbo", "--prompt", "hi", "--top-k", "257",
            ])
        }
    }

    @Test func helpListsExactlyThePublicOptions() {
        let expected: Set<String> = [
            "--model", "--prompt", "--messages-file", "--chat", "--system", "--reuse-prefix", "--show-reasoning", "--shadow-budget",
            "--max-new", "--max-context",
            "--temperature", "--top-k", "--top-p", "--repetition-penalty",
            "--repeat-penalty", "--min-p", "--presence-penalty", "--frequency-penalty", "--repeat-last-n",
            "--seed", "--stop", "--prefill-chunk", "--quiet", "--help",
            "--rdadvise", "--expert-cache-slots", "--flash-head", "--verify",
            "--kv-paged", "--kv-topk", "--kv-pool-pages", "--kv-reserve", "--reasoning-effort",
        ]
        let words = Args.usage.split { $0.isWhitespace || $0 == "(" || $0 == ")" }
        let options = Set(words.map(String.init).filter { $0.hasPrefix("--") })
        #expect(options == expected)
    }

    @Test func unsupportedSelectorsAreRejected() {
        for flag in ["--runtime-profile", "--experiment-id", "-h"] {
            #expect(throws: ArgsError.unknownFlag(flag)) {
                _ = try Args.parse(["--model", "m.gturbo", "--prompt", "hi", flag])
            }
        }
    }

    @Test func modelAndPromptAreRequired() {
        #expect(throws: ArgsError.requiredMissing("--model")) {
            _ = try Args.parse(["--prompt", "hi"])
        }
        #expect(throws: ArgsError.modeMissing) {
            _ = try Args.parse(["--model", "m.gturbo"])
        }
    }

    @Test func messagesFileSelectsChatMode() throws {
        let arguments = try Args.parse([
            "--model", "m.gturbo", "--messages-file", "chat.json",
        ])
        #expect(arguments.prompt == nil)
        #expect(arguments.messagesFile == "chat.json")
    }

    @Test func promptAndMessagesFileAreMutuallyExclusive() {
        #expect(throws: ArgsError.mutuallyExclusive("--prompt", "--messages-file")) {
            _ = try Args.parse([
                "--model", "m.gturbo", "--prompt", "hi",
                "--messages-file", "chat.json",
            ])
        }
    }

    @Test func chatSelectsInteractiveModeWithoutAPrompt() throws {
        let arguments = try Args.parse(["--model", "m.gturbo", "--chat"])
        #expect(arguments.chat)
        #expect(arguments.prompt == nil)
        #expect(arguments.messagesFile == nil)
        #expect(arguments.systemPrompt == nil)
    }

    @Test func chatAndPromptAreMutuallyExclusive() {
        #expect(throws: ArgsError.mutuallyExclusive("--prompt", "--chat")) {
            _ = try Args.parse(["--model", "m.gturbo", "--prompt", "hi", "--chat"])
        }
    }

    @Test func chatAndMessagesFileAreMutuallyExclusive() {
        #expect(throws: ArgsError.mutuallyExclusive("--messages-file", "--chat")) {
            _ = try Args.parse([
                "--model", "m.gturbo", "--messages-file", "chat.json", "--chat",
            ])
        }
    }

    @Test func repeatedSystemFlagsJoinIntoOneMessage() throws {
        let arguments = try Args.parse([
            "--model", "m.gturbo", "--chat",
            "--system", "be terse", "--system", "answer in English",
        ])
        #expect(arguments.systemPrompt == "be terse\nanswer in English")
    }

    @Test func systemRequiresChatMode() {
        #expect(throws: ArgsError.invalidValue(flag: "--system", value: "requires --chat")) {
            _ = try Args.parse([
                "--model", "m.gturbo", "--prompt", "hi", "--system", "be terse",
            ])
        }
    }

    @Test func chatResetsEveryTurnUnlessPrefixReuseIsRequested() throws {
        let plain = try Args.parse(["--model", "m.gturbo", "--chat"])
        #expect(!plain.reusePrefix)
        let reuse = try Args.parse(["--model", "m.gturbo", "--chat", "--reuse-prefix"])
        #expect(reuse.reusePrefix)
    }

    @Test func reusePrefixRequiresChatMode() {
        #expect(throws: ArgsError.invalidValue(flag: "--reuse-prefix", value: "requires --chat")) {
            _ = try Args.parse(["--model", "m.gturbo", "--prompt", "hi", "--reuse-prefix"])
        }
    }

    @Test func chatHidesReasoningUnlessAskedToShowIt() throws {
        let plain = try Args.parse(["--model", "m.gturbo", "--chat"])
        #expect(!plain.showReasoning)
        let shown = try Args.parse(["--model", "m.gturbo", "--chat", "--show-reasoning"])
        #expect(shown.showReasoning)
    }

    @Test func showReasoningRequiresChatMode() {
        #expect(throws: ArgsError.invalidValue(flag: "--show-reasoning", value: "requires --chat")) {
            _ = try Args.parse(["--model", "m.gturbo", "--prompt", "hi", "--show-reasoning"])
        }
    }

    /// `--reasoning-effort` takes the Qwen 3.8 source template's levels, only
    /// with `--chat` or messages. Inkling 2's other names (minimal, high, max)
    /// parse too, and `run` refuses them for every other dialect
    /// (`otherDialectsRefuseInklingNamesAsParsingRefusesUnknownValues`).
    @Test func reasoningEffortParsesOnlyInChatMode() throws {
        for effort in QwenReasoningEffort.allCases {
            let args = try Args.parse(["--model", "candidate.gturbo", "--chat", "--reasoning-effort", effort.rawValue])
            #expect(args.reasoningEffort == effort)
            #expect(args.inkling2Effort == nil)
        }
        for value in ["low", "high"] {
            #expect(throws: ArgsError.invalidValue(flag: "--reasoning-effort",
                                                   value: "requires chat or messages, not raw --prompt")) {
                try Args.parse(["--model", "candidate.gturbo", "--prompt", "Hi", "--reasoning-effort", value])
            }
        }
        for value in ["extreme", "HIGH", " high", "high ", "0.5", "", "on"] {
            #expect(throws: ArgsError.invalidValue(flag: "--reasoning-effort", value: value), "\(value)") {
                try Args.parse(["--model", "candidate.gturbo", "--chat", "--reasoning-effort", value])
            }
        }
    }

    /// Inkling 2's names come from the template's own table, the one the
    /// server's `reasoning_effort` uses: the shared four stay
    /// `reasoningEffort`, the other three are `inkling2Effort`. The last flag
    /// given wins.
    @Test func inklingEffortNamesParseFromTheSharedTable() throws {
        for name in MFTokenizer.Inkling2Effort.namedLevels.keys {
            let args = try Args.parse(["--model", "m.gturbo", "--chat", "--reasoning-effort", name])
            if let shared = QwenReasoningEffort(rawValue: name) {
                #expect(args.reasoningEffort == shared && args.inkling2Effort == nil, "\(name)")
            } else {
                #expect(args.inkling2Effort == .named(name) && args.reasoningEffort == nil, "\(name)")
            }
        }
        let own = MFTokenizer.Inkling2Effort.namedLevels.keys.filter { QwenReasoningEffort(rawValue: $0) == nil }
        #expect(Set(own) == ["minimal", "high", "max"])
        let lowLast = try Args.parse(["--model", "m.gturbo", "--chat",
                                      "--reasoning-effort", "high", "--reasoning-effort", "low"])
        #expect(lowLast.reasoningEffort == .low && lowLast.inkling2Effort == nil)
        let maxLast = try Args.parse(["--model", "m.gturbo", "--messages-file", "m.json",
                                      "--reasoning-effort", "low", "--reasoning-effort", "max"])
        #expect(maxLast.inkling2Effort == .named("max") && maxLast.reasoningEffort == nil)
    }

    /// Every other dialect refuses minimal, high and max with exactly what
    /// `main` writes for an unknown value at parse time (the error, the
    /// usage, exit 2); Inkling 2 takes them. The shared names are never refused.
    @Test func otherDialectsRefuseInklingNamesAsParsingRefusesUnknownValues() throws {
        func parseFailureText(_ value: String) -> String? {
            do {
                _ = try Args.parse(["--model", "m.gturbo", "--chat", "--reasoning-effort", value])
                return nil
            } catch {
                return "error: \(error)\n\n" + Args.usage + "\n"   // as `main` writes it
            }
        }
        #expect(parseFailureText("extreme") == "error: invalid value for --reasoning-effort: extreme\n\n"
                + Args.usage + "\n")
        for name in ["minimal", "high", "max"] {
            let args = try Args.parse(["--model", "m.gturbo", "--chat", "--reasoning-effort", name])
            let expected = "error: invalid value for --reasoning-effort: \(name)\n\n" + Args.usage + "\n"
            for dialect in [ChatDialect.chatml, .minicpm] {
                #expect(inkling2EffortRefusal(args, dialect: dialect) == expected, "\(name) \(dialect)")
            }
            #expect(inkling2EffortRefusal(args, dialect: .inkling2) == nil, "\(name)")
        }
        for name in ["xhigh", "medium", "low", "none"] {
            let args = try Args.parse(["--model", "m.gturbo", "--chat", "--reasoning-effort", name])
            for dialect in [ChatDialect.chatml, .minicpm, .inkling2] {
                #expect(inkling2EffortRefusal(args, dialect: dialect) == nil, "\(name) \(dialect)")
            }
        }
        #expect(inkling2EffortRefusal(try Args.parse(["--model", "m.gturbo", "--chat"]), dialect: .chatml) == nil)
    }

    /// The help keeps its Qwen lines word for word and adds one for Inkling 2,
    /// aligned with them.
    @Test func reasoningEffortHelpKeepsTheQwenLines() throws {
        let lines = Args.usage.components(separatedBy: "\n")
        let first = try #require(lines.firstIndex { $0.contains("--reasoning-effort <mode>") })
        #expect(lines[first].trimmingCharacters(in: .whitespaces)
                == "--reasoning-effort <mode>  Qwen chat: xhigh, medium, low, or none.")
        let next = Array(lines[(first + 1)...(first + 3)])
        #expect(next.map { $0.trimmingCharacters(in: .whitespaces) }
                == ["Qwen 3.6 uses binary on/off aliases. Not", "applied to raw prompts.",
                    "Inkling 2 also takes minimal, high, or max."])
        let indent = { (line: String) in line.prefix { $0 == " " }.count }
        #expect(Set(next.map(indent)).count == 1)
    }

    @Test func shadowBudgetIsUnsetUnlessGiven() throws {
        #expect(try Args.parse(["--model", "m.gturbo", "--prompt", "hi"]).shadowBudget == nil)
        #expect(try Args.parse(["--model", "m.gturbo", "--prompt", "hi", "--shadow-budget", "4"]).shadowBudget == 4)
        #expect(try Args.parse(["--model", "m.gturbo", "--chat", "--shadow-budget", "0"]).shadowBudget == 0)
    }

    @Test(arguments: ["9", "-1", "two", ""])
    func shadowBudgetRejectsValuesOutsideZeroToEight(value: String) {
        #expect(throws: ArgsError.invalidValue(flag: "--shadow-budget", value: value)) {
            _ = try Args.parse(["--model", "m.gturbo", "--prompt", "hi", "--shadow-budget", value])
        }
    }

    /// `--max-context max` becomes the model's native context once the
    /// install is known; a number stays as given.
    @Test func maxContextMaxResolvesToTheModelsNativeContext() throws {
        let directory = try QwenToySynthetic.write()
        defer { try? FileManager.default.removeItem(at: directory) }
        let parsed = try Args.parse(["--model", directory.path, "--prompt", "hi", "--max-context", "max"])
        #expect(parsed.usesModelMaxContext)
        let resolved = try parsed.resolvingModelMaxContext()
        #expect(resolved.maxContext == 262_144)
        #expect(!resolved.usesModelMaxContext)
        let fixed = try Args.parse(["--model", directory.path, "--prompt", "hi", "--max-context", "8192"])
        #expect(try fixed.resolvingModelMaxContext().maxContext == 8_192)
        #expect(throws: ArgsError.invalidValue(flag: "--max-context", value: "MAX")) {
            _ = try Args.parse(["--model", directory.path, "--prompt", "hi", "--max-context", "MAX"])
        }
    }

    @Test func kvGrowsUnlessReserveIsGiven() throws {
        #expect(try !Args.parse(["--model", "m.gturbo", "--prompt", "hi"]).reserveFullKV)
        #expect(try Args.parse(["--model", "m.gturbo", "--prompt", "hi", "--kv-reserve"]).reserveFullKV)
    }

    @Test func prefillChunkDefaultsToAuto() throws {
        let arguments = try Args.parse(["--model", "m.gturbo", "--prompt", "hi"])
        #expect(arguments.prefillChunk == .auto)
        #expect(!arguments.flashHead)
    }

    @Test func chatAutoChunkFollowsTheFamilyServerChunk() {
        let gib = UInt64(1) << 30
        #expect(PrefillChunkChoice.auto.chatChunkTokens(
            for: .qwen36, physicalMemoryBytes: 16 * gib) == 2048)
        #expect(PrefillChunkChoice.auto.chatChunkTokens(
            for: .maple, physicalMemoryBytes: 16 * gib) == 128)
        #expect(PrefillChunkChoice.auto.chatChunkTokens(
            for: .qwen36, physicalMemoryBytes: 8 * gib) == 128)
    }

    @Test func chatFixedChunkWinsOverTheFamilyDefault() {
        let gib = UInt64(1) << 30
        #expect(PrefillChunkChoice.fixed(256).chatChunkTokens(
            for: .qwen36, physicalMemoryBytes: 16 * gib) == 256)
    }

    @Test func flashHeadRequiresExplicitOptIn() throws {
        let arguments = try Args.parse([
            "--model", "m.gturbo", "--prompt", "hi", "--flash-head",
        ])
        #expect(arguments.flashHead)
    }

    @Test func expertCacheSlotsDefaultToModelAwareAuto() throws {
        let arguments = try Args.parse(["--model", "m.gturbo", "--prompt", "hi"])
        #expect(arguments.expertCacheSlots == .auto)
    }

    @Test func expertCacheSlotsAcceptAutoAndExplicitOverride() throws {
        let automatic = try Args.parse([
            "--model", "m.gturbo", "--prompt", "hi",
            "--expert-cache-slots", "auto",
        ])
        let constrained = try Args.parse([
            "--model", "m.gturbo", "--prompt", "hi",
            "--expert-cache-slots", "16",
        ])
        #expect(automatic.expertCacheSlots == .auto)
        #expect(constrained.expertCacheSlots == .fixed(16))
    }

    @Test(arguments: [32, 64, 128, 256, 512, 1024, 2048, 4096])
    func prefillChunkAcceptsAllowedSizes(size: Int) throws {
        let arguments = try Args.parse([
            "--model", "m.gturbo", "--prompt", "hi",
            "--prefill-chunk", String(size),
        ])
        #expect(arguments.prefillChunk == .fixed(size))
    }

    @Test func prefillChunkAcceptsAuto() throws {
        let arguments = try Args.parse([
            "--model", "m.gturbo", "--prompt", "hi",
            "--prefill-chunk", "auto",
        ])
        #expect(arguments.prefillChunk == .auto)
    }

    @Test(arguments: ["0", "100", "8192", "-128", "big"])
    func prefillChunkRejectsDisallowedValues(value: String) {
        #expect(throws: ArgsError.invalidValue(flag: "--prefill-chunk",
                                               value: value)) {
            _ = try Args.parse([
                "--model", "m.gturbo", "--prompt", "hi",
                "--prefill-chunk", value,
            ])
        }
    }

    /// Verification defaults to the strict policy: skipping the per-expert
    /// hashes is a deliberate opt-in, not something a caller inherits.
    @Test func verificationDefaultsToTheInstallReceiptWhenItIsValid() throws {
        let arguments = try Args.parse(["--model", "m.gturbo", "--prompt", "hi"])
        #expect(arguments.verification == .trustedReceiptWhenValid)
    }

    @Test func verifyAcceptsAuto() throws {
        let arguments = try Args.parse(["--model", "m.gturbo", "--prompt", "hi", "--verify", "auto"])
        #expect(arguments.verification == .trustedReceiptWhenValid)
    }

    @Test func verifyAcceptsFullSha256() throws {
        let arguments = try Args.parse([
            "--model", "m.gturbo", "--prompt", "hi", "--verify", "full-sha256",
        ])
        #expect(arguments.verification == .fullSha256)
    }

    @Test func verifyAcceptsTrustedReceipt() throws {
        let arguments = try Args.parse([
            "--model", "m.gturbo", "--prompt", "hi",
            "--verify", "trusted-receipt",
        ])
        #expect(arguments.verification == .sizeCheckTrustedReceipt)
    }

    @Test(arguments: ["", "none", "sha", "trusted", "full"])
    func verifyRejectsUnknownPolicies(value: String) {
        #expect(throws: ArgsError.invalidValue(flag: "--verify", value: value)) {
            _ = try Args.parse([
                "--model", "m.gturbo", "--prompt", "hi", "--verify", value,
            ])
        }
    }
}

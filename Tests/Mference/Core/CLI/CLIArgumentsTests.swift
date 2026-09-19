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
        let swift = try #require(emptyResponseLimitNotice(reason: .maxTokens,
            hasVisibleText: false, isSwiftQwen: true))
        #expect(swift.contains("--max-new"))
        #expect(swift.contains("--max-context"))
        #expect(swift.contains("--reasoning-effort medium or low"))
        let base = try #require(emptyResponseLimitNotice(reason: .maxTokens,
            hasVisibleText: false, isSwiftQwen: false))
        #expect(!base.contains("--reasoning-effort"))
        #expect(emptyResponseLimitNotice(reason: .maxTokens,
            hasVisibleText: true, isSwiftQwen: true) == nil)
        #expect(emptyResponseLimitNotice(reason: .endOfTurn,
            hasVisibleText: false, isSwiftQwen: true) == nil)
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
            "--model", "--prompt", "--messages-file", "--chat", "--system",
            "--max-new", "--max-context",
            "--temperature", "--top-k", "--top-p", "--repetition-penalty",
            "--repeat-penalty", "--min-p", "--presence-penalty", "--frequency-penalty", "--repeat-last-n",
            "--seed", "--stop", "--prefill-chunk", "--quiet", "--help",
            "--rdadvise", "--expert-cache-slots", "--flash-head", "--verify",
            "--kv-paged", "--kv-topk", "--kv-pool-pages", "--reasoning-effort",
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

    @Test func prefillChunkDefaultsToAuto() throws {
        let arguments = try Args.parse(["--model", "m.gturbo", "--prompt", "hi"])
        #expect(arguments.prefillChunk == .auto)
        #expect(!arguments.flashHead)
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
    @Test func verificationDefaultsToFullSha256() throws {
        let arguments = try Args.parse(["--model", "m.gturbo", "--prompt", "hi"])
        #expect(arguments.verification == .fullSha256)
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

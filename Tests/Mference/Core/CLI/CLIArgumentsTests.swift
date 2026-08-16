import Testing
import Mference
@testable import MferenceCLICore

@Suite struct CLIArgumentsTests {
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
        #expect(arguments.temperature == 0.2)
        #expect(arguments.topK == 64)
        #expect(arguments.topP == 0.95)
        #expect(arguments.repetitionPenalty == 1)
        #expect(arguments.seed == nil)
        #expect(arguments.stops.isEmpty)
        #expect(!arguments.quiet)
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
            "--seed", "--stop", "--prefill-chunk", "--quiet", "--help",
            "--rdadvise", "--expert-cache-slots", "--flash-head", "--verify",
            "--kv-paged", "--kv-topk", "--kv-pool-pages",
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

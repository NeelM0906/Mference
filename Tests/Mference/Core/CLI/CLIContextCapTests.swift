import Foundation
import Testing
@testable import Mference
@testable import MferenceCLICore
@testable import MferenceRepackCore

/// `MferenceCLI --max-context` under the Inkling 2 host cap: on a Mac whose
/// expert residency plan stages layers (under 48 GB), Inkling 2 Mini runs at
/// most `Inkling2MemoryFootprint.stagedHostContextCap` tokens, `max`
/// included; every other family resolves exactly as before the cap (the
/// byte-identity runs use this path).
@Suite struct CLIContextCapTests {
    /// The default, explicit values below, at, between and above the cap and
    /// the native contexts, and `max`.
    static let settings: [String?] = [nil, "4096", "16384", "40960", "40961", "65536", "131072", "200000",
                                      "262144", "max"]
    static let memories: [UInt64] = [8 << 30, 16 << 30, 32 << 30, 36 << 30, 47 << 30, 48 << 30, 64 << 30]

    static func parse(_ setting: String?) throws -> Args {
        var argv = ["--model", "m.gturbo", "--prompt", "hi"]
        if let setting { argv += ["--max-context", setting] }
        return try Args.parse(argv)
    }

    /// The context before the cap: the given value, 4,096 by default, or the
    /// native context for `max`.
    static func before(_ setting: String?, family: ModelFamily) -> Int {
        switch setting {
        case nil: 4_096
        case "max": family.maximumContext
        case let value?: Int(value)!
        }
    }

    @Test func existingFamiliesResolveAsBefore() throws {
        for family in [ModelFamily.qwen36, .qwen38, .minicpm5, .maple] {
            for memory in Self.memories {
                for setting in Self.settings {
                    let resolved = try Self.parse(setting).resolvingMaxContext(family: family, physicalMemory: memory)
                    #expect(resolved.maxContext == Self.before(setting, family: family),
                            "\(family) \(setting ?? "default") at \(memory >> 30) GB")
                    #expect(resolved.hostContextCap == nil && !resolved.usesModelMaxContext)
                }
            }
        }
        // An explicit value with no readable family stays as given.
        #expect(try Self.parse("200000").resolvingMaxContext(family: nil, physicalMemory: 32 << 30).maxContext
                == 200_000)
    }

    /// Through the real peek: an explicit value with a model whose manifest
    /// cannot be read resolves as given, without throwing (the load reports
    /// it later, as before); `max` still needs the manifest, as before.
    @Test func anUnreadableModelFailsWhereItDidBefore() throws {
        let missing = "/nonexistent-\(UUID().uuidString)/m.gturbo"
        for setting in ["4096", "65536", "262144"] {
            let args = try Args.parse(["--model", missing, "--prompt", "hi", "--max-context", setting])
            let resolved = try args.resolvingModelMaxContext(physicalMemory: 32 << 30)
            #expect(resolved.maxContext == Int(setting)! && resolved.hostContextCap == nil)
        }
        let byDefault = try Args.parse(["--model", missing, "--prompt", "hi"])
        #expect(try byDefault.resolvingModelMaxContext(physicalMemory: 32 << 30).maxContext == 4_096)
        let max = try Args.parse(["--model", missing, "--prompt", "hi", "--max-context", "max"])
        #expect(throws: (any Error).self) { _ = try max.resolvingModelMaxContext(physicalMemory: 32 << 30) }
    }

    @Test func inkling2IsCappedOnlyWhereLayersAreStaged() throws {
        let cap = Inkling2MemoryFootprint.stagedHostContextCap
        #expect(cap == 40_960)
        for memory in Self.memories {
            let staged = memory < 48 << 30
            for setting in Self.settings {
                let asked = Self.before(setting, family: .inkling2Loop)
                let resolved = try Self.parse(setting).resolvingMaxContext(family: .inkling2Loop,
                                                                           physicalMemory: memory)
                let label = "\(setting ?? "default") at \(memory >> 30) GB"
                #expect(resolved.maxContext == (staged ? min(asked, cap) : asked), "\(label)")
                #expect(resolved.hostContextCap == (staged && asked > cap ? cap : nil), "\(label)")
            }
        }
    }

    /// Through the manifest: a qwen36 install is untouched on a 32 GB Mac; an
    /// Inkling 2 install is capped there and not at 48 GB, `max` and
    /// explicit alike.
    @Test func throughTheManifest() async throws {
        let qwen = try QwenToySynthetic.write()
        defer { try? FileManager.default.removeItem(at: qwen) }
        for setting in ["max", "65536", "8192"] {
            let args = try Args.parse(["--model", qwen.path, "--prompt", "hi", "--max-context", setting])
            let resolved = try args.resolvingModelMaxContext(physicalMemory: 32 << 30)
            #expect(resolved.maxContext == (setting == "max" ? 262_144 : Int(setting)!))
            #expect(resolved.hostContextCap == nil)
        }

        let gguf = try Inkling2InstallerTests.writeToy()
        let output = Inkling2InstallerTests.tmp("cli-cap") + ".gturbo"
        defer {
            try? FileManager.default.removeItem(atPath: output)
            try? FileManager.default.removeItem(atPath: (gguf.path as NSString).deletingLastPathComponent)
        }
        _ = try await Inkling2InstallerTests.install(gguf, output: output,
                                                     pin: Inkling2InstallerTests.pin(for: gguf.data))
        for setting in ["max", "65536"] {
            let args = try Args.parse(["--model", output, "--prompt", "hi", "--max-context", setting])
            #expect(try args.resolvingModelMaxContext(physicalMemory: 32 << 30).maxContext == 40_960)
            #expect(try args.resolvingModelMaxContext(physicalMemory: 48 << 30).maxContext
                    == (setting == "max" ? 262_144 : 65_536))
        }
        let small = try Args.parse(["--model", output, "--prompt", "hi", "--max-context", "8192"])
        #expect(try small.resolvingModelMaxContext(physicalMemory: 32 << 30).maxContext == 8_192)
    }
}

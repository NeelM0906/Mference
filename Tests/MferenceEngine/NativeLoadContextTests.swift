import Foundation
import Testing
import Mference
@testable import MferenceEngine

/// The context a native load runs with. Every existing family loads exactly
/// as before the Inkling 2 host cap (`options.nativeMaxContext ??
/// family.maximumContext`) at the default, explicit and `max` settings, on
/// every host size; Inkling 2 is capped where its residency plan stages
/// layers, whoever built the descriptor.
@Suite struct NativeLoadContextTests {
    static let existing: [ModelFamily] = [.qwen36, .qwen38, .minicpm5, .maple]
    /// The default, explicit values below, between and above the families'
    /// native contexts, and `max` (nil).
    static let settings: [Int?] = [EngineLoadOptions.defaultNativeMaxContext, 4_096, 32_768, 40_960, 65_536,
                                   131_072, 200_000, 262_144, nil]
    static let memories: [UInt64] = [8 << 30, 16 << 30, 24 << 30, 32 << 30, 36 << 30, 47 << 30, 48 << 30, 64 << 30]

    /// The descriptor discovery builds for one install of `family`.
    static func discovered(_ family: ModelFamily, memory: UInt64) throws -> EngineModelDescriptor {
        let install = URL(fileURLWithPath: "/lib/x.gturbo", isDirectory: true)
        let disk = EngineCatalogDiscovery.FileSystem(
            childDirectories: { _ in [install] },
            ggufFiles: { _ in [] },
            probe: { $0.standardizedFileURL.path == install.path ? .complete(family: family) : .notAModelDirectory },
            checkpointID: { _ in nil },
            readGGUF: { _ in throw EngineGGUFReader.ReadError.notGGUF },
            size: { _ in 1 })
        let catalog = EngineCatalogDiscovery.discover(roots: [URL(fileURLWithPath: "/lib", isDirectory: true)],
                                                      ggufDirectories: [], scope: .all, fileSystem: disk,
                                                      physicalMemory: memory)
        return try #require(catalog.models.first, "\(family) at \(memory >> 30) GB")
    }

    @Test func existingFamiliesLoadWithTheContextTheyHadBefore() throws {
        for family in Self.existing {
            for memory in Self.memories {
                let descriptor = try Self.discovered(family, memory: memory)
                #expect(descriptor.hostContextCap == nil)
                #expect(family.memoryContextCap(physicalMemory: memory) == nil)
                for setting in Self.settings {
                    let options = EngineLoadOptions(nativeMaxContext: setting)
                    let before = setting ?? family.maximumContext
                    let label = "\(family) \(setting.map(String.init) ?? "max") at \(memory >> 30) GB"
                    #expect(NativeEngineModel.loadContext(descriptor: descriptor, options: options, family: family,
                                                          physicalMemory: memory) == before, "\(label)")
                    #expect(descriptor.resolvedContext(options: options) == before, "\(label)")
                }
            }
        }
    }

    @Test func inkling2IsCappedWhereLayersAreStagedWhoeverBuiltTheDescriptor() throws {
        let cap = Inkling2MemoryFootprint.stagedHostContextCap
        #expect(cap == 40_960)
        // A descriptor not built by discovery (no host cap of its own).
        let bare = EngineModelDescriptor(id: "inkling", displayName: "Inkling", engine: .mference,
                                         format: .gturbo(family: .inkling2Loop, checkpointID: nil),
                                         location: URL(fileURLWithPath: "/lib/inkling2mini.gturbo"),
                                         sizeBytes: 1, contextLength: 262_144, product: .inkling2Mini)
        for memory in Self.memories {
            let staged = memory < 48 << 30
            // Discovery lists the install only from 32 GB (the slot's floor).
            var descriptors = [bare]
            if memory >= 32 << 30 {
                let discovered = try Self.discovered(.inkling2Loop, memory: memory)
                #expect(discovered.hostContextCap == (staged ? cap : nil))
                descriptors.append(discovered)
            }
            for setting in Self.settings {
                let options = EngineLoadOptions(nativeMaxContext: setting)
                let asked = setting ?? 262_144
                let expected = staged ? min(asked, cap) : asked
                for descriptor in descriptors {
                    #expect(NativeEngineModel.loadContext(descriptor: descriptor, options: options,
                                                          family: .inkling2Loop, physicalMemory: memory) == expected)
                }
            }
        }
    }
}

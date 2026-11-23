import Foundation
import Testing
@testable import Mference

/// The memory figures the product shows for Inkling 2 Mini, from the
/// runtime's own residency plan and the installer's blob strides.
@Suite struct Inkling2MemoryFootprintTests {
    /// The verified install's bytes (`--verify-install`, 31 files).
    static let installBytes: UInt64 = 17_662_322_836
    static let pool: UInt64 = 15_829_303_296

    @Test func blobStridesAreTheInstallersOwn() {
        #expect(Inkling2ExpertLayout.blobStride(format: .nvfp4, config: .inkling2Mini) == 4_489_216)
        #expect(Inkling2ExpertLayout.blobStride(format: .q8_0, config: .inkling2Mini) == 8_470_528)
        let layers = Inkling2MemoryFootprint.expertLayerBytes(config: .inkling2Mini)
        #expect(layers?.count == 24)
        #expect(layers?.reduce(0, +) == Self.pool)
        #expect(Inkling2MemoryFootprint.expertLayerBytes(config: .maplePreview) == nil)
        #expect(Inkling2MemoryFootprint.estimate(installBytes: 1, physicalMemory: 64 << 30, config: .maplePreview) == nil)
    }

    @Test func footprintFollowsTheResidencyPlan() throws {
        let nvfp4Layer: UInt64 = 128 * 4_489_216
        let dense = Self.installBytes - Self.pool
        let ink = ArchConfig.inkling2Mini.inkling2
        let layerBytes = try #require(Inkling2MemoryFootprint.expertLayerBytes(config: .inkling2Mini)).map { Int($0) }
        // Coordinator decision 6 (2026-11-23): below 48 GB three staging slabs take what the 40%
        // expert guard leaves beside the resident layers (filled with decode slots), so the staging
        // bytes depend on the host: about 6.84 GB at 32 GB, 8.55 at 36, 9.95 at 47 (where the slots
        // reach every expert of a layer). 30 to 47 GB: the looped core (layers 6-17, NVFP4)
        // resident, the other 12 layers staged.
        for gigabytes: UInt64 in [32, 36, 47] {
            let memory = gigabytes << 30
            let f = try #require(Inkling2MemoryFootprint.estimate(installBytes: Self.installBytes,
                                                                  physicalMemory: memory))
            let plan = Inkling2ExpertResidencyPlan.make(layerBytes: layerBytes,
                                                        loopedLayers: ink.loopCoreStart..<ink.loopCoreEnd,
                                                        physicalMemory: memory)
            #expect(f.expertResidentBytes == 12 * nvfp4Layer)
            #expect(plan.stagingSlabs == 3 && f.expertStagingBytes == UInt64(plan.stagingBytes))
            #expect(f.expertResidentBytes + f.expertStagingBytes
                    <= UInt64(Inkling2ExpertResidencyPlan.expertBudget(physicalMemory: memory)), "\(gigabytes) GB")
            #expect(f.denseBytes == dense)
            #expect(f.bytesBeforeContext
                    == f.expertResidentBytes + f.expertStagingBytes + dense + Inkling2MemoryFootprint.stateBytes)
        }
        // 32 GB in figures: 6.90 GB resident plus 3 x 2.28 GB of slab.
        let at32 = try #require(Inkling2MemoryFootprint.estimate(installBytes: Self.installBytes, physicalMemory: 32 << 30))
        #expect(at32.expertStagingBytes > 6_800_000_000 && at32.expertStagingBytes < 6_860_000_000)
        // 48 GB and up: every layer resident, nothing staged.
        let big = try #require(Inkling2MemoryFootprint.estimate(installBytes: Self.installBytes,
                                                                physicalMemory: 48 << 30))
        #expect(big.expertResidentBytes == Self.pool && big.expertStagingBytes == 0)
        #expect(big.bytesBeforeContext == Self.installBytes + Inkling2MemoryFootprint.stateBytes)
        // Below 30 GB (not offered): everything staged, the slabs at the guard.
        let small = try #require(Inkling2MemoryFootprint.estimate(installBytes: Self.installBytes,
                                                                  physicalMemory: 24 << 30))
        #expect(small.expertResidentBytes == 0 && small.expertStagingBytes > 0
                && small.expertStagingBytes <= UInt64(Inkling2ExpertResidencyPlan.expertBudget(physicalMemory: 24 << 30)))
    }

    /// Global KV: 8 visits of a global layer per token (5 and 23 once, 11
    /// and 17 in each of the three core passes), 8 KV heads of 128, FP16 K
    /// and V: 32 KiB a token, 1 GiB at 32K.
    @Test func globalKVIsThirtyTwoKiBPerToken() throws {
        let f = try #require(Inkling2MemoryFootprint.estimate(installBytes: Self.installBytes,
                                                              physicalMemory: 32 << 30))
        #expect(f.globalKVBytesPerToken == 32_768)
        #expect(f.bytes(contextTokens: 32_768) - f.bytesBeforeContext == 1 << 30)
        #expect(f.bytes(contextTokens: -1) == f.bytesBeforeContext)
    }

    /// 16,384 tokens on hosts where the plan stages layers (under 48 GB),
    /// none where every layer is resident; no other family has a host cap.
    @Test func contextIsCappedWhereLayersAreStaged() {
        for gigabytes: UInt64 in [24, 32, 36, 47] {
            #expect(Inkling2MemoryFootprint.contextCap(physicalMemory: gigabytes << 30) == 16_384)
            #expect(ModelFamily.inkling2Loop.memoryContextCap(physicalMemory: gigabytes << 30) == 16_384)
        }
        for gigabytes: UInt64 in [48, 64, 128] {
            #expect(Inkling2MemoryFootprint.contextCap(physicalMemory: gigabytes << 30) == nil)
        }
        for family in ModelFamily.allCases where family != .inkling2Loop {
            #expect(family.memoryContextCap(physicalMemory: 8 << 30) == nil)
        }
    }

    /// The footprint is the residency plan's own output, slab count and sizes
    /// included, for every host size: a change to `make` moves it with no
    /// edit here.
    @Test func footprintIsTheResidencyPlansOwnOutput() throws {
        let layerBytes = try #require(Inkling2MemoryFootprint.expertLayerBytes(config: .inkling2Mini))
        let ink = ArchConfig.inkling2Mini.inkling2
        for gigabytes: UInt64 in [16, 24, 30, 32, 36, 47, 48, 64, 128] {
            let memory = gigabytes << 30
            let plan = Inkling2ExpertResidencyPlan.make(layerBytes: layerBytes.map { Int($0) },
                                                        loopedLayers: ink.loopCoreStart..<ink.loopCoreEnd,
                                                        physicalMemory: memory)
            let f = try #require(Inkling2MemoryFootprint.estimate(installBytes: Self.installBytes,
                                                                  physicalMemory: memory))
            #expect(f.expertResidentBytes == UInt64(plan.residentBytes), "\(gigabytes) GB")
            #expect(f.expertStagingBytes == UInt64(plan.stagingBytes), "\(gigabytes) GB")
            #expect((f.expertStagingBytes == 0) == (plan.stagingSlabs == 0), "\(gigabytes) GB")
        }
    }

    /// Nothing in the footprint assumes cached reads: wherever the plan
    /// stages a layer, the runtime's default read mode
    /// (`Inkling2StagedReadMode.forHost`, which every staged reader uses
    /// unless a caller overrides it) is uncached, so the staged files' pages
    /// are rightly not counted; where it reads through the cache, nothing is
    /// staged.
    @Test func stagedLayersAreReadUncachedWhereverThePlanStages() throws {
        let layerBytes = try #require(Inkling2MemoryFootprint.expertLayerBytes(config: .inkling2Mini))
        let ink = ArchConfig.inkling2Mini.inkling2
        for gigabytes: UInt64 in [16, 24, 30, 32, 36, 47, 48, 64, 128] {
            let memory = gigabytes << 30
            let plan = Inkling2ExpertResidencyPlan.make(layerBytes: layerBytes.map { Int($0) },
                                                        loopedLayers: ink.loopCoreStart..<ink.loopCoreEnd,
                                                        physicalMemory: memory)
            let stages = plan.placements.contains(.staged)
            switch Inkling2StagedReadMode.forHost(physicalMemory: memory) {
            case .uncached: break
            case .cached: #expect(!stages, "\(gigabytes) GB: staged layers read through the file cache")
            }
        }
    }
}

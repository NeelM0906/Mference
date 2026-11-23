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
        let nvfp4Layer: UInt64 = 128 * 4_489_216, q8Layer: UInt64 = 128 * 8_470_528
        let dense = Self.installBytes - Self.pool
        // 30 to 47 GB (the 32 and 36 GB Macs): the looped core (layers 6-17, NVFP4) resident, two
        // slabs the size of the largest staged layer (Q8_0).
        for gigabytes: UInt64 in [32, 36, 47] {
            let f = try #require(Inkling2MemoryFootprint.estimate(installBytes: Self.installBytes,
                                                                  physicalMemory: gigabytes << 30))
            #expect(f.expertResidentBytes == 12 * nvfp4Layer)
            #expect(f.expertStagingBytes == 2 * q8Layer)
            #expect(f.denseBytes == dense)
            #expect(f.bytesBeforeContext > 12_100_000_000 && f.bytesBeforeContext < 12_400_000_000)
        }
        // 48 GB and up: every layer resident, nothing staged.
        let big = try #require(Inkling2MemoryFootprint.estimate(installBytes: Self.installBytes,
                                                                physicalMemory: 48 << 30))
        #expect(big.expertResidentBytes == Self.pool && big.expertStagingBytes == 0)
        #expect(big.bytesBeforeContext == Self.installBytes + Inkling2MemoryFootprint.stateBytes)
        // Below 30 GB (not offered): everything staged.
        let small = try #require(Inkling2MemoryFootprint.estimate(installBytes: Self.installBytes,
                                                                  physicalMemory: 24 << 30))
        #expect(small.expertResidentBytes == 0 && small.expertStagingBytes == 2 * q8Layer)
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

    /// 40,960 tokens on hosts where the plan stages layers (under 48 GB),
    /// none where every layer is resident; no other family has a host cap.
    @Test func contextIsCappedWhereLayersAreStaged() {
        for gigabytes: UInt64 in [24, 32, 36, 47] {
            #expect(Inkling2MemoryFootprint.contextCap(physicalMemory: gigabytes << 30) == 40_960)
            #expect(ModelFamily.inkling2Loop.memoryContextCap(physicalMemory: gigabytes << 30) == 40_960)
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

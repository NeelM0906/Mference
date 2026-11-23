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
}

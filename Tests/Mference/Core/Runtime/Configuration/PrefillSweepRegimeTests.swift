import Foundation
import Testing
@testable import Mference

/// The regime rule for unbuffered prefill sweeps. Sizes are the pinned
/// Qwen 3.6 install's: 40 routed layers of 256 experts at a 1,769,472-byte
/// stride, 96 slots per layer at 24 GiB and above, and a 1,389,476,096-byte
/// `model_weights.bin`.
@Suite struct PrefillSweepRegimeTests {

    static let gib = UInt64(1) << 30
    static let stride = UInt64(1_769_472)
    static let qwenPool = 40 * 256 * stride            // 18,119,393,280
    static let qwenCore = UInt64(1_389_476_096)
    static func qwenSlots(_ slots: UInt64) -> UInt64 { slots * 40 * stride }

    static func rule(_ family: ModelFamily, physical: UInt64, slots: UInt64 = 96) -> Bool {
        RuntimeConfiguration.prefillSweepBypassesPageCache(
            for: family,
            physicalMemoryBytes: physical,
            expertPoolBytes: qwenPool,
            wiredSlotBytes: qwenSlots(slots),
            coreWeightsBytes: qwenCore)
    }

    @Test func reserveIsTwelveGiB() {
        #expect(RuntimeConfiguration.prefillSweepReserveBytes == 12 * Self.gib)
    }

    /// The measured host: 32 GiB, where the pool did not stay cached.
    @Test func qwenOn32GiBBypasses() {
        #expect(Self.rule(.qwen36, physical: 32 * Self.gib))
    }

    @Test func qwenOnSmallerHostsBypasses() {
        #expect(Self.rule(.qwen36, physical: 24 * Self.gib))
        #expect(Self.rule(.qwen36, physical: 16 * Self.gib, slots: 32))
    }

    /// Pool, slots, core and reserve fit: sweeps stay buffered, as on main.
    @Test func qwenWherePoolFitsStaysBuffered() {
        #expect(!Self.rule(.qwen36, physical: 48 * Self.gib))
        #expect(!Self.rule(.qwen36, physical: 64 * Self.gib))
        #expect(!Self.rule(.qwen36, physical: 128 * Self.gib))
    }

    @Test func boundaryIsStrict() {
        let required = Self.qwenPool + Self.qwenSlots(96) + Self.qwenCore
            + RuntimeConfiguration.prefillSweepReserveBytes
        #expect(!Self.rule(.qwen36, physical: required))
        #expect(Self.rule(.qwen36, physical: required - 1))
    }

    @Test(arguments: [ModelFamily.qwen38, .maple, .minicpm5])
    func otherFamiliesNeverBypass(family: ModelFamily) {
        #expect(!Self.rule(family, physical: 8 * Self.gib))
        #expect(!Self.rule(family, physical: 32 * Self.gib))
        #expect(!RuntimeConfiguration.prefillSweepBypassesPageCache(
            for: family,
            physicalMemoryBytes: 32 * Self.gib,
            expertPoolBytes: .max,
            wiredSlotBytes: .max,
            coreWeightsBytes: .max))
    }

    @Test func overflowingSumBypasses() {
        #expect(RuntimeConfiguration.prefillSweepBypassesPageCache(
            for: .qwen36,
            physicalMemoryBytes: .max,
            expertPoolBytes: .max,
            wiredSlotBytes: 1,
            coreWeightsBytes: 0))
    }

    /// `Model` feeds the rule from the layout: file bytes for the pool, page-
    /// rounded slots for routed layers only, and nothing outside `.pread`.
    @Test func modelInputsFromLayout() {
        let experts = (0..<256).map {
            ExpertEntry(expert: $0, offset: UInt64($0) * Self.stride,
                        size: Self.stride, subTensors: [:])
        }
        let layers = (0..<41).map { index in
            // A leading dense layer: no experts, no slots.
            LayerLayout(layer: index, file: String(format: "layer_%02d.bin", index),
                        experts: index == 0 ? [] : experts)
        }
        let layout = PackedExpertsLayout(expertStride: Self.stride, numLayers: 41,
                                         expertsPerLayer: 256, layers: layers)
        func rule(_ family: ModelFamily, _ mode: ExpertStreamingMode, _ physical: UInt64) -> Bool {
            Model.prefillSweepBypassesPageCache(family: family,
                                                streamingMode: mode,
                                                layout: layout,
                                                coreWeightsBytes: Self.qwenCore,
                                                physicalMemoryBytes: physical)
        }
        #expect(rule(.qwen36, .pread(slotCount: 96), 32 * Self.gib))
        #expect(!rule(.qwen36, .pread(slotCount: 96), 64 * Self.gib))
        #expect(!rule(.qwen36, .resident, 8 * Self.gib))
        #expect(!rule(.maple, .pread(slotCount: 16), 8 * Self.gib))

        // Exactly the 40 routed layers count: at the strict boundary the
        // dense layer adding a 41st slot set would flip the answer.
        let required = Self.qwenPool + Self.qwenSlots(96) + Self.qwenCore
            + RuntimeConfiguration.prefillSweepReserveBytes
        #expect(!rule(.qwen36, .pread(slotCount: 96), required))
        #expect(rule(.qwen36, .pread(slotCount: 96), required - 1))
    }
}

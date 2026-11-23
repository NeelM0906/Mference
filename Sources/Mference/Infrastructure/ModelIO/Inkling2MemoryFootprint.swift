import Foundation

/// What an Inkling 2 install keeps in memory on a host, known before it is
/// loaded, so the product's memory figures and the runtime cannot disagree.
///
/// The routed experts follow the runtime's own residency plan
/// (`Inkling2ExpertResidencyPlan.make`: its host-memory thresholds, which
/// layers it holds, its staging slabs), applied to the per-layer expert file
/// sizes the installer writes for this baseline. The dense weights are the
/// rest of the install. The global-attention KV per token follows from the
/// baseline's layer visits.
public struct Inkling2MemoryFootprint: Sendable, Equatable {
    /// Routed experts read once and kept (the looped core's layers on 30 to
    /// 47 GB hosts, every layer from 48 GB).
    public let expertResidentBytes: UInt64
    /// Staging slabs for the layers the plan does not hold. Below 48 GB that
    /// is three slabs filling what the runner's 40% expert guard
    /// (`Inkling2ExpertResidencyPlan.expertBudget`) leaves beside the resident
    /// layers, carved into decode slots for the staged layers; prefill stages
    /// whole layers through them (the coordinator's three-slab ruling of
    /// 2026-11-23, K2's hand-in 06). At
    /// 32 GB that is 6.84 GB: 88 decode slots for four staged layers and 104
    /// for the other eight, 13.74 GB of experts bound with the resident ones. The staged layers' files are not
    /// counted: wherever the plan stages, the runtime reads them uncached
    /// (`Inkling2StagedReadMode.forHost`), so their pages do not stay in the
    /// file cache.
    public let expertStagingBytes: UInt64
    /// `model_weights.bin` and the sidecars: the install less its expert pool.
    public let denseBytes: UInt64
    /// Sliding-window KV rings and runner and MoE scratch at prefill chunk
    /// 2,048, independent of context (`stateBytes`).
    public let fixedStateBytes: UInt64
    /// Global-attention KV per context token, FP16 keys and values on every
    /// visit of a global layer.
    public let globalKVBytesPerToken: UInt64

    /// K3's sliding-window rings (480 MiB) and the runner's and MoE's scratch
    /// at chunk 2,048 (about 800 MiB), as K3 and K4 sized them on 2026-11-23.
    /// Chunk 2,048 is Inkling 2's prefill chunk on every path
    /// (`RuntimeConfiguration.familyPrefillChunkTokens`).
    ///
    /// Against the measured footprint (K2, 2026-11-23, the three-slab layout
    /// on a 32 GB Mac): 15.3 GB after prefilling a 4K-token prompt at chunk
    /// 2,048 (`phys_footprint` logged after each prefill), and a 14.2 GB peak
    /// decoding. Neither counts the mapped `model_weights.bin` (1.82 GB,
    /// file-backed and clean). This estimate on that Mac is 16.92 GB before
    /// context and 17.05 GB at a 4,096-token context; less its dense term
    /// (1.83 GB: the weights file and the sidecars), 15.22 GB. That is within 0.1 GB of the prefill peak
    /// and about 1.0 GB above the decode peak, which does not touch the chunk
    /// scratch. The app shows it at 32K tokens (`EngineProductSlot
    /// .estimateContextTokens`), 17.99 GB with the dense weights. The earlier
    /// two-slab layout's estimate was 12.24 GB before context, against a
    /// measured peak of 9.47 GB on a 5-token prompt (K4).
    public static let stateBytes: UInt64 = (480 << 20) + (800 << 20)

    /// Everything but the context's global KV.
    public var bytesBeforeContext: UInt64 {
        expertResidentBytes + expertStagingBytes + denseBytes + fixedStateBytes
    }

    /// The whole footprint with `contextTokens` of global KV.
    public func bytes(contextTokens: Int) -> UInt64 {
        bytesBeforeContext + globalKVBytesPerToken * UInt64(max(0, contextTokens))
    }

    /// Per-layer expert file sizes (128 blobs at the installer's stride) for
    /// `config`, or nil when it is not an Inkling 2 baseline.
    public static func expertLayerBytes(config: ArchConfig) -> [UInt64]? {
        let formats = config.inkling2.routedExpertFormats
        guard config.family == .inkling2Loop, formats.count == config.numLayers else { return nil }
        var out: [UInt64] = []
        for name in formats {
            guard let format = GGMLBlockFormat(name: name),
                  let stride = Inkling2ExpertLayout.blobStride(format: format, config: config) else { return nil }
            out.append(stride * UInt64(config.numExperts))
        }
        return out
    }

    /// The footprint of an install of `installBytes` on a host with
    /// `physicalMemory`, or nil when `config` is not an Inkling 2 baseline.
    public static func estimate(installBytes: UInt64, physicalMemory: UInt64,
                                config: ArchConfig = .inkling2Mini) -> Inkling2MemoryFootprint? {
        guard let layerBytes = expertLayerBytes(config: config) else { return nil }
        let ink = config.inkling2
        let plan = Inkling2ExpertResidencyPlan.make(layerBytes: layerBytes.map { Int($0) },
                                                    loopedLayers: ink.loopCoreStart..<ink.loopCoreEnd,
                                                    physicalMemory: physicalMemory)
        let pool = layerBytes.reduce(0, +)
        let globalVisits = ink.layerVisits(numLayers: config.numLayers).filter { config.layerIsFull($0.layer) }.count
        let kvPerToken = UInt64(globalVisits * config.numFullKVHeads * config.fullHeadDim * 2 * 2)
        return Inkling2MemoryFootprint(expertResidentBytes: UInt64(plan.residentBytes),
                                       expertStagingBytes: UInt64(plan.stagingBytes),
                                       denseBytes: installBytes > pool ? installBytes - pool : 0,
                                       fixedStateBytes: stateBytes,
                                       globalKVBytesPerToken: kvPerToken)
    }
}

extension Inkling2MemoryFootprint {
    /// The context Inkling 2 runs with at most on a host where the residency
    /// plan stages layers (it cannot hold every layer's experts): 40,960
    /// tokens, whose global KV is 1.25 GiB beside the about 12.2 GB the plan
    /// keeps resident. Room for the acceptance run's longest case (about 26K
    /// prompt tokens and a 4,096-token completion) with margin; the program
    /// coordinator's choice (2026-11-23), reversible here alone.
    public static let stagedHostContextCap = 40_960

    /// `stagedHostContextCap` when `Inkling2ExpertResidencyPlan.make` stages
    /// any layer on a host with `physicalMemory` (under 48 GB today), else
    /// nil: the threshold is the plan's, not restated here.
    public static func contextCap(physicalMemory: UInt64, config: ArchConfig = .inkling2Mini) -> Int? {
        guard let layerBytes = expertLayerBytes(config: config) else { return nil }
        let ink = config.inkling2
        let plan = Inkling2ExpertResidencyPlan.make(layerBytes: layerBytes.map { Int($0) },
                                                    loopedLayers: ink.loopCoreStart..<ink.loopCoreEnd,
                                                    physicalMemory: physicalMemory)
        return plan.placements.contains(.staged) ? stagedHostContextCap : nil
    }
}

extension ModelFamily {
    /// The context a host with `physicalMemory` allows this family when that
    /// is below its native context, else nil. Only Inkling 2 has one
    /// (`Inkling2MemoryFootprint.contextCap`); every other family runs to its
    /// native context.
    public func memoryContextCap(physicalMemory: UInt64) -> Int? {
        switch self {
        case .inkling2Loop: Inkling2MemoryFootprint.contextCap(physicalMemory: physicalMemory)
        case .qwen36, .qwen38, .maple, .minicpm5: nil
        }
    }
}

import Darwin
import Foundation
import Metal

/// Buffer capacities, not physical residency. Shared aliases are counted once.
func uniqueBufferBytes(_ buffers: [MTLBuffer]) -> UInt64 {
    var seen = Set<ObjectIdentifier>()
    return buffers.reduce(0) { bytes, buffer in
        bytes + (seen.insert(ObjectIdentifier(buffer)).inserted ? UInt64(buffer.length) : 0)
    }
}

protocol RuntimeMemoryReporting {
    /// Target-model KV and recurrent-state buffers; excludes speculative drafts.
    var diagnosticKVStateBytes: UInt64? { get }
    /// Complete runner scratch only; nil when that ownership is not inventoried.
    var diagnosticScratchBytes: UInt64? { get }
}

extension RuntimeMemoryReporting {
    var diagnosticScratchBytes: UInt64? { nil }
}

/// Current post-generation snapshot. Metrics overlap and must not be added.
/// Null values explicitly mean unavailable; these are not peak measurements.
public struct RuntimeMemorySnapshot: Sendable, Equatable, Encodable {
    public let bytes: [String: UInt64?]
    public let scope = "current_process_and_model; overlapping_metrics; not_peak"

    public static func capture(model: Model, producer: any LogitProducer,
                               scratch: RawCompletionScratch) -> Self {
        let reporting = producer as? any RuntimeMemoryReporting
        var metrics = model.diagnosticMemoryBytes
        metrics["targetKVStateBuffers"] = .some(reporting?.diagnosticKVStateBytes)
        metrics["gemmaPrefixRecoveryBuffers"] = .some((producer as? any GemmaPrefixRecovering)?.gemmaRecoveryBytes)
        metrics["runnerScratchBuffers"] = .some(reporting?.diagnosticScratchBytes)
        metrics["completionScratchBuffers"] = scratch.diagnosticBufferBytes
        metrics["metalAllocated"] = UInt64(model.device.currentAllocatedSize)
        metrics["systemPhysicalMemory"] = ProcessInfo.processInfo.physicalMemory
        metrics["mappedWeightsPhysicalResidency"] = .some(nil)
        metrics["filesystemCache"] = .some(nil)
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size
                                           / MemoryLayout<integer_t>.size)
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        metrics["processPhysicalFootprint"] = .some(result == KERN_SUCCESS
                                                    ? UInt64(info.phys_footprint) : nil)
        metrics["processRSS"] = .some(result == KERN_SUCCESS
                                      ? UInt64(info.resident_size) : nil)
        return Self(bytes: metrics)
    }
}

public struct GemmaPrefixRecoveryDiagnostics: Sendable, Equatable, Encodable {
    public let outcome: String
    public let capture: String?
    public let allocatedBytes: UInt64

    public init(outcome: String, capture: String?, allocatedBytes: UInt64) {
        self.outcome = outcome
        self.capture = capture
        self.allocatedBytes = allocatedBytes
    }
}

/// Additive operator diagnostics; no prompt or generated text is included.
public struct RuntimeDiagnostics: Sendable, Equatable, Encodable {
    public let schemaVersion = 1
    public let cachedPromptTokens: Int
    public let computedPrefillTokens: Int
    public let prefill: PrefillExecutionReport?
    public let memory: RuntimeMemorySnapshot

    public let gemmaRecovery: GemmaPrefixRecoveryDiagnostics?

    public init(result: RawDecodeResult, memory: RuntimeMemorySnapshot,
                gemmaRecovery: GemmaPrefixRecoveryDiagnostics? = nil) {
        self.gemmaRecovery = gemmaRecovery
        cachedPromptTokens = result.cachedPromptTokens
        computedPrefillTokens = result.computedPrefillTokens
        prefill = result.prefillExecution
        self.memory = memory
    }

    /// Explicit null for an uninstrumented third-party producer.
    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(schemaVersion, forKey: .schemaVersion)
        try c.encode(cachedPromptTokens, forKey: .cachedPromptTokens)
        try c.encode(computedPrefillTokens, forKey: .computedPrefillTokens)
        try c.encode(prefill, forKey: .prefill)
        try c.encode(memory, forKey: .memory)
        try c.encodeIfPresent(gemmaRecovery, forKey: .gemmaRecovery)
    }

    private enum CodingKeys: String, CodingKey {
        case schemaVersion, cachedPromptTokens, computedPrefillTokens, prefill, memory, gemmaRecovery
    }

    public static var enabled: Bool {
        ProcessInfo.processInfo.environment["MFERENCE_DIAGNOSTICS"] == "1"
    }

    public func jsonLine() throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        return String(decoding: try encoder.encode(self), as: UTF8.self)
    }
}

extension Model {
    /// Reads only already-opened allocations; never faults in or opens weights.
    var diagnosticMemoryBytes: [String: UInt64?] {
        streamersQueue.sync {
            let coreBuffers = residentBuffer.chunks.map(\.buffer)
            let coreIdentities = Set(coreBuffers.map { ObjectIdentifier($0) })
            // residentAsF32 also caches original views for already-FP32 tensors.
            // They are not converted allocations and still belong to core.
            let convertedBuffers = convertedBox.views.values.map(\.buffer).filter {
                !coreIdentities.contains(ObjectIdentifier($0))
            }
            var slots: UInt64 = 0, mapped: UInt64 = 0, copied: UInt64 = 0
            var metadata: UInt64 = 0
            for backend in streamersBox.streamers.compactMap({ $0 }) {
                switch backend {
                case .pread(let streamer):
                    slots += streamer.diagnosticSlotBytes
                    metadata += streamer.diagnosticMetadataBytes
                case .resident(let streamer):
                    mapped += streamer.diagnosticMappedBytes
                    copied += streamer.diagnosticCopiedBytes
                    metadata += streamer.diagnosticMetadataBytes
                }
            }
            return [
                "mappedCoreWeightBuffers": uniqueBufferBytes(coreBuffers),
                "mappedExpertRegions": mapped,
                "copiedExpertBuffers": copied,
                "expertSlotBuffers": slots,
                "expertMetadataBuffers": metadata,
                "convertedWeightBuffers": uniqueBufferBytes(convertedBuffers),
            ]
        }
    }
}

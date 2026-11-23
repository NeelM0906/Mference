import Foundation
@testable import MferenceAppCore
import MferenceEngine
import Testing

/// Inkling 2 Mini in the app: an engine model in Also on this Mac, offered
/// from 32 GB up (the engine slot's floor), with an install hint while it is
/// missing. Model-free: fake catalogs only.
@Suite struct Inkling2AppModelTests {
    static let reason32 = "Needs a Mac with 32 GB of memory or more."
    static let installSize: UInt64 = 17_650_000_000

    @Test func itIsAnEngineModelFirstInAlsoOnThisMac() {
        let model = AppModel.inkling2Mini
        #expect(model.modelID == EngineProductSlot.inkling2Mini.modelID)
        #expect(AppModel(modelID: "inkling-2-mini-ud-q4_k_xl") == .inkling2Mini)
        #expect(model.isEngineModel && AppModel.engineModels.contains(model))
        #expect(model.group == .alsoOnThisMac)
        #expect(AppModel.allCases.filter { $0.group == .alsoOnThisMac } == [.inkling2Mini, .qwen36, .minicpm5, .maple])
        #expect(AppModel.allCases.filter { $0.group == .recommended } == [.quick, .deep])
        #expect(model.engineLabel == "Mference" && model.isPreview && model.supportsThinking)
        #expect(model.name == "Inkling 2 Mini" && model.shortName == "Inkling 2" && model.detail == "Mini")
        #expect(model.summary == "Deep-tier mixture-of-experts model.")
    }

    @Test func itIsOfferedFromThirtyTwoGigabytes() {
        for gigabytes: UInt64 in [8, 16, 24] {
            let memory = gigabytes << 30
            #expect(EngineService.memoryGateReason(.inkling2Mini, physicalMemory: memory) == Self.reason32)
            let fit = MemoryFit(physicalMemory: memory)
            #expect(!fit.isOffered(.inkling2Mini))
            #expect(fit.notOfferedReason(.inkling2Mini) == Self.reason32)
            #expect(fit.verdict(for: .inkling2Mini, sizeBytes: Self.installSize) == .tooLarge)
        }
        #expect(EngineService.memoryGateReason(.inkling2Mini, physicalMemory: (32 << 30) - 1) == Self.reason32)
        for gigabytes: UInt64 in [32, 48, 64] {
            let fit = MemoryFit(physicalMemory: gigabytes << 30)
            #expect(EngineService.memoryGateReason(.inkling2Mini, physicalMemory: gigabytes << 30) == nil)
            #expect(fit.isOffered(.inkling2Mini))
            #expect(fit.verdict(for: .inkling2Mini, sizeBytes: Self.installSize) == .fits)
        }
        // The runtime's figures (its expert residency plan, 32K context):
        // below 48 GB the looped core, two staging slabs (1.68 GB each with
        // option D's 48 decode slots per staged layer), the dense weights and
        // state, about 13.4 GB before context plus 1 GiB of global KV;
        // from 48 GB the whole install plus the same state.
        let at32 = MemoryFit(physicalMemory: 32 << 30).estimatedResidentBytes(.inkling2Mini, sizeBytes: Self.installSize)
        #expect(at32 == EngineService.runtimeResidentEstimate(.inkling2Mini, sizeBytes: Self.installSize,
                                                              physicalMemory: 32 << 30))
        #expect(at32 > 13_000_000_000 && at32 < 14_800_000_000)
        let at48 = MemoryFit(physicalMemory: 48 << 30).estimatedResidentBytes(.inkling2Mini, sizeBytes: Self.installSize)
        #expect(at48 > 19_800_000_000 && at48 < 20_300_000_000)
        // Every other model keeps the app's own profile.
        #expect(EngineService.runtimeResidentEstimate(.qwen36, sizeBytes: 1 << 30, physicalMemory: 32 << 30) == nil)
        #expect(MemoryFit(physicalMemory: 32 << 30).estimatedResidentBytes(.qwen36, sizeBytes: 10_000_000_000)
                == 3_000_000_000 + (2_048 << 20))
        // The other models' gates are unchanged.
        #expect(EngineService.memoryGateReason(.qwen36, physicalMemory: 8 << 30) == nil)
        #expect(EngineService.memoryGateReason(.maple, physicalMemory: 8 << 30) == nil)
        #expect(EngineService.memoryGateReason(.deep, physicalMemory: 24 << 30) == nil)
        #expect(EngineService.memoryGateReason(.deep, physicalMemory: 16 << 30) != nil)
    }

    @Test func pickerRowsFollowTheMemoryFloor() {
        let catalog = AppServices.fakeCatalog([.inkling2Mini])
        let big = EngineService.options(catalog: catalog, status: EngineHostStatus(), physicalMemory: 32 << 30)
        #expect(big.first { $0.model == .inkling2Mini }?.availability == .ready)
        let small = EngineService.options(catalog: catalog, status: EngineHostStatus(), physicalMemory: 16 << 30)
        guard case .notInstalled(let reason)? = small.first(where: { $0.model == .inkling2Mini })?.availability else {
            Issue.record("Inkling 2 Mini should not be offered at 16 GB"); return
        }
        #expect(reason == Self.reason32)
        let row = ModelPickerRow(small.first { $0.model == .inkling2Mini }!, selected: nil, physicalMemory: 16 << 30)
        #expect(row.statusText == "Needs 32 GB or more" && !row.isSelectable)
    }

    @Test func aMissingInstallSaysHowToMakeOne() {
        let rows = EngineService.options(catalog: EngineCatalog(models: []), status: EngineHostStatus(),
                                         physicalMemory: 32 << 30)
        guard case .notInstalled(let reason)? = rows.first(where: { $0.model == .inkling2Mini })?.availability,
              case .notInstalled(let other)? = rows.first(where: { $0.model == .qwen36 })?.availability else {
            Issue.record("expected both rows not installed"); return
        }
        #expect(reason.contains("MferenceRepack --model inkling2mini"))
        #expect(other == "Not installed on this Mac.")
        // A catalog reason (a gated runner, the engine's memory floor) wins.
        let skipped = EngineCatalog(models: [], skipped: [EngineCatalogSkip(
            location: URL(fileURLWithPath: "/lib/inkling2mini.gturbo"), reason: "not runnable (x): y",
            modelID: "inkling-2-mini-ud-q4_k_xl")])
        let gated = EngineService.options(catalog: skipped, status: EngineHostStatus(), physicalMemory: 32 << 30)
        #expect(gated.first { $0.model == .inkling2Mini }?.availability == .notInstalled(reason: "not runnable (x): y"))
    }
}

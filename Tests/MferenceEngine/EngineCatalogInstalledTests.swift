import Foundation
import Testing
import MferenceEngine

/// Opt-in check of discovery over this machine's real library roots. Gate:
/// `MFERENCE_ENGINE_CATALOG_CHECK=1`. Reads manifests, receipts, and GGUF
/// headers only; loads no model. Expects all five product models installed.
@Suite struct EngineCatalogInstalledTests {
    private static let enabled = ProcessInfo.processInfo.environment["MFERENCE_ENGINE_CATALOG_CHECK"] == "1"

    @Test(.enabled(if: enabled)) func realLibraryListsExactlyTheFiveProductModels() {
        let started = ContinuousClock.now
        let catalog = EngineCatalogDiscovery.discoverDefault()
        let elapsed = started.duration(to: .now)
        for model in catalog.models {
            FileHandle.standardError.write(Data(
                "[engine catalog] \(model.id) engine=\(model.engine.rawValue) context=\(model.contextLength) bytes=\(model.sizeBytes) at \(model.location.path)\n".utf8))
        }
        for skip in catalog.skipped {
            FileHandle.standardError.write(Data(
                "[engine catalog] skipped \(skip.location.path): \(skip.reason)\n".utf8))
        }
        FileHandle.standardError.write(Data("[engine catalog] discovery took \(elapsed)\n".utf8))
        #expect(catalog.models.map(\.id).sorted() == EngineProductSlot.allCases.map(\.modelID).sorted())
        #expect(catalog.descriptor(for: "qwen3.8-27b").map {
            if case .gguf(let metadata) = $0.format { metadata.hasMTP } else { false }
        } == true)
        // A retired Flash-Next install may still be on disk; it is never listed.
        #expect(!catalog.models.contains { $0.id.contains("flash-next") })
    }
}

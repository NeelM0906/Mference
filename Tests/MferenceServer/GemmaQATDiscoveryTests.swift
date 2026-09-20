import Foundation
import Testing
import Mference
@testable import MferenceServerCore

extension ServerLibraryProbeTests {
    @Test func qatDamagedLayoutIsNotReportedAsAnIntactUnsupportedInstall() throws {
        let root = try ServerLibraryFixture.makeRoot("qat-layout-integrity")
        defer { try? FileManager.default.removeItem(at: root) }
        let directory = try ServerLibraryFixture.makeQATInstall(in: root, named: "qat")
        try Data("[]".utf8).write(to: directory.appendingPathComponent("packed_experts/layout.json"))
        guard case .partial(let reason) = ServerLibraryProbe.probe(directory: directory) else {
            Issue.record("Same-size layout corruption must fail integrity before capability refusal")
            return
        }
        #expect(reason == "SHA-256 of packed_experts/layout.json does not match manifest.files[packed_experts/layout.json].sha256")
    }

    @Test func qatIntegrityIsCheckedBeforeReportingRunnable() throws {
        let root = try ServerLibraryFixture.makeRoot("qat-integrity")
        defer { try? FileManager.default.removeItem(at: root) }
        let directory = try ServerLibraryFixture.makeQATInstall(in: root, named: "alternate-name")
        #expect(ServerLibraryProbe.probe(directory: directory) == .complete(family: .gemma4))
        _ = try ServerLibraryFixture.makeCompleteInstall(in: root, named: "original")
        let index = ServerLibraryDiscovery.discover(roots: [root])
        #expect(Set(index.entries.map(\.modelID)) == ["gemma-4-26b-a4b-it", CheckpointIdentity.gemma4QAT])
        #expect(index.skipped.isEmpty)
        for name in ["tokenizer/generation_config.json", "model_weights.bin", "verified-install.json"] {
            let url = directory.appendingPathComponent(name)
            let data = try Data(contentsOf: url)
            try Data("[]".utf8).write(to: url)
            if case .partial = ServerLibraryProbe.probe(directory: directory) {} else {
                Issue.record("Damaged \(name) must be incomplete, not a valid-but-unavailable QAT install")
            }
            try data.write(to: url)
            try FileManager.default.removeItem(at: url)
            if case .partial = ServerLibraryProbe.probe(directory: directory) {} else {
                Issue.record("Missing \(name) must be incomplete")
            }
            try data.write(to: url)
        }
    }
}

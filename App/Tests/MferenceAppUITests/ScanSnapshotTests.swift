import MferenceAppCore
@testable import MferenceAppUI
import MferenceDesign
import SwiftData
import SwiftUI
import Testing

@Suite(.serialized) @MainActor struct ScanSnapshotTests {
    @Test func scan() async throws {
        let container = try AppModelContainer.inMemory()
        let model = ScriptedAssistantTextModel(kind: .appleOnDevice, answer: AppleUIFixtures.receiptAnswer)
        let receipt = try AppleUIFixtures.receipt()
        let session = ScanSession(scanner: AppleUIFixtures.FixedScanner(scanLines: receipt.lines),
                                  assistant: AppleModelService(primary: model, fallback: nil),
                                  context: container.mainContext,
                                  imageDirectory: URL.temporaryDirectory.appending(path: "mference-scan-snapshots"))
        session.open(receipt.url)
        await session.waitForReading()
        #expect(session.phase == .review)
        #expect(Set(session.highlights.keys) == [.merchant, .date, .tax, .total])
        for scheme in [ColorScheme.light, .dark] {
            try AppleSnapshot.render(AppleSnapshot.board(.scan) { ScanView(session: session) }, name: "07-scan", colorScheme: scheme)
        }

        let empty = ScanSession(scanner: AppleUIFixtures.FixedScanner(lines: []),
                                assistant: AppleModelService(primary: model, fallback: nil),
                                context: container.mainContext)
        try AppleSnapshot.render(AppleSnapshot.board(.scan) { ScanView(session: empty) }, name: "07-scan-empty")
    }

    @Test func shellSlotsBuildWithoutServices() {
        // The no-argument initializers the shell routes to render an honest
        // placeholder until the app injects the models.
        let renderer = ImageRenderer(content: ScanView().frame(width: 600, height: 600))
        #expect(renderer.nsImage != nil)
    }
}

import MferenceAppCore
@testable import MferenceAppUI
import MferenceDesign
import SwiftData
import SwiftUI
import Testing

@Suite(.serialized) @MainActor struct MailSnapshotTests {
    /// Kept for the whole run: a screen rendered in the shell starts its own
    /// load (`.task`), which can outlive the test; SwiftData traps if its
    /// container is gone by then.
    static let container = try! AppModelContainer.inMemory()

    @Test func mail() async throws {
        let container = Self.container
        let source = AppleUIFixtures.FixedMailSource(status: .available(version: "V10"), inbox: AppleUIFixtures.inbox,
                                                     bodies: AppleUIFixtures.bodies)
        let assistant = ScriptedAssistantTextModel(kind: .appleOnDevice, answers: [
            .success(AppleUIFixtures.eventAnswer), .success(AppleUIFixtures.sortAnswer),
        ])
        let draft = "Hi Priya, 11 works. I’ve moved it on my side. I’ll have the copy decisions written up by Tuesday so we can spend the time on the screens. See you Wednesday!"
        let model = MailInboxModel(source: source, context: container.mainContext,
                                   assistant: AppleModelService(primary: assistant, fallback: nil),
                                   drafter: MailReplyDrafter(engine: AppleUIFixtures.FixedEngine(answer: draft)),
                                   sender: RecordingMailSender(), broker: AppleUIFixtures.NoPromptBroker())
        await model.load()
        await model.writeDraft()
        #expect(model.count(in: .bucket("important")) == 4)
        #expect(model.eventCard != nil)
        #expect(model.draftState == .ready)
        for scheme in [ColorScheme.light, .dark] {
            try Snapshot.render(RootView(model: ShellModel(selection: .mail, defaults: nil), chrome: .inline)
                .environment(model), name: "03-mail", colorScheme: scheme)
        }

        let blocked = MailInboxModel(source: AppleUIFixtures.FixedMailSource(status: .needsFullDiskAccess, inbox: [],
                                                                             bodies: [:]),
                                     context: container.mainContext,
                                     assistant: AppleModelService(primary: assistant, fallback: nil), drafter: nil,
                                     sender: RecordingMailSender(), broker: AppleUIFixtures.NoPromptBroker())
        await blocked.load()
        try Snapshot.render(RootView(model: ShellModel(selection: .mail, defaults: nil), chrome: .inline)
                .environment(blocked), name: "03-mail-access")
    }

    @Test func shellSlotBuildsWithoutServices() {
        let renderer = ImageRenderer(content: MailView().frame(width: 600, height: 600))
        #expect(renderer.nsImage != nil)
    }
}

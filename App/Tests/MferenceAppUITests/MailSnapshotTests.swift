import MferenceAppCore
@testable import MferenceAppUI
import MferenceDesign
import SwiftUI
import Testing

@Suite(.serialized) @MainActor struct MailSnapshotTests {
    @Test func mail() async throws {
        let source = AppleUIFixtures.FixedMailSource(status: .available(version: "V10"), inbox: AppleUIFixtures.inbox,
                                                     bodies: AppleUIFixtures.bodies)
        let assistant = ScriptedAssistantTextModel(kind: .appleOnDevice, answers: [.success(AppleUIFixtures.eventAnswer)])
        let draft = "Hi Priya, 11 works. I’ve moved it on my side. I’ll have the copy decisions written up by Tuesday so we can spend the time on the screens. See you Wednesday!"
        let model = MailInboxModel(source: source, assistant: AppleModelService(primary: assistant, fallback: nil),
                                   drafter: MailReplyDrafter(engine: AppleUIFixtures.FixedEngine(answer: draft)),
                                   sender: RecordingMailSender(), broker: AppleUIFixtures.NoPromptBroker(),
                                   mailApp: FixedMailApp())
        await model.load()
        await model.writeDraft()
        #expect(model.count(in: .favorite(.inbox)) == 43_330)
        #expect(model.accountSections.map(\.account.title) == ["Google"] && model.localMailboxes.count == 1)
        #expect(model.eventCard != nil)
        #expect(model.draftState == .ready)
        for scheme in [ColorScheme.light, .dark] {
            try Snapshot.render(RootView(model: ShellModel(selection: .mail, defaults: nil), chrome: .inline)
                .environment(model), name: "03-mail", colorScheme: scheme)
        }

        let blocked = MailInboxModel(source: AppleUIFixtures.FixedMailSource(status: .needsFullDiskAccess, inbox: [],
                                                                             bodies: [:]),
                                     assistant: AppleModelService(primary: assistant, fallback: nil), drafter: nil,
                                     sender: RecordingMailSender(), broker: AppleUIFixtures.NoPromptBroker(),
                                     mailApp: FixedMailApp())
        await blocked.load()

        try Snapshot.render(RootView(model: ShellModel(selection: .mail, defaults: nil), chrome: .inline)
                .environment(blocked), name: "03-mail-access")

        // Mail is not running: the sidebar says so and offers to open it.
        let closed = MailInboxModel(source: source, assistant: AppleModelService(primary: assistant, fallback: nil),
                                    drafter: nil, sender: RecordingMailSender(),
                                    broker: AppleUIFixtures.NoPromptBroker(), mailApp: FixedMailApp(running: false))
        await closed.load()
        #expect(closed.syncState == .mailClosed)
        try Snapshot.render(RootView(model: ShellModel(selection: .mail, defaults: nil), chrome: .inline)
                .environment(closed), name: "03-mail-closed")
    }

    @Test func shellSlotBuildsWithoutServices() {
        let renderer = ImageRenderer(content: MailView().frame(width: 600, height: 600))
        #expect(renderer.nsImage != nil)
    }
}

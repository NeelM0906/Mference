import Foundation
import MferenceAppCore
import SwiftData
import Testing

@Suite struct SchemaTests {
    @Test @MainActor func conversationKeepsItsMessagesAndCascadesDelete() throws {
        let container = try AppModelContainer.inMemory()
        let context = container.mainContext
        let conversation = Conversation(title: "Standing desk warranties")
        context.insert(conversation)
        let first = StoredMessage(index: 0, role: .user, content: "Which desk?")
        let second = StoredMessage(index: 1, role: .assistant, content: "Stride E2.",
                                   modelRawValue: AppModel.quick.rawValue)
        conversation.messages.append(contentsOf: [first, second])
        try context.save()

        let fetched = try context.fetch(FetchDescriptor<Conversation>())
        #expect(fetched.count == 1)
        #expect(fetched.first?.messages.count == 2)

        context.delete(conversation)
        try context.save()
        #expect(try context.fetch(FetchDescriptor<StoredMessage>()).isEmpty)
    }

    @Test @MainActor func memoryAndMailRecordsPersist() throws {
        let container = try AppModelContainer.inMemory()
        let context = container.mainContext
        context.insert(MemoryItem(text: "Prefers aisle seats", sourceKind: "chat", sourceLabel: "Chat: Lisbon trip"))
        context.insert(MailBucket(key: "needs-reply", name: "Needs reply", sortOrder: 0))
        context.insert(MailBucketAssignment(messageID: "<a@b>", bucketKey: "needs-reply", isManual: true))
        context.insert(ScanRecord(kind: "receipt", title: "Cafe", recognizedText: "Total 4.50"))
        try context.save()
        #expect(try context.fetchCount(FetchDescriptor<MemoryItem>()) == 1)
        #expect(try context.fetchCount(FetchDescriptor<MailBucket>()) == 1)
        #expect(try context.fetchCount(FetchDescriptor<MailBucketAssignment>()) == 1)
        #expect(try context.fetchCount(FetchDescriptor<ScanRecord>()) == 1)
    }

    // MARK: V2 (Notes program spec 8)

    @Test @MainActor func notesFoldersAndRecordingsPersist() throws {
        let container = try AppModelContainer.inMemory()
        let context = container.mainContext
        let folder = NoteFolder(name: "Work", sortOrder: 0)
        context.insert(folder)
        let meeting = Note(title: "Design review", kind: .meeting, folder: folder)
        let recording = NoteRecording(sourceFileName: nil)
        recording.transcriptJSON = Data("[]".utf8)
        recording.summaryState = .pending
        recording.transcriptionState = .live
        recording.gapsJSON = Data("[]".utf8)
        meeting.recording = recording
        context.insert(meeting)
        context.insert(Note(title: "Groceries"))
        try context.save()

        let notes = try context.fetch(FetchDescriptor<Note>(sortBy: [SortDescriptor(\.title)]))
        #expect(notes.map(\.title) == ["Design review", "Groceries"])
        #expect(notes[0].kind == .meeting && notes[0].folder?.name == "Work")
        #expect(notes[0].recording?.summaryState == .pending)
        #expect(notes[0].recording?.transcriptionState == .live)
        #expect(notes[0].recording?.transcriptJSON == Data("[]".utf8))
        #expect(notes[1].kind == .note && notes[1].recording == nil && notes[1].deletedAt == nil)
        #expect(folder.notes.count == 1)
    }

    @Test @MainActor func deletingANoteDeletesItsRecordingButNotItsFolder() throws {
        let container = try AppModelContainer.inMemory()
        let context = container.mainContext
        let folder = NoteFolder(name: "Calls", sortOrder: 0)
        let note = Note(title: "Standup", kind: .meeting, folder: folder)
        note.recording = NoteRecording()
        context.insert(folder)
        context.insert(note)
        try context.save()

        context.delete(note)
        try context.save()
        #expect(try context.fetchCount(FetchDescriptor<NoteRecording>()) == 0)
        #expect(try context.fetchCount(FetchDescriptor<NoteFolder>()) == 1)
    }

    @Test @MainActor func deletingAFolderKeepsItsNotes() throws {
        let container = try AppModelContainer.inMemory()
        let context = container.mainContext
        let folder = NoteFolder(name: "Old", sortOrder: 0)
        context.insert(folder)
        context.insert(Note(title: "Keep me", folder: folder))
        try context.save()

        context.delete(folder)
        try context.save()
        let notes = try context.fetch(FetchDescriptor<Note>())
        #expect(notes.count == 1)
        #expect(notes.first?.folder == nil)
    }

    @Test func unknownKindsReadAsPlainNotes() {
        #expect(NoteKind(rawValue: "meeting") == .meeting)
        #expect(NoteKind.allCases.map(\.rawValue) == ["note", "meeting", "file"])
        #expect(RecordingSummaryState.allCases.map(\.rawValue) == ["none", "pending", "written", "failed"])
        #expect(TranscriptionState.allCases.map(\.rawValue) == ["none", "queued", "running", "live", "done", "failed"])
        #expect(NoteRecording().transcriptionState == .none)
    }

    /// A store written by the V1 app opens with V2: nothing lost, notes added.
    @Test @MainActor func aVersion1StoreOpensWithVersion2() throws {
        let directory = FileManager.default.temporaryDirectory.appending(path: "schema-v1-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appending(path: "Mference.store")

        do {
            let v1 = Schema(versionedSchema: MferenceSchemaV1.self)
            let container = try ModelContainer(for: v1, configurations: ModelConfiguration(schema: v1, url: url))
            let context = container.mainContext
            // V1's own classes: the `Conversation` alias now names V3's.
            let conversation = MferenceSchemaV1.Conversation(title: "Before notes")
            context.insert(conversation)
            conversation.messages.append(MferenceSchemaV1.StoredMessage(index: 0, role: .user, content: "Hello"))
            context.insert(MemoryItem(text: "Likes tea"))
            try context.save()
        }

        let container = try AppModelContainer.make(at: url)
        let context = container.mainContext
        let conversations = try context.fetch(FetchDescriptor<Conversation>())
        #expect(conversations.map(\.title) == ["Before notes"])
        #expect(conversations.first?.messages.first?.content == "Hello")
        #expect(try context.fetchCount(FetchDescriptor<MemoryItem>()) == 1)
        context.insert(Note(title: "First note"))
        try context.save()
        #expect(try context.fetchCount(FetchDescriptor<Note>()) == 1)
    }

    /// A store written by the V2 app opens with V3: chats kept, and messages
    /// can now carry attachments.
    @Test @MainActor func aVersion2StoreOpensWithVersion3() throws {
        let directory = FileManager.default.temporaryDirectory.appending(path: "schema-v2-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appending(path: "Mference.store")

        do {
            let v2 = Schema(versionedSchema: MferenceSchemaV2.self)
            let container = try ModelContainer(for: v2, configurations: ModelConfiguration(schema: v2, url: url))
            let context = container.mainContext
            let conversation = MferenceSchemaV1.Conversation(title: "Before attachments")
            context.insert(conversation)
            conversation.messages.append(MferenceSchemaV1.StoredMessage(index: 0, role: .user, content: "Hello"))
            context.insert(Note(title: "A note"))
            try context.save()
        }

        let container = try AppModelContainer.make(at: url)
        let context = container.mainContext
        let conversations = try context.fetch(FetchDescriptor<Conversation>())
        #expect(conversations.map(\.title) == ["Before attachments"])
        let message = try #require(conversations.first?.messages.first)
        #expect(message.content == "Hello")
        #expect(message.attachments.isEmpty)
        #expect(try context.fetchCount(FetchDescriptor<Note>()) == 1)

        message.attachments = [ChatAttachment(filename: "a.txt", kind: .document, byteCount: 1, text: "a")]
        try context.save()
        let reread = try ModelContext(container).fetch(FetchDescriptor<StoredMessage>())
        #expect(reread.first?.attachments.map(\.filename) == ["a.txt"])
    }
}

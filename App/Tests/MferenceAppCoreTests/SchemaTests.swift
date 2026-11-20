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
}

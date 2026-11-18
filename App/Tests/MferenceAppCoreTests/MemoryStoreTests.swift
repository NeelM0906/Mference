import Foundation
import MferenceAppCore
import SwiftData
import Testing

@Suite struct MemoryStoreTests {
    @MainActor static func store() throws -> MemoryStore {
        MemoryStore(container: try AppModelContainer.inMemory())
    }

    @Test @MainActor func addEditDeleteRoundTrip() throws {
        let store = try Self.store()
        let item = try #require(try store.add("  Prefers aisle seats \n"))
        #expect(item.text == "Prefers aisle seats")
        #expect(item.sourceKind == "manual")
        try store.edit(item, text: "Prefers window seats")
        #expect(try store.items().map(\.text) == ["Prefers window seats"])
        try store.setEnabled(item, false)
        #expect(try store.enabledFacts().isEmpty)
        try store.delete(item)
        #expect(try store.count() == 0)
    }

    @Test @MainActor func emptyEditDeletesAndDuplicatesMerge() throws {
        let store = try Self.store()
        let first = try #require(try store.add("Dentist is Northside Dental"))
        let again = try store.add("dentist is northside dental")
        #expect(again?.id == first.id)
        #expect(try store.count() == 1)
        #expect(try store.add("   ") == nil)
        try store.edit(first, text: " ")
        #expect(try store.count() == 0)
    }

    @Test @MainActor func privateChatsNeverFeedMemory() throws {
        let container = try AppModelContainer.inMemory()
        let store = MemoryStore(container: container)
        let secret = Conversation(title: "Party", isPrivate: true)
        let lisbon = Conversation(title: "Lisbon trip")
        container.mainContext.insert(secret)
        container.mainContext.insert(lisbon)
        try container.mainContext.save()

        #expect(try store.learn("Has a surprise party planned", from: .chat(id: secret.id)) == nil)
        #expect(try store.learn("Unknown chat fact", from: .chat(id: UUID())) == nil)
        #expect(try store.count() == 0)

        let learned = try #require(try store.learn("Partner: Lena", from: .chat(id: lisbon.id)))
        #expect(learned.sourceKind == "chat")
        #expect(learned.sourceID == lisbon.id)
        #expect(learned.sourceLabel == "Lisbon trip")

        // The chat is made private: what it taught goes, and it teaches no more.
        lisbon.isPrivate = true
        try container.mainContext.save()
        #expect(try store.forget(learnedFrom: lisbon.id) == 1)
        #expect(try store.learn("Partner: Lena", from: .chat(id: lisbon.id)) == nil)
        #expect(try store.count() == 0)
    }

    @Test @MainActor func switchedOffFactsStayOffUnlessAddedByHand() throws {
        let container = try AppModelContainer.inMemory()
        let store = MemoryStore(container: container)
        let chat = Conversation(title: "Dentist")
        container.mainContext.insert(chat)
        try container.mainContext.save()
        let item = try #require(try store.learn("Dentist is Northside Dental", from: .chat(id: chat.id)))
        try store.setEnabled(item, false)
        // Learning it again does not switch it back on or change its source.
        #expect(try store.learn("dentist is northside dental", from: .mail(id: nil, label: "1 email")) == nil)
        #expect(item.isEnabled == false)
        #expect(item.sourceKind == "chat")
        // The person adding it by hand does.
        _ = try store.add("Dentist is Northside Dental")
        #expect(item.isEnabled)
    }

    @Test @MainActor func whereLearnedNamesTheSourceAndDate() throws {
        let store = try Self.store()
        var components = DateComponents()
        components.year = 2026; components.month = 9; components.day = 14; components.hour = 12
        let utc = TimeZone(identifier: "UTC")!
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = utc
        let date = calendar.date(from: components)!
        let mail = try #require(try store.learn("Dentist is Northside Dental",
                                                from: .mail(id: nil, label: "1 email"), now: date))
        let manual = try #require(try store.add("Vegetarian", now: date))
        let locale = Locale(identifier: "en_US")
        #expect(MemoryStore.whereLearned(mail, locale: locale, timeZone: utc) == "From Mail, Sep 14")
        #expect(MemoryStore.whereLearned(manual, locale: locale, timeZone: utc) == "Added by you, Sep 14")
    }

    @Test @MainActor func newestChangeFirst() throws {
        let store = try Self.store()
        let old = try #require(try store.add("A", now: Date(timeIntervalSince1970: 1)))
        _ = try store.add("B", now: Date(timeIntervalSince1970: 2))
        try store.edit(old, text: "A2", now: Date(timeIntervalSince1970: 3))
        #expect(try store.items().map(\.text) == ["A2", "B"])
    }
}

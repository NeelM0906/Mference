import Foundation
import SwiftData

/// The built-in buckets (mockup 03: "Sorted for you"). Names are the
/// person's to change (L8); keys and descriptions stay fixed.
public enum MailBucketDefaults {
    public struct Bucket: Sendable, Hashable {
        public var key: String
        public var name: String
        public var description: String
    }

    public static let all: [Bucket] = [
        Bucket(key: "important", name: "Important",
               description: "From a real person and needs a reply or a decision from the reader"),
        Bucket(key: "calendar", name: "Calendar",
               description: "Invitations, meeting changes, bookings and appointments"),
        Bucket(key: "reading", name: "Reading",
               description: "Newsletters, articles and digests to read later"),
        Bucket(key: "other", name: "Other",
               description: "Receipts, notifications, promotions and everything else"),
    ]

    /// Messages the person marked done (E). Not a sorter choice.
    public static let doneKey = "done"
}

/// Buckets and which message is in which, stored in SwiftData. Mail's own
/// store is never written.
@MainActor
public final class MailBucketStore {
    private let context: ModelContext

    public init(context: ModelContext) {
        self.context = context
    }

    /// The buckets in order, creating the built-in ones on first use.
    public func buckets() throws -> [MailBucket] {
        var existing = try context.fetch(FetchDescriptor<MailBucket>(sortBy: [SortDescriptor(\.sortOrder)]))
        let keys = Set(existing.map(\.key))
        var added = false
        for (index, bucket) in MailBucketDefaults.all.enumerated() where !keys.contains(bucket.key) {
            let model = MailBucket(key: bucket.key, name: bucket.name, sortOrder: index)
            context.insert(model)
            existing.append(model)
            added = true
        }
        if added { try context.save() }
        return existing.sorted { $0.sortOrder < $1.sortOrder }
    }

    public func rename(key: String, to name: String) throws {
        let trimmed = name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        let descriptor = FetchDescriptor<MailBucket>(predicate: #Predicate { $0.key == key })
        guard let bucket = try context.fetch(descriptor).first else { return }
        bucket.name = String(trimmed.prefix(40))
        try context.save()
    }

    /// Message-ID to bucket key.
    public func assignments(for messageIDs: [String]) throws -> [String: MailBucketAssignment] {
        let ids = messageIDs
        let descriptor = FetchDescriptor<MailBucketAssignment>(predicate: #Predicate { ids.contains($0.messageID) })
        return Dictionary(try context.fetch(descriptor).map { ($0.messageID, $0) }, uniquingKeysWith: { first, _ in first })
    }

    /// The person moved a message: it sticks.
    public func assign(_ messageID: String, to key: String, manual: Bool, at date: Date = .now) throws {
        let id = messageID
        let descriptor = FetchDescriptor<MailBucketAssignment>(predicate: #Predicate { $0.messageID == id })
        if let existing = try context.fetch(descriptor).first {
            guard manual || !existing.isManual else { return }
            existing.bucketKey = key
            existing.isManual = manual
            existing.assignedAt = date
        } else {
            context.insert(MailBucketAssignment(messageID: messageID, bucketKey: key, isManual: manual, assignedAt: date))
        }
        try context.save()
    }

    public func removeAssignment(_ messageID: String) throws {
        let id = messageID
        try context.delete(model: MailBucketAssignment.self, where: #Predicate { $0.messageID == id })
        try context.save()
    }
}

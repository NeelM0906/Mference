import Foundation
import MferenceAppCore
import Observation
import SwiftData

/// `MemoryListModel` over U2's `MemoryStore`. The list reloads after every
/// change made here and whenever the store's context saves elsewhere (a chat
/// that learned something).
@Observable
public final class LiveMemoryListModel: MemoryListModel {
    public private(set) var memories: [MemoryEntry] = []
    /// The last error, for a quiet line under the list; nil when fine.
    public private(set) var lastError: String?

    @ObservationIgnored private let store: MemoryStore
    @ObservationIgnored private var saveObserver: (any NSObjectProtocol)?

    public init(store: MemoryStore) {
        self.store = store
        reload()
        saveObserver = NotificationCenter.default.addObserver(
            forName: ModelContext.didSave, object: nil, queue: .main) { [weak self] _ in
                MainActor.assumeIsolated { self?.reload() }
            }
    }

    isolated deinit {
        if let saveObserver { NotificationCenter.default.removeObserver(saveObserver) }
    }

    public func add(_ text: String) {
        perform { _ = try store.add(text) }
    }

    public func update(_ id: UUID, text: String) {
        perform {
            guard let item = try item(id) else { return }
            try store.edit(item, text: text)
        }
    }

    public func delete(_ id: UUID) {
        perform {
            guard let item = try item(id) else { return }
            try store.delete(item)
        }
    }

    // MARK: Helpers

    private func item(_ id: UUID) throws -> MemoryItem? {
        try store.items().first { $0.id == id }
    }

    private func perform(_ change: () throws -> Void) {
        do {
            try change()
            lastError = nil
        } catch {
            lastError = "Memory could not be saved: \(error.localizedDescription)"
        }
        reload()
    }

    func reload() {
        do {
            memories = try store.items().map {
                MemoryEntry(id: $0.id, text: $0.text, source: MemoryStore.whereLearned($0))
            }
        } catch {
            lastError = "Memory could not be read: \(error.localizedDescription)"
        }
    }
}

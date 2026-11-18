import Foundation
import Observation
import SwiftData

/// The Scan screen's state: read an image, fill the fields, let the person
/// check them, save the confirmed scan, export CSV.
@MainActor @Observable
public final class ScanSession {
    public enum Phase: Sendable, Hashable {
        /// Nothing opened yet.
        case empty
        /// Vision is reading, then the model is filling the fields.
        case reading
        /// Fields are shown for the person to check.
        case review
        /// The person confirmed; the scan is saved.
        case saved
        /// `message` is user-facing.
        case failed(message: String)
    }

    public private(set) var phase: Phase = .empty
    /// The image being reviewed.
    public private(set) var imageURL: URL?
    public private(set) var reading: ScanReading?
    /// What the person sees and edits.
    public var fields = ScanFields(fields: [])
    /// The id of the saved `ScanRecord` once confirmed.
    public private(set) var savedRecordID: UUID?

    private let scanner: any DocumentRecognizing
    private let assistant: any AppleModelServicing
    private let context: ModelContext
    private let imageDirectory: URL
    private var readTask: Task<Void, Never>?
    /// Every read started and not yet awaited, so tests can wait for a
    /// replaced read to finish too.
    @ObservationIgnored private var pendingReads: [Task<Void, Never>] = []

    /// `imageDirectory` keeps copies of scanned images (default: the app's
    /// Application Support folder, `Scans/`).
    public init(scanner: any DocumentRecognizing, assistant: any AppleModelServicing, context: ModelContext,
                imageDirectory: URL = AppStorageLocation.applicationSupport.appending(path: "Scans",
                                                                                      directoryHint: .isDirectory)) {
        self.scanner = scanner
        self.assistant = assistant
        self.context = context
        self.imageDirectory = imageDirectory
    }

    /// "Receipt from Corner Hardware", or "Receipt" before the merchant is known.
    public var title: String {
        fields.value(.merchant).map { "Receipt from \($0)" } ?? "Receipt"
    }

    /// Who filled the fields, for the line under the title.
    public var filledByLabel: String? { fields.filledBy?.label }

    /// Image types Scan accepts from Open file and drag-drop.
    public nonisolated static let acceptedExtensions: Set<String> = ["png", "jpg", "jpeg", "heic", "heif", "tif", "tiff", "pdf",
                                                         "gif", "bmp", "webp"]

    public nonisolated static func accepts(_ url: URL) -> Bool {
        acceptedExtensions.contains(url.pathExtension.lowercased())
    }

    /// Starts reading an image file (Open file or drag-drop); replaces any scan
    /// being reviewed.
    public func open(_ url: URL) {
        readTask?.cancel()
        imageURL = url
        reading = nil
        savedRecordID = nil
        fields = ScanFields(fields: [])
        phase = .reading
        let task = Task { await read(url) }
        readTask = task
        pendingReads.append(task)
    }

    /// Waits for every read started so far, replaced ones included (tests).
    public func waitForReading() async {
        while !pendingReads.isEmpty {
            await pendingReads.removeFirst().value
        }
    }

    private func read(_ url: URL) async {
        do {
            let reading = try await scanner.recognize(imageAt: url)
            try Task.checkCancellation()
            let printed = ReceiptHeuristics.read(reading)
            var model: ReceiptDraft?
            var filledBy: AssistantModelKind?
            if let result = try? await assistant.fillReceipt(from: reading.transcript) {
                model = result.value
                filledBy = result.model
            }
            try Task.checkCancellation()
            self.reading = reading
            fields = ReceiptFieldMerger.merge(model: model, text: printed, reading: reading, filledBy: filledBy)
            phase = .review
        } catch {
            // A replaced read may still fail after the new one finished:
            // only the current read may change the screen.
            guard !Task.isCancelled, !(error is CancellationError) else { return }
            let scanError = error as? ScanError ?? .recognitionFailed(error.localizedDescription)
            phase = .failed(message: scanError.userMessage)
        }
    }

    /// The person edited a field: it is now theirs, so it no longer needs checking.
    public func setValue(_ value: String, for key: ScanFieldKey) {
        fields[key] = ScanField(key: key, value: value, confidence: .sure)
    }

    /// Saves the confirmed scan with a copy of its image. Throws, saving
    /// nothing, when the image can no longer be copied (moved or deleted).
    @discardableResult
    public func confirm() throws -> ScanRecord {
        guard phase == .review, let reading else { throw ScanSessionError.nothingToConfirm }
        let record = ScanRecord(kind: "receipt", title: title, recognizedText: reading.transcript)
        record.fieldsJSON = try fields.encoded()
        record.isConfirmed = true
        if let imageURL {
            record.imageRelativePath = try copyImage(imageURL, id: record.id)
        }
        context.insert(record)
        try context.save()
        savedRecordID = record.id
        phase = .saved
        return record
    }

    /// Clears the screen for the next scan.
    public func reset() {
        readTask?.cancel()
        readTask = nil
        phase = .empty
        imageURL = nil
        reading = nil
        savedRecordID = nil
        fields = ScanFields(fields: [])
    }

    /// CSV of every confirmed receipt, oldest first.
    public func exportCSV() throws -> String {
        let descriptor = FetchDescriptor<ScanRecord>(
            predicate: #Predicate { $0.isConfirmed && $0.kind == "receipt" },
            sortBy: [SortDescriptor(\.createdAt)])
        let rows = try context.fetch(descriptor).compactMap { record -> ScanExportRow? in
            guard let data = record.fieldsJSON, let fields = try? ScanFields.decode(data) else { return nil }
            return ScanExportRow(scannedAt: record.createdAt, fields: fields)
        }
        return ScanCSVExporter.csv(rows)
    }

    /// Path relative to Application Support/Mference.
    private func copyImage(_ url: URL, id: UUID) throws -> String {
        try FileManager.default.createDirectory(at: imageDirectory, withIntermediateDirectories: true)
        let name = "\(id.uuidString).\(url.pathExtension.lowercased())"
        let destination = imageDirectory.appending(path: name)
        try FileManager.default.copyItem(at: url, to: destination)
        return "Scans/\(name)"
    }
}

public enum ScanSessionError: Error, Sendable, Hashable {
    case nothingToConfirm
}

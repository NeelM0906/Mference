import Foundation

/// Files a note shows (images, audio clips), kept in
/// `Application Support/Mference/Notes/<note-id>/` (Notes spec 7). An
/// attachment's id is its file name in that folder. Meeting and file audio
/// lives in `Recordings/` and is the speech service's; this never touches it.
public struct NoteAttachments: Sendable {
    public let root: URL

    public init(root: URL) {
        self.root = root
    }

    /// The app's folder.
    public static var live: NoteAttachments {
        NoteAttachments(root: AppStorageLocation.applicationSupport.appending(path: "Notes", directoryHint: .isDirectory))
    }

    /// A fresh folder under the temporary directory (tests and previews).
    public static func temporary() -> NoteAttachments {
        NoteAttachments(root: FileManager.default.temporaryDirectory
            .appending(path: "mference-notes-\(UUID().uuidString)", directoryHint: .isDirectory))
    }

    public func directory(for noteID: UUID) -> URL {
        root.appending(path: noteID.uuidString, directoryHint: .isDirectory)
    }

    /// The file for an attachment id. An id is one file name: anything
    /// else (a path, "..", a hidden name) maps to a name that never exists,
    /// so a crafted id can never reach outside the note's folder.
    public func url(of attachmentID: String, in noteID: UUID) -> URL {
        directory(for: noteID).appending(path: Self.isValidID(attachmentID) ? attachmentID : "invalid-attachment-id",
                                         directoryHint: .notDirectory)
    }

    /// One plain file name: no separators, not hidden, not "." or "..".
    public static func isValidID(_ id: String) -> Bool {
        !id.isEmpty && !id.hasPrefix(".") && !id.contains("/") && !id.contains("\\") && !id.contains(":")
            && id.count <= 255
    }

    /// Letters and digits only, lowercased ("png", "m4a").
    static func cleanExtension(_ ext: String) -> String {
        String(ext.lowercased().filter { $0.isASCII && ($0.isLetter || $0.isNumber) }.prefix(10))
    }

    /// Copies `file` into the note's folder; returns the new attachment id.
    public func importFile(_ file: URL, into noteID: UUID) throws -> String {
        let directory = directory(for: noteID)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let ext = Self.cleanExtension(file.pathExtension)
        let id = UUID().uuidString + (ext.isEmpty ? "" : ".\(ext)")
        try FileManager.default.copyItem(at: file, to: directory.appending(path: id))
        return id
    }

    /// Writes `data` as a new attachment (pasted images).
    public func write(_ data: Data, fileExtension: String, into noteID: UUID) throws -> String {
        let directory = directory(for: noteID)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let ext = Self.cleanExtension(fileExtension)
        let id = UUID().uuidString + (ext.isEmpty ? "" : ".\(ext)")
        try data.write(to: directory.appending(path: id), options: .atomic)
        return id
    }

    /// Copies every attachment of one note to another (Duplicate).
    public func copyAll(from source: UUID, to destination: UUID) throws {
        let from = directory(for: source)
        guard FileManager.default.fileExists(atPath: from.path(percentEncoded: false)) else { return }
        let to = directory(for: destination)
        if FileManager.default.fileExists(atPath: to.path(percentEncoded: false)) {
            try FileManager.default.removeItem(at: to)
        }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        try FileManager.default.copyItem(at: from, to: to)
    }

    /// Deletes the note's folder (purge). Missing is fine.
    public func deleteAll(for noteID: UUID) throws {
        let directory = directory(for: noteID)
        guard FileManager.default.fileExists(atPath: directory.path(percentEncoded: false)) else { return }
        try FileManager.default.removeItem(at: directory)
    }

    /// Attachment ids present on disk for a note.
    public func list(for noteID: UUID) -> [String] {
        (try? FileManager.default.contentsOfDirectory(atPath: directory(for: noteID).path(percentEncoded: false)))?
            .filter { !$0.hasPrefix(".") }.sorted() ?? []
    }
}

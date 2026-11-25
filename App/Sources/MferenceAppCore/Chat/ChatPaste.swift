import AppKit
import Foundation

/// What ⌘V in the chat composer attaches: files copied in Finder, or a
/// picture copied as data (a screenshot, an image copied in Preview). Text,
/// including text copied with a picture beside it, stays with the text field.
public enum ChatPasteContent: Equatable, Sendable {
    case files([URL])
    case image(Data, fileExtension: String)

    /// The content to attach, or nil when the paste is text for the field.
    public static func read(_ pasteboard: NSPasteboard) -> ChatPasteContent? {
        let urls = pasteboard.readObjects(forClasses: [NSURL.self],
                                          options: [.urlReadingFileURLsOnly: true]) as? [URL] ?? []
        if !urls.isEmpty { return .files(urls) }
        guard pasteboard.string(forType: .string) == nil else { return nil }
        if let png = pasteboard.data(forType: .png) { return .image(png, fileExtension: "png") }
        if let tiff = pasteboard.data(forType: .tiff) { return .image(tiff, fileExtension: "tiff") }
        return nil
    }
}

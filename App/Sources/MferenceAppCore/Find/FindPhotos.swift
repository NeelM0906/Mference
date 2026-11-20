import AppKit
import AVFoundation
import Foundation
import ImageIO
import Photos
import Synchronization
import UniformTypeIdentifiers

/// The Photos library as a source for Find (Settings › Find › Photos
/// library). Photos keeps its pictures inside its own library package, which
/// Find never crawls, so the library is read through PhotoKit with the
/// person's permission (`AppPermission.photos`).
///
/// Each photo or video is indexed as a file at a path under `rootPath`
/// ("/Photos Library/<identifier>/<name>"), so the index, search and chat
/// treat it like any other file; `FindLibraryReader` and the UI know to
/// fetch its pixels from PhotoKit instead of the disk.
public enum FindPhotos {
    /// The index root for the library (not a real folder).
    public static let rootPath = "/Photos Library"
    public static var root: URL { URL(filePath: rootPath, directoryHint: .isDirectory) }

    /// The index path of an asset.
    public static func url(identifier: String, name: String) -> URL {
        let encoded = identifier.addingPercentEncoding(withAllowedCharacters: .alphanumerics) ?? identifier
        return URL(filePath: "\(rootPath)/\(encoded)/\(name)")
    }

    /// Whether `url` is a Photos library item.
    public static func isLibraryItem(_ url: URL) -> Bool {
        url.path.hasPrefix(rootPath + "/")
    }

    /// The PhotoKit identifier of a library item.
    public static func identifier(of url: URL) -> String? {
        guard isLibraryItem(url) else { return nil }
        let parts = url.path.dropFirst(rootPath.count + 1).split(separator: "/", maxSplits: 1)
        return parts.first.flatMap { String($0).removingPercentEncoding }
    }

    /// A name that reads well in results and is searchable by its words:
    /// "Photo 2024-05-03 at 18.22.heic", "Screenshot …", "Video …".
    static func name(kind: FindKind, isScreenshot: Bool, date: Date?) -> String {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "yyyy-MM-dd 'at' HH.mm.ss"
        let when = date.map { " " + formatter.string(from: $0) } ?? ""
        switch kind {
        case .video: return "Video\(when).mov"
        default: return isScreenshot ? "Screenshot\(when).png" : "Photo\(when).heic"
        }
    }

    // MARK: For the UI

    /// A picture of a library item (a photo, or a video's poster frame).
    public static func image(for url: URL, maxPixel: Int) async -> CGImage? {
        guard let identifier = identifier(of: url) else { return nil }
        return await LivePhotoLibrary.image(identifier: identifier, maxPixel: maxPixel)
    }

    /// A playable video for the preview.
    public static func playerItem(for url: URL) async -> AVPlayerItem? {
        guard let identifier = identifier(of: url),
              let asset = PHAsset.fetchAssets(withLocalIdentifiers: [identifier], options: nil).firstObject
        else { return nil }
        let options = PHVideoRequestOptions()
        options.isNetworkAccessAllowed = true
        options.deliveryMode = .automatic
        return await withCheckedContinuation { (continuation: CheckedContinuation<LivePhotoLibrary.Unchecked<AVPlayerItem?>, Never>) in
            PHImageManager.default().requestPlayerItem(forVideo: asset, options: options) { item, _ in
                continuation.resume(returning: LivePhotoLibrary.Unchecked(item))
            }
        }.value
    }

    /// Opens a library item: a photo as a copy in the default viewer, a
    /// video in the default player (Photos has no public way to show one
    /// item).
    @MainActor
    public static func open(_ url: URL) async {
        guard let identifier = identifier(of: url) else { return }
        if let file = await LivePhotoLibrary.viewableFile(identifier: identifier) {
            NSWorkspace.shared.open(file)
        } else {
            showInPhotos()
        }
    }

    /// Brings Photos forward.
    @MainActor
    public static func showInPhotos() {
        if let photos = NSWorkspace.shared.urlForApplication(withBundleIdentifier: "com.apple.Photos") {
            NSWorkspace.shared.openApplication(at: photos, configuration: NSWorkspace.OpenConfiguration())
        }
    }
}

/// The Photos library behind Find; tests use a fake over plain files.
public protocol FindPhotoLibrary: Sendable {
    /// Every photo and video, as crawl entries under `FindPhotos.root`.
    func entries() async -> [FindCrawlEntry]
    /// A local file Find's reader can read for this item: a picture written
    /// to a temporary file (the caller deletes it, `isTemporary`), or a
    /// video's own file. Nil when the item is not on this Mac.
    func readableFile(for url: URL, kind: FindKind, maxPixel: Int) async -> (url: URL, isTemporary: Bool)?
    /// Calls `handler` when the library changes (after `entries` was read).
    func observeChanges(_ handler: @escaping @Sendable () -> Void)
    /// The picture of a photo, at most `maxPixel` on its long side, already
    /// decoded (for the encoders: no file round trip).
    func picture(for url: URL, maxPixel: Int) async -> CGImage?
}

/// The real library, through PhotoKit. Reads only what is on this Mac: an
/// item kept only in iCloud ("Optimize Mac Storage") is read from its local
/// preview, or left for later.
public final class LivePhotoLibrary: NSObject, FindPhotoLibrary, PHPhotoLibraryChangeObserver, @unchecked Sendable {
    private let handler = Mutex<(@Sendable () -> Void)?>(nil)

    override public init() {
        super.init()
    }

    deinit {
        PHPhotoLibrary.shared().unregisterChangeObserver(self)
    }

    public func entries() async -> [FindCrawlEntry] {
        guard PHPhotoLibrary.authorizationStatus(for: .readWrite) == .authorized
            || PHPhotoLibrary.authorizationStatus(for: .readWrite) == .limited else { return [] }
        return await Task.detached(priority: .utility) {
            let options = PHFetchOptions()
            options.predicate = NSPredicate(format: "mediaType == %d OR mediaType == %d",
                                            PHAssetMediaType.image.rawValue, PHAssetMediaType.video.rawValue)
            options.includeHiddenAssets = false
            let assets = PHAsset.fetchAssets(with: options)
            var entries: [FindCrawlEntry] = []
            entries.reserveCapacity(assets.count)
            assets.enumerateObjects { asset, _, _ in
                let kind: FindKind = asset.mediaType == .video ? .video : .image
                let name = FindPhotos.name(kind: kind, isScreenshot: asset.mediaSubtypes.contains(.photoScreenshot),
                                           date: asset.creationDate)
                entries.append(FindCrawlEntry(
                    url: FindPhotos.url(identifier: asset.localIdentifier, name: name), kind: kind,
                    size: Int64(asset.pixelWidth) * Int64(asset.pixelHeight),
                    modified: asset.modificationDate ?? asset.creationDate ?? .distantPast, inode: 0, device: 0))
            }
            return entries
        }.value
    }

    public func readableFile(for url: URL, kind: FindKind, maxPixel: Int) async -> (url: URL, isTemporary: Bool)? {
        guard let identifier = FindPhotos.identifier(of: url) else { return nil }
        if kind == .video {
            return await Self.videoFile(identifier: identifier).map { ($0, false) }
        }
        try? FileManager.default.createDirectory(at: FindMediaFolder.url, withIntermediateDirectories: true)
        // Keep the name, so a screenshot is still read as one.
        let stem = "\(UUID().uuidString) \(url.deletingPathExtension().lastPathComponent)"
        // The photo's own encoded bytes, written as they are: the reader
        // decodes them once, at the size it needs (no decode and re-encode
        // here). A photo kept only in iCloud falls back to its local preview.
        if let original = await Self.originalData(identifier: identifier) {
            let file = FindMediaFolder.url.appending(path: "\(stem).\(original.ext)")
            if (try? original.data.write(to: file)) != nil { return (file, true) }
        }
        guard let image = await Self.image(identifier: identifier, maxPixel: maxPixel) else { return nil }
        // A JPEG: written by the hardware encoder, far faster than PNG, and
        // clean enough for reading text.
        let file = FindMediaFolder.url.appending(path: "\(stem).jpg")
        guard let destination = CGImageDestinationCreateWithURL(file as CFURL, UTType.jpeg.identifier as CFString, 1, nil)
        else { return nil }
        CGImageDestinationAddImage(destination, image, [kCGImageDestinationLossyCompressionQuality: 0.92] as CFDictionary)
        guard CGImageDestinationFinalize(destination) else { return nil }
        return (file, true)
    }

    public func picture(for url: URL, maxPixel: Int) async -> CGImage? {
        guard let identifier = FindPhotos.identifier(of: url) else { return nil }
        return await Self.image(identifier: identifier, maxPixel: maxPixel)
    }

    public func observeChanges(_ handler: @escaping @Sendable () -> Void) {
        let first = self.handler.withLock { current -> Bool in
            defer { current = handler }
            return current == nil
        }
        if first { PHPhotoLibrary.shared().register(self) }
    }

    public func photoLibraryDidChange(_ changeInstance: PHChange) {
        handler.withLock { $0 }?()
    }

    // MARK: PhotoKit

    struct Unchecked<T>: @unchecked Sendable {
        let value: T
        init(_ value: T) { self.value = value }
    }

    /// The encoded bytes of a photo's current version (edits applied) and a
    /// file extension for them, if they are on this Mac. Files over 60 MB
    /// (huge panoramas, RAW) are left to the preview path.
    static func originalData(identifier: String) async -> (data: Data, ext: String)? {
        await Task.detached(priority: Task.currentPriority) { () -> Unchecked<(Data, String)?> in
            guard let asset = PHAsset.fetchAssets(withLocalIdentifiers: [identifier], options: nil).firstObject else {
                return Unchecked(nil)
            }
            let options = PHImageRequestOptions()
            options.isSynchronous = true
            options.version = .current
            options.deliveryMode = .highQualityFormat
            options.isNetworkAccessAllowed = false
            var result: (Data, String)?
            PHImageManager.default().requestImageDataAndOrientation(for: asset, options: options) { data, type, _, _ in
                guard let data, data.count < 60 << 20 else { return }
                let ext = type.flatMap { UTType($0)?.preferredFilenameExtension } ?? "heic"
                result = (data, ext)
            }
            return Unchecked(result)
        }.value.value.map { (data: $0.0, ext: $0.1) }
    }

    /// The best local picture of an asset, at most `maxPixel` on its long
    /// side; never downloads.
    static func image(identifier: String, maxPixel: Int) async -> CGImage? {
        await Task.detached(priority: Task.currentPriority) { () -> Unchecked<CGImage?> in
            guard let asset = PHAsset.fetchAssets(withLocalIdentifiers: [identifier], options: nil).firstObject else {
                return Unchecked(nil)
            }
            let size = CGSize(width: maxPixel, height: maxPixel)
            for mode in [PHImageRequestOptionsDeliveryMode.highQualityFormat, .fastFormat] {
                let options = PHImageRequestOptions()
                options.isSynchronous = true
                options.deliveryMode = mode
                options.resizeMode = .exact
                options.isNetworkAccessAllowed = false
                var result: CGImage?
                PHImageManager.default().requestImage(for: asset, targetSize: size, contentMode: .aspectFit,
                                                      options: options) { image, _ in
                    var rect = CGRect(origin: .zero, size: image?.size ?? .zero)
                    result = image?.cgImage(forProposedRect: &rect, context: nil, hints: nil)
                }
                if let result { return Unchecked(result) }
            }
            return Unchecked(nil)
        }.value.value
    }

    /// The local file of a video, if it is on this Mac.
    static func videoFile(identifier: String) async -> URL? {
        guard let asset = PHAsset.fetchAssets(withLocalIdentifiers: [identifier], options: nil).firstObject else {
            return nil
        }
        let options = PHVideoRequestOptions()
        options.isNetworkAccessAllowed = false
        options.version = .current
        return await withCheckedContinuation { continuation in
            PHImageManager.default().requestAVAsset(forVideo: asset, options: options) { avAsset, _, _ in
                continuation.resume(returning: (avAsset as? AVURLAsset)?.url)
            }
        }
    }

    /// A file to open: a photo written out as a JPEG copy, or a video's file.
    static func viewableFile(identifier: String) async -> URL? {
        guard let asset = PHAsset.fetchAssets(withLocalIdentifiers: [identifier], options: nil).firstObject else {
            return nil
        }
        if asset.mediaType == .video { return await videoFile(identifier: identifier) }
        guard let image = await image(identifier: identifier, maxPixel: 4_096) else { return nil }
        let folder = FileManager.default.temporaryDirectory.appending(path: "MferenceFindOpen", directoryHint: .isDirectory)
        try? FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let file = folder.appending(path: "\(FindPhotos.name(kind: .image, isScreenshot: false, date: asset.creationDate)).jpg")
        guard let destination = CGImageDestinationCreateWithURL(file as CFURL, UTType.jpeg.identifier as CFString, 1, nil)
        else { return nil }
        CGImageDestinationAddImage(destination, image, [kCGImageDestinationLossyCompressionQuality: 0.9] as CFDictionary)
        return CGImageDestinationFinalize(destination) ? file : nil
    }
}

/// Reads files as the base reader does, and Photos library items by first
/// getting a readable copy from the library: photos as temporary pictures
/// for the reader (in the helper, like any image), videos from their own
/// file in this process (the helper cannot open files inside the library).
public struct FindLibraryReader: FindExtracting {
    let base: any FindExtracting
    let library: any FindPhotoLibrary

    public init(base: any FindExtracting, library: any FindPhotoLibrary) {
        self.base = base
        self.library = library
    }

    public func extract(_ url: URL, kind: FindKind, options: FindExtractOptions) async -> FindExtraction {
        guard FindPhotos.isLibraryItem(url) else { return await base.extract(url, kind: kind, options: options) }
        guard let file = await library.readableFile(for: url, kind: kind, maxPixel: 2_048) else {
            return .skipped(reason: "Not on this Mac yet (kept in iCloud): searched by date only.")
        }
        defer { if file.isTemporary { try? FileManager.default.removeItem(at: file.url) } }
        return kind == .video
            ? await InProcessFindExtractor(timeout: .seconds(300)).extract(file.url, kind: kind, options: options)
            : await base.extract(file.url, kind: kind, options: options)
    }

    public func prepare(_ url: URL, kind: FindKind, pieces: [FindMediaPiece]) async -> [URL?] {
        guard FindPhotos.isLibraryItem(url) else { return await base.prepare(url, kind: kind, pieces: pieces) }
        if kind == .image {
            // Photos hands over the picture decoded: size it for the encoder
            // here, with no file for a reader to decode again.
            return await Task.detached(priority: Task.currentPriority) { [library] in
                var files: [URL?] = []
                for piece in pieces {
                    let maxPixel = piece.tokens >= FindMedia.screenshotTokens ? 1_024 : 768
                    guard let image = await library.picture(for: url, maxPixel: maxPixel),
                          let data = FindMedia.rgb(image, tokens: piece.tokens) else {
                        files.append(nil)
                        continue
                    }
                    try? FileManager.default.createDirectory(at: FindMediaFolder.url, withIntermediateDirectories: true)
                    let file = FindMediaFolder.url.appending(path: "\(UUID().uuidString).fmd")
                    files.append((try? data.write(to: file)) == nil ? nil : file)
                }
                return files
            }.value
        }
        guard let file = await library.readableFile(for: url, kind: kind, maxPixel: 1_024) else {
            return pieces.map { _ in nil }
        }
        defer { if file.isTemporary { try? FileManager.default.removeItem(at: file.url) } }
        if kind == .video {
            return await InProcessFindExtractor().prepare(file.url, kind: kind, pieces: pieces)
        }
        return await base.prepare(file.url, kind: kind, pieces: pieces)
    }

    public var parallelism: Int { base.parallelism }

    public func setHurry(_ hurry: Bool) async {
        await base.setHurry(hurry)
    }

    public func relax() async {
        await base.relax()
    }

    public func shutdown() async {
        await base.shutdown()
    }
}

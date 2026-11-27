import AppKit
import AVKit
import MferenceAppCore
import MferenceDesign
import QuickLookThumbnailing
import SwiftUI

/// Layout sizes for Find's screens (per-screen metrics, CONTRACTS.md).
enum FindMetrics {
    static let filterWidth: CGFloat = 220
    static let previewWidth: CGFloat = 360
    static let tileSize = CGSize(width: 30, height: 36)
    static let tileRadius: CGFloat = 6
    static let searchHeight: CGFloat = 42
    static let panelSize = CGSize(width: 960, height: 600)
    static let panelListWidth: CGFloat = 440
    static let thumbnail = CGSize(width: 300, height: 220)
    static let audioHeight: CGFloat = 56
    static let setupMaxWidth: CGFloat = 540
    static let setupSideWidth: CGFloat = 400
    static let setupIcon: CGFloat = 52
    static let stepNumber: CGFloat = 26
}

/// The coloured tile with the file's extension.
struct FindKindTile: View {
    let kind: FindKind
    let url: URL

    var body: some View {
        Text(kind.tileLabel(for: url))
            .font(MFFont.micro)
            .foregroundStyle(foreground)
            .frame(width: FindMetrics.tileSize.width, height: FindMetrics.tileSize.height, alignment: .bottom)
            .padding(.bottom, MFSpacing.xs)
            .background(background, in: RoundedRectangle(cornerRadius: FindMetrics.tileRadius))
            .accessibilityHidden(true)
    }

    private var tone: MFTone {
        switch kind {
        case .pdf: .preview
        case .document: .accent
        case .image: .success
        case .code: .preview
        case .slides, .sheet: .success
        case .text: .neutral
        case .video: .preview
        case .audio: .accent
        }
    }

    private var foreground: Color { tone == .neutral ? MFColor.secondary : tone.tileForeground }
    private var background: Color { tone.tileBackground }
}

extension MFTone {
    var tileForeground: Color {
        switch self {
        case .neutral: MFColor.secondary
        case .success: MFColor.success
        case .accent: MFColor.accentStrong
        case .preview: MFColor.preview
        case .ink: MFColor.onInk
        }
    }

    var tileBackground: Color {
        switch self {
        case .neutral: MFColor.fill
        case .success: MFColor.successFill
        case .accent: MFColor.accentFill
        case .preview: MFColor.previewFill
        case .ink: MFColor.ink
        }
    }
}

/// A match reason as a chip ("Meaning", "Words", "Text in image").
struct FindReasonChip: View {
    let reason: FindMatchReason

    var body: some View {
        MFChip(reason.title, tone: tone)
    }

    private var tone: MFTone {
        switch reason {
        case .meaning: .accent
        case .words: .neutral
        case .name: .preview
        case .textInImage: .success
        case .picture: .success
        case .sound: .accent
        }
    }
}

/// The passage with its hit highlighted.
struct FindSnippet: View {
    let result: FindResult
    var lineLimit = 2

    /// Before and after in secondary, the hit in bold ink.
    private var attributed: AttributedString {
        var hit = AttributedString(result.hit)
        hit.foregroundColor = MFColor.ink
        hit.inlinePresentationIntent = .stronglyEmphasized
        return AttributedString(result.before) + hit + AttributedString(result.after)
    }

    var body: some View {
        Text(attributed)
            .font(MFFont.preview)
            .foregroundStyle(MFColor.secondary)
            .lineLimit(lineLimit)
            .fixedSize(horizontal: false, vertical: true)
    }
}

/// One result row: tile, name, date, folder, passage, reasons.
struct FindResultRow: View {
    let result: FindResult
    let isSelected: Bool

    var body: some View {
        HStack(alignment: .top, spacing: MFSpacing.m) {
            FindKindTile(kind: result.kind, url: result.url)
            VStack(alignment: .leading, spacing: MFSpacing.xxs + 1) {
                HStack(alignment: .firstTextBaseline, spacing: MFSpacing.s) {
                    Text(result.name)
                        .font(MFFont.rowTitle)
                        .foregroundStyle(MFColor.ink)
                        .lineLimit(1)
                        .truncationMode(.middle)
                    Spacer(minLength: MFSpacing.xs)
                    Text(FindDates.short(result.modified))
                        .font(MFFont.caption)
                        .foregroundStyle(MFColor.tertiary)
                }
                Text(Self.folder(of: result.url))
                    .font(MFFont.caption)
                    .foregroundStyle(MFColor.tertiary)
                    .lineLimit(1)
                    .truncationMode(.middle)
                if !result.hit.isEmpty {
                    FindSnippet(result: result)
                }
                HStack(spacing: MFSpacing.xs) {
                    ForEach(result.reasons, id: \.self) { FindReasonChip(reason: $0) }
                    if let location = result.location { MFChip(location) }
                    if result.copies > 0 { MFChip(result.copies == 1 ? "1 copy" : "\(result.copies) copies") }
                }
            }
        }
        .padding(.horizontal, MFSpacing.m)
        .padding(.vertical, MFSpacing.compact)
        .background(isSelected ? MFColor.accentFillFaint : .clear, in: RoundedRectangle(cornerRadius: MFRadius.m))
        .contentShape(Rectangle())
        .accessibilityElement(children: .combine)
        .accessibilityAddTraits(isSelected ? .isSelected : [])
    }

    /// "~/Documents/Home" for the row's second line.
    static func folder(of url: URL) -> String {
        if FindPhotos.isLibraryItem(url) { return "Photos library" }
        let folder = url.deletingLastPathComponent().path
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        return folder.hasPrefix(home) ? "~" + folder.dropFirst(home.count) : folder
    }
}

/// The preview pane: a thumbnail of the file and the passage that matched.
struct FindPreview: View {
    let result: FindResult
    let askAbout: (URL) -> Void
    @State private var thumbnail: NSImage?

    var body: some View {
        VStack(alignment: .leading, spacing: MFSpacing.m) {
            HStack(spacing: MFSpacing.compact) {
                FindKindTile(kind: result.kind, url: result.url)
                VStack(alignment: .leading, spacing: MFSpacing.xxs) {
                    Text(result.name).font(MFFont.headline).foregroundStyle(MFColor.ink).lineLimit(2)
                    Text([result.location, FindDates.short(result.modified)].compactMap { $0 }.joined(separator: " · "))
                        .font(MFFont.caption).foregroundStyle(MFColor.tertiary)
                }
            }
            if result.kind == .audio || (result.kind == .video && result.time != nil) {
                FindMediaPreview(url: result.url, time: result.time ?? 0, isVideo: result.kind == .video)
                    .frame(maxWidth: .infinity)
                    .frame(height: result.kind == .video ? FindMetrics.thumbnail.height : FindMetrics.audioHeight)
                    .clipShape(RoundedRectangle(cornerRadius: MFRadius.s))
                    .overlay(RoundedRectangle(cornerRadius: MFRadius.s).strokeBorder(MFColor.hairline))
            } else if let thumbnail {
                Image(nsImage: thumbnail)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(maxWidth: .infinity, maxHeight: FindMetrics.thumbnail.height)
                    .clipShape(RoundedRectangle(cornerRadius: MFRadius.s))
                    .overlay(RoundedRectangle(cornerRadius: MFRadius.s).strokeBorder(MFColor.hairline))
                    .accessibilityLabel("Preview of \(result.name)")
            }
            if !result.passage.isEmpty {
                ScrollView {
                    passage
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(MFSpacing.comfortable)
                }
                .background(MFColor.surface, in: RoundedRectangle(cornerRadius: MFRadius.s))
                .overlay(RoundedRectangle(cornerRadius: MFRadius.s).strokeBorder(MFColor.hairline))
            } else {
                Spacer(minLength: 0)
            }
            Text(Self.reasonLine(result))
                .font(MFFont.caption)
                .foregroundStyle(MFColor.secondary)
                .fixedSize(horizontal: false, vertical: true)
            FindActions(result: result, askAbout: askAbout)
        }
        .task(id: result.url) {
            thumbnail = result.kind == .audio || (result.kind == .video && result.time != nil)
                ? nil : await Self.thumbnail(for: result.url)
        }
    }

    /// The whole passage, with the hit in ink.
    private var passage: some View {
        let text = result.passage
        let attributed: AttributedString = {
            var attributed = AttributedString(text)
            let hit = result.hit.hasSuffix("…") ? String(result.hit.dropLast()) : result.hit
            if !hit.isEmpty, let range = attributed.range(of: hit.prefix(120)) {
                attributed[range].foregroundColor = MFColor.ink
                attributed[range].backgroundColor = MFColor.accentFill
            }
            return attributed
        }()
        return Text(attributed)
            .font(MFFont.preview)
            .foregroundStyle(MFColor.secondary)
            .lineSpacing(MFLineSpacing.normal)
            .textSelection(.enabled)
    }

    static func reasonLine(_ result: FindResult) -> String {
        let at = result.location.map { " at \($0)" } ?? ""
        if result.reasons.contains(.picture) {
            return result.kind == .video
                ? "Matched by what the video shows\(at), seen by the search model on this Mac."
                : "Matched by what the picture shows, seen by the search model on this Mac."
        }
        if result.reasons.contains(.sound) {
            return "Matched by what it sounds like\(at), heard by the search model on this Mac."
        }
        if result.reasons.contains(.textInImage) {
            return "The words were read from this image with Vision, on this Mac."
        }
        if result.reasons.contains(.meaning), !result.reasons.contains(.words) {
            return "Matched by meaning: the passage is about what you asked, in other words."
        }
        if result.reasons.contains(.meaning) { return "Matched by meaning and by your words." }
        if result.reasons == [.name] { return "Matched by the file's name." }
        return "Matched by your words."
    }

    static func thumbnail(for url: URL) async -> NSImage? {
        if FindPhotos.isLibraryItem(url) {
            let size = Int(max(FindMetrics.thumbnail.width, FindMetrics.thumbnail.height) * 2)
            return await FindPhotos.image(for: url, maxPixel: size).map { NSImage(cgImage: $0, size: .zero) }
        }
        let request = QLThumbnailGenerator.Request(fileAt: url, size: FindMetrics.thumbnail,
                                                   scale: 2, representationTypes: .thumbnail)
        return try? await QLThumbnailGenerator.shared.generateBestRepresentation(for: request).nsImage
    }
}

/// A video or recording, cued to where it matched; plays in place.
struct FindMediaPreview: View {
    let url: URL
    let time: Double
    let isVideo: Bool
    @State private var player: AVPlayer?

    var body: some View {
        ZStack {
            MFColor.surface
            if let player {
                VideoPlayer(player: player)
                    .accessibilityLabel(isVideo ? "Video, cued to \(FindMedia.clock(time))"
                                        : "Recording, cued to \(FindMedia.clock(time))")
            }
        }
        .task(id: "\(url.path)#\(time)") {
            let player: AVPlayer
            if FindPhotos.isLibraryItem(url) {
                guard let item = await FindPhotos.playerItem(for: url) else { return }
                player = AVPlayer(playerItem: item)
            } else {
                player = AVPlayer(url: url)
            }
            await player.seek(to: CMTime(seconds: time, preferredTimescale: 600),
                              toleranceBefore: .zero, toleranceAfter: .zero)
            self.player = player
        }
        .onDisappear { player?.pause() }
    }
}

/// Open, Show in Finder, Ask about this.
struct FindActions: View {
    let result: FindResult
    let askAbout: (URL) -> Void

    var body: some View {
        HStack(spacing: MFSpacing.s) {
            Button("Open") { FindFiles.open(result.url) }
                .buttonStyle(.mfPrimary)
                .keyboardShortcut(.return, modifiers: [])
            Button(FindPhotos.isLibraryItem(result.url) ? "Show in Photos" : "Show in Finder") {
                FindFiles.reveal(result.url)
            }
            .buttonStyle(.mfSecondary)
            .keyboardShortcut(.return, modifiers: .command)
            if !FindPhotos.isLibraryItem(result.url) {
                Button {
                    askAbout(result.url)
                } label: {
                    Label("Ask about this", systemImage: "bubble.left")
                }
                .buttonStyle(.mfSecondary)
                .keyboardShortcut("j", modifiers: .command)
            }
        }
    }
}

/// Opening and revealing files.
enum FindFiles {
    static func open(_ url: URL) {
        if FindPhotos.isLibraryItem(url) {
            Task { await FindPhotos.open(url) }
        } else {
            NSWorkspace.shared.open(url)
        }
    }

    static func reveal(_ url: URL) {
        if FindPhotos.isLibraryItem(url) {
            FindPhotos.showInPhotos()
        } else {
            NSWorkspace.shared.activateFileViewerSelecting([url])
        }
    }
}

/// The banner while Find is still reading or learning.
struct FindStatusBanner: View {
    let status: FindIndexStatus
    let modelState: FindModelState
    var mediaModelState: FindModelState = .notDownloaded

    var body: some View {
        if let line {
            HStack(alignment: .top, spacing: MFSpacing.s) {
                if case .downloading(let fraction) = modelState {
                    MFProgressBar(fraction, label: "Downloading the search model")
                        .frame(width: 80)
                } else if case .downloading(let fraction) = mediaModelState {
                    MFProgressBar(fraction, label: "Downloading the photo, video and audio model")
                        .frame(width: 80)
                } else if status.phase == .learning {
                    MFProgressBar(status.learnedFraction, label: "Learning meaning")
                        .frame(width: 80)
                }
                Text(line)
                    .font(MFFont.caption)
                    .foregroundStyle(MFColor.accentStrong)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .padding(.horizontal, MFSpacing.m)
            .padding(.vertical, MFSpacing.s)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(MFColor.accentFillFaint, in: RoundedRectangle(cornerRadius: MFRadius.m))
        }
    }

    private var line: String? {
        switch modelState {
        case .downloading(let fraction):
            return "Downloading the search model (\(FindModel.sizeLabel), \(Int(fraction * 100))%). Word search already works."
        case .verifying:
            return "Checking the search model…"
        case .failed(let message):
            return "The search model could not be downloaded: \(message) Word search still works."
        case .waitingForOfflineMode:
            return "The search model downloads when Offline mode is off. Word search already works."
        default:
            break
        }
        if case .downloading(let fraction) = mediaModelState {
            return "Downloading the photo, video and audio model (\(FindModelFile.media.sizeLabel), "
                + "\(Int(fraction * 100))%). Everything else already works."
        }
        if case .waitingForOfflineMode = mediaModelState {
            return "The photo, video and audio model downloads when Offline mode is off. Everything else already works."
        }
        return status.line
    }
}

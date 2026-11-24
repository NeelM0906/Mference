import AppKit
import MferenceAppCore
import MferenceDesign
import SwiftUI
import UniformTypeIdentifiers

/// Settings > Models: the model library (`08b-model-library.png`). Installed
/// and downloading models, the active downloads, adding a model file, and
/// model behaviour (default model, MTP, idle unload). Deep downloads from
/// Settings > Models after approval (L4); browsing other models is Later.
public struct ModelLibraryView: View {
    let library: any ModelLibraryModel
    /// False in offline mode: downloads are off and say why.
    let downloadsAllowed: Bool
    @State private var detailsID: String?
    /// A download waiting for the person's approval (the one dialog every
    /// Deep download goes through, L4).
    @State private var pendingDownload: ModelEntry?
    /// An image-understanding download waiting for approval.
    @State private var pendingImageDownload: ModelEntry?
    @State private var importing = false
    @State private var importFailure: ImportFailure?

    public init(library: any ModelLibraryModel, downloadsAllowed: Bool = true) {
        self.library = library
        self.downloadsAllowed = downloadsAllowed
    }

    public var body: some View {
        MFScroll {
            VStack(alignment: .leading, spacing: MFSpacing.roomy) {
                header
                ForEach(library.entries.filter { $0.state.transfer != nil }) { entry in
                    DownloadBanner(entry: entry, library: library, downloadsAllowed: downloadsAllowed)
                }
                modelTable
                behaviour
            }
            .padding(.horizontal, LibraryMetrics.pagePadding.width)
            .padding(.vertical, LibraryMetrics.pagePadding.height)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .sheet(item: Binding(get: { detailsID.map(DetailsTarget.init) },
                             set: { detailsID = $0?.id })) { target in
            ModelDetailsSheet(library: library, id: target.id)
        }
        .confirmationDialog(
            pendingDownload.map { "Download \($0.title)?" } ?? "",
            isPresented: Binding(get: { pendingDownload != nil }, set: { if !$0 { pendingDownload = nil } }),
            presenting: pendingDownload
        ) { entry in
            Button("Download") { Self.approveDownload(entry, in: library) }
            Button("Not now", role: .cancel) {}
        } message: { entry in
            Text(Self.approvalMessage(for: entry))
        }
        .confirmationDialog(
            pendingImageDownload.map { "Download image understanding for \($0.productName)?" } ?? "",
            isPresented: Binding(get: { pendingImageDownload != nil },
                                 set: { if !$0 { pendingImageDownload = nil } }),
            presenting: pendingImageDownload
        ) { entry in
            Button("Download") { Self.approveImageUnderstanding(entry, in: library) }
            Button("Not now", role: .cancel) {}
        } message: { entry in
            Text(Self.imageUnderstandingApprovalMessage(for: entry))
        }
        .fileImporter(isPresented: $importing, allowedContentTypes: Self.modelFileTypes) { result in
            guard case .success(let url) = result else { return }
            Task {
                do { try await library.addModelFile(at: url) } catch { importFailure = ImportFailure(error) }
            }
        }
        .alert("The model file was not added",
               isPresented: Binding(get: { importFailure != nil }, set: { if !$0 { importFailure = nil } }),
               presenting: importFailure) { failure in
            if let file = failure.fileToShow {
                Button("Show in Finder") { NSWorkspace.shared.activateFileViewerSelecting([file]) }
            }
            Button("OK", role: .cancel) {}
        } message: { failure in
            Text(failure.message)
        }
    }

    static let modelFileTypes: [UTType] = [UTType(filenameExtension: "gguf") ?? .data]

    /// The approval dialog's "Download" (here and in Settings > Models): the
    /// only way a download that needs approval starts or restarts. Checks the
    /// disk margin. Returns whether it started.
    @discardableResult
    static func approveDownload(_ entry: ModelEntry, in library: any ModelLibraryModel) -> Bool {
        switch entry.state {
        case .notInstalled, .downloadFailed: break
        default: return false
        }
        guard library.machine.hasRoom(for: entry.sizeBytes) else { return false }
        library.download(entry.id)
        return true
    }

    /// Retry after a failed download without a dialog: only for Quick, which
    /// installs without approval. Every other model (Deep) returns false here
    /// and goes through the approval dialog (`approveDownload`) again.
    /// Checks the disk margin. Returns whether it restarted.
    @discardableResult
    static func retryDownload(_ entry: ModelEntry, in library: any ModelLibraryModel) -> Bool {
        guard entry.installsWithoutApproval, case .downloadFailed = entry.state,
              library.machine.hasRoom(for: entry.sizeBytes) else { return false }
        library.download(entry.id)
        return true
    }

    /// The image-understanding dialog's "Download" (here and in the chat's
    /// offer): the only way that download starts or restarts. Checks the
    /// disk margin. Returns whether it started.
    @discardableResult
    static func approveImageUnderstanding(_ entry: ModelEntry, in library: any ModelLibraryModel) -> Bool {
        guard let addOn = entry.imageUnderstanding, entry.state.isInstalled else { return false }
        switch addOn.state {
        case .notInstalled, .downloadFailed, .paused: break
        default: return false
        }
        guard library.machine.hasRoom(for: addOn.sizeBytes) else { return false }
        library.downloadImageUnderstanding(entry.id)
        return true
    }

    static func imageUnderstandingApprovalMessage(for entry: ModelEntry) -> String {
        let size = entry.imageUnderstanding.map { " (\(ModelFormat.bytes($0.sizeBytes)))" } ?? ""
        return "\(entry.productName)’s image understanding downloads\(size) to this Mac and runs here. "
            + "Pictures you attach are then seen by the model. Nothing is sent anywhere."
    }

    static func approvalMessage(for entry: ModelEntry) -> String {
        let size = entry.sizeBytes.map { " (\(ModelFormat.bytes($0)))" } ?? ""
        return "\(entry.productName) downloads\(size) to this Mac and runs here. Nothing else is sent."
    }

    /// Models not installed that may be offered on this Mac (L4 via `isOffered`).
    static func downloadable(_ entries: [ModelEntry], on machine: MachineSummary) -> [ModelEntry] {
        entries.filter { entry in
            switch entry.state {
            case .notInstalled, .downloadFailed: entry.isOffered(on: machine)
            default: false
            }
        }
    }

    private var header: some View {
        HStack(alignment: .bottom, spacing: MFSpacing.m) {
            VStack(alignment: .leading, spacing: MFSpacing.xs) {
                Text("Models")
                    .font(MFFont.title)
                    .foregroundStyle(MFColor.ink)
                    .accessibilityAddTraits(.isHeader)
                Text(ModelFormat.machineLine(library.machine)
                     + " One model is loaded at a time; switching unloads the current one first.")
                    .font(MFFont.callout)
                    .foregroundStyle(MFColor.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            Button("Add a model file") { importing = true }
                .buttonStyle(.mfSecondary)
                .help("Add a GGUF model file from this Mac.")
            browseMenu
        }
    }

    /// Only Quick and Deep download in this version (Deep from Settings >
    /// Models); browsing other models comes later.
    private var browseMenu: some View {
        HStack(spacing: MFSpacing.s) {
            Button("Browse models") {}
                .buttonStyle(.mfSecondary)
                .disabled(true)
            MFBadge(.later)
        }
        .help("Coming later")
        .accessibilityElement(children: .combine)
        .accessibilityLabel("Browse models, coming later")
    }

    private var modelTable: some View {
        MFCard {
            HStack(spacing: MFSpacing.m) {
                MFSectionHeader("Model").frame(maxWidth: .infinity, alignment: .leading)
                MFSectionHeader("On disk").frame(width: LibraryMetrics.sizeColumn, alignment: .leading)
                MFSectionHeader("Status").frame(width: LibraryMetrics.statusColumn, alignment: .leading)
                Color.clear.frame(width: LibraryMetrics.actionColumn, height: MFLayout.hairline)
            }
            .padding(.horizontal, MFSpacing.l)
            .padding(.vertical, MFSpacing.s)
            .background(MFColor.canvas)
            // Installed, downloading and failed models, plus any model this
            // Mac is offered (L4) but has not downloaded: its Download opens
            // the approval dialog.
            ForEach(library.entries.filter { $0.state != .notInstalled || $0.isOffered(on: library.machine) }) { entry in
                ModelTableRow(entry: entry,
                              canRetry: downloadsAllowed && library.machine.hasRoom(for: entry.sizeBytes),
                              onDetails: { detailsID = entry.id },
                              onRetry: {
                                  if entry.installsWithoutApproval, case .downloadFailed = entry.state {
                                      Self.retryDownload(entry, in: library)
                                  } else {
                                      pendingDownload = entry
                                  }
                              })
                if entry.state.isInstalled, let addOn = entry.imageUnderstanding {
                    ImageUnderstandingLine(
                        entry: entry, addOn: addOn,
                        canDownload: downloadsAllowed && library.machine.hasRoom(for: addOn.sizeBytes),
                        onDownload: { pendingImageDownload = entry },
                        onPause: { library.pauseImageUnderstanding(entry.id) },
                        onRemove: { library.removeImageUnderstanding(entry.id) })
                }
            }
        }
    }

    private var behaviour: some View {
        let defaultTitle = library.entries.first { $0.id == library.defaultModelID }?.title ?? "None"
        let deepInstalled = library.entries.contains { $0.id == AppModel.deep.modelID && $0.state.isInstalled }
        return MFCard {
            MFListRow("Default model", subtitle: "Used for new chats, mail drafts and scanning.") {
                Menu {
                    ForEach(library.entries.filter(\.state.isInstalled)) { entry in
                        Button(entry.title) { library.defaultModelID = entry.id }
                    }
                } label: {
                    Text(defaultTitle)
                }
                .menuStyle(.button)
                .buttonStyle(.mfSecondary)
                .fixedSize()
                .accessibilityLabel("Default model")
                .accessibilityValue(defaultTitle)
            }
            if deepInstalled {
                MFListRow("Faster answers from Deep (MTP)",
                          subtitle: "Drafts several tokens per step and checks them. Same quality, less waiting.") {
                    MFBadge(.custom("On", .success))
                }
            }
            MFListRow("Unload when idle",
                      subtitle: "Frees memory after \(Self.idleText(library.idleUnloadMinutes)) without use.") {
                Menu {
                    ForEach(LibraryMetrics.idleChoices, id: \.self) { minutes in
                        Button(Self.idleText(minutes)) { library.idleUnloadMinutes = minutes }
                    }
                } label: {
                    Text(Self.idleShort(library.idleUnloadMinutes))
                }
                .menuStyle(.button)
                .buttonStyle(.mfQuiet)
                .fixedSize()
                .accessibilityLabel("Unload when idle")
                .accessibilityValue(Self.idleText(library.idleUnloadMinutes))
            }
        }
    }

    static func idleText(_ minutes: Int) -> String {
        if minutes == 1 { return "1 minute" }
        if minutes == 60 { return "1 hour" }
        if minutes > 60, minutes % 60 == 0 { return "\(minutes / 60) hours" }
        return "\(minutes) minutes"
    }

    static func idleShort(_ minutes: Int) -> String {
        minutes >= 60 && minutes % 60 == 0 ? "\(minutes / 60) h" : "\(minutes) min"
    }
}

/// What the import alert shows: the error's message and, when the error
/// points at a file (U2's `ModelImportError` via `FileRevealingError`), a
/// "Show in Finder" for it.
struct ImportFailure {
    let message: String
    let fileToShow: URL?

    init(_ error: any Error) {
        message = error.localizedDescription
        fileToShow = (error as? any FileRevealingError)?.fileToShow
    }
}

/// `sheet(item:)` needs an Identifiable; Details looks the entry up live.
struct DetailsTarget: Identifiable {
    let id: String
}

enum LibraryMetrics {
    static let pagePadding = CGSize(width: 30, height: 22)
    static let sizeColumn: CGFloat = 170
    static let statusColumn: CGFloat = 200
    static let actionColumn: CGFloat = 96
    static let detailsWidth: CGFloat = 440
    static let idleChoices = [5, 10, 30, 60]
}

/// One installed, downloading or failed model.
struct ModelTableRow: View {
    let entry: ModelEntry
    /// Downloads are allowed (not offline) and the disk has room (Download
    /// and Retry).
    let canRetry: Bool
    let onDetails: () -> Void
    let onRetry: () -> Void

    var body: some View {
        HStack(spacing: MFSpacing.m) {
            VStack(alignment: .leading, spacing: MFSpacing.xxs) {
                Text(entry.title)
                    .font(MFFont.calloutSemibold)
                    .foregroundStyle(MFColor.ink)
                Text(entry.summary)
                    .font(MFFont.caption)
                    .foregroundStyle(MFColor.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                // Why a load or download failed, so the person knows what
                // to fix before Retry (the badge stays short).
                if let reason = Self.failureLine(entry.state) {
                    Text(reason)
                        .font(MFFont.caption)
                        .foregroundStyle(MFColor.preview)
                        .fixedSize(horizontal: false, vertical: true)
                        .textSelection(.enabled)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            Text(entry.sizeBytes.map(ModelFormat.bytes) ?? "—")
                .font(MFFont.callout)
                .foregroundStyle(MFColor.ink)
                .lineLimit(1)
                .frame(width: LibraryMetrics.sizeColumn, alignment: .leading)
            HStack {
                ModelStatusBadge(state: entry.state)
            }
            .frame(width: LibraryMetrics.statusColumn, alignment: .leading)
            Group {
                if case .downloadFailed = entry.state {
                    Button("Retry", action: onRetry)
                        .buttonStyle(.mfSecondary)
                        .disabled(!canRetry)
                        .help(canRetry ? "Download \(entry.title) again" : "Offline mode is on, or the disk needs more free space.")
                        .accessibilityLabel("Retry downloading \(entry.title)")
                } else if entry.state == .notInstalled {
                    Button("Download…", action: onRetry)
                        .buttonStyle(.mfSecondary)
                        .disabled(!canRetry)
                        .help(canRetry ? "Download \(entry.title), after you approve" : "Offline mode is on, or the disk needs more free space.")
                        .accessibilityLabel("Download \(entry.title)")
                } else {
                    Button("Details", action: onDetails)
                        .buttonStyle(.mfSecondary)
                        .accessibilityLabel("Details for \(entry.title)")
                }
            }
            .frame(width: LibraryMetrics.actionColumn, alignment: .trailing)
        }
        .padding(.horizontal, MFSpacing.l)
        .padding(.vertical, MFSpacing.m)
        .accessibilityElement(children: .contain)
        .accessibilityLabel(entry.title)
    }
}

extension ModelTableRow {
    /// "Didn't load: ..." / "Download failed: ...", or nil when nothing failed.
    static func failureLine(_ state: ModelEntryState) -> String? {
        switch state {
        case .loadFailed(let message): "Didn't load: \(message)"
        case .downloadFailed(let message): "Download failed: \(message)"
        default: nil
        }
    }
}

/// "Image understanding" under Quick or Deep: download (after approval),
/// pause, resume or remove the model's vision projector.
struct ImageUnderstandingLine: View {
    let entry: ModelEntry
    let addOn: ImageUnderstandingEntry
    /// Downloads are allowed (not offline) and the disk has room.
    let canDownload: Bool
    let onDownload: () -> Void
    let onPause: () -> Void
    let onRemove: () -> Void

    var body: some View {
        HStack(spacing: MFSpacing.m) {
            VStack(alignment: .leading, spacing: MFSpacing.xxs) {
                Text(ImageUnderstandingEntry.title)
                    .font(MFFont.callout)
                    .foregroundStyle(MFColor.ink)
                Text(ImageUnderstandingEntry.summary)
                    .font(MFFont.caption)
                    .foregroundStyle(MFColor.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                if case .downloadFailed(let message) = addOn.state {
                    Text("Download failed: \(message)")
                        .font(MFFont.caption)
                        .foregroundStyle(MFColor.preview)
                        .fixedSize(horizontal: false, vertical: true)
                }
                if let progress = addOn.state.transfer {
                    MFProgressBar(progress.fraction, label: "Downloading image understanding for \(entry.productName)")
                        .padding(.top, MFSpacing.xs)
                }
            }
            .padding(.leading, MFSpacing.l)
            .frame(maxWidth: .infinity, alignment: .leading)
            Text(ModelFormat.bytes(addOn.sizeBytes))
                .font(MFFont.callout)
                .foregroundStyle(MFColor.ink)
                .lineLimit(1)
                .frame(width: LibraryMetrics.sizeColumn, alignment: .leading)
            HStack {
                MFBadge(Self.badge(addOn.state))
            }
            .frame(width: LibraryMetrics.statusColumn, alignment: .leading)
            action
                .frame(width: LibraryMetrics.actionColumn, alignment: .trailing)
        }
        .padding(.horizontal, MFSpacing.l)
        .padding(.bottom, MFSpacing.m)
        .accessibilityElement(children: .contain)
        .accessibilityLabel("\(ImageUnderstandingEntry.title) for \(entry.productName)")
    }

    @ViewBuilder private var action: some View {
        switch addOn.state {
        case .ready, .loaded, .loading, .loadFailed:
            Button("Remove", action: onRemove)
                .buttonStyle(.mfQuiet)
                .accessibilityLabel("Remove image understanding for \(entry.productName)")
        case .downloading:
            Button("Pause", action: onPause)
                .buttonStyle(.mfSecondary)
                .accessibilityLabel("Pause downloading image understanding for \(entry.productName)")
        case .paused, .notInstalled, .downloadFailed:
            Button(Self.downloadTitle(addOn.state), action: onDownload)
                .buttonStyle(.mfSecondary)
                .disabled(!canDownload)
                .help(canDownload ? "Download image understanding, after you approve"
                                  : "Offline mode is on, or the disk needs more free space.")
                .accessibilityLabel("\(Self.downloadTitle(addOn.state)) image understanding for \(entry.productName)")
        }
    }

    static func downloadTitle(_ state: ModelEntryState) -> String {
        switch state {
        case .paused: "Resume…"
        case .downloadFailed: "Retry…"
        default: "Download…"
        }
    }

    static func badge(_ state: ModelEntryState) -> MFBadge.Kind {
        switch state {
        case .ready, .loaded, .loading, .loadFailed: .custom("Installed", .success)
        case .downloading: .custom("Downloading", .accent)
        case .paused: .custom("Paused", .neutral)
        case .notInstalled: .custom("Not installed", .neutral)
        case .downloadFailed: .custom("Download failed", .preview)
        }
    }
}

/// The status chip for a model state.
struct ModelStatusBadge: View {
    let state: ModelEntryState

    var body: some View {
        MFBadge(Self.kind(for: state))
    }

    static func kind(for state: ModelEntryState) -> MFBadge.Kind {
        switch state {
        case .loaded: .loaded
        case .ready: .ready
        case .loading: .custom("Loading", .accent)
        case .downloading: .custom("Downloading", .accent)
        case .paused: .custom("Paused", .neutral)
        case .notInstalled: .custom("Not installed", .neutral)
        case .loadFailed: .custom("Didn't load", .preview)
        case .downloadFailed: .custom("Download failed", .preview)
        }
    }
}

/// An active download, with pause/resume and cancel.
struct DownloadBanner: View {
    let entry: ModelEntry
    let library: any ModelLibraryModel
    /// False in offline mode: Resume waits until downloads are allowed.
    var downloadsAllowed = true

    private var paused: Bool {
        if case .paused = entry.state { return true }
        return false
    }

    /// "ISTA ... · 12.1 GB · approved by you"; Quick says it installs on every Mac.
    var detail: String {
        [entry.formatDetail, entry.sizeBytes.map(ModelFormat.bytes),
         entry.installsWithoutApproval ? "installs on every Mac" : "approved by you"]
            .compactMap { $0 }.filter { !$0.isEmpty }.joined(separator: " · ")
    }

    var body: some View {
        let progress = entry.state.transfer
        HStack(spacing: MFSpacing.comfortable) {
            VStack(alignment: .leading, spacing: MFSpacing.xxs) {
                Text("\(paused ? "Paused" : "Downloading") \(entry.title)")
                    .font(MFFont.headline)
                    .foregroundStyle(MFColor.ink)
                Text(detail)
                    .font(MFFont.callout)
                    .foregroundStyle(MFColor.secondary)
                MFProgressBar(progress?.fraction, label: "Downloading \(entry.title)")
                    .padding(.top, MFSpacing.xs)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            if let progress {
                Text(ModelFormat.transfer(progress))
                    .font(MFFont.caption)
                    .foregroundStyle(MFColor.secondary)
            }
            Button(paused ? "Resume" : "Pause") {
                if paused { library.resume(entry.id) } else { library.pause(entry.id) }
            }
            .buttonStyle(.mfSecondary)
            .disabled(paused && !downloadsAllowed)
            .help(paused && !downloadsAllowed ? "Offline mode is on." : "")
            .accessibilityLabel("\(paused ? "Resume" : "Pause") downloading \(entry.title)")
            Button("Cancel") { library.cancelDownload(entry.id) }
                .buttonStyle(.mfQuiet)
                .accessibilityLabel("Cancel downloading \(entry.title)")
        }
        .padding(.horizontal, MFSpacing.l)
        .padding(.vertical, MFSpacing.comfortable)
        .background(RoundedRectangle(cornerRadius: MFRadius.l).fill(MFColor.accentFillFaint))
        .overlay(RoundedRectangle(cornerRadius: MFRadius.l).strokeBorder(MFColor.accentFill, lineWidth: MFLayout.hairline))
        .accessibilityElement(children: .contain)
    }
}

/// Details: the engine and files behind a model (the only place, with the
/// picker, where engine names show; L2). Reads the entry live by id.
struct ModelDetailsSheet: View {
    let library: any ModelLibraryModel
    let id: String
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(alignment: .leading, spacing: MFSpacing.l) {
            if let entry = library.entries.first(where: { $0.id == id }) {
                Text(entry.title)
                    .font(MFFont.headline)
                    .foregroundStyle(MFColor.ink)
                Text(entry.summary)
                    .font(MFFont.callout)
                    .foregroundStyle(MFColor.secondary)
                MFCard {
                    MFListRow("Engine") { value(entry.engineName) }
                    MFListRow("Format") { value(entry.formatDetail) }
                    if let licence = entry.licence {
                        MFListRow("Licence") { value(licence) }
                    }
                    MFListRow("Size") { value(entry.sizeBytes.map(ModelFormat.bytes) ?? "Unknown") }
                    MFListRow("Status", subtitle: Self.failure(entry.state)) { ModelStatusBadge(state: entry.state) }
                    if let location = entry.location {
                        MFListRow("Location", subtitle: location)
                    }
                }
            } else {
                Text("This model is no longer in the library.")
                    .font(MFFont.callout)
                    .foregroundStyle(MFColor.secondary)
            }
            HStack {
                Spacer()
                Button("Done") { dismiss() }
                    .buttonStyle(.mfPrimary)
                    .keyboardShortcut(.defaultAction)
            }
        }
        .padding(MFSpacing.xl)
        .frame(width: LibraryMetrics.detailsWidth)
    }

    static func failure(_ state: ModelEntryState) -> String? {
        switch state {
        case .loadFailed(let message), .downloadFailed(let message): message
        default: nil
        }
    }

    private func value(_ text: String) -> some View {
        Text(text)
            .font(MFFont.callout)
            .foregroundStyle(MFColor.ink)
            .textSelection(.enabled)
    }
}

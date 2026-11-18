import MferenceDesign
import SwiftUI

/// Settings > Models: the model library (`08b-model-library.png`). Installed
/// and downloading models, the active download, and model behaviour (default
/// model, MTP, idle unload). A download starts only after the person approves
/// it here (L4).
public struct ModelLibraryView: View {
    let library: any ModelLibraryModel
    @State private var details: ModelEntry?
    @State private var pendingDownload: ModelEntry?

    public init(library: any ModelLibraryModel) {
        self.library = library
    }

    public var body: some View {
        MFScroll {
            VStack(alignment: .leading, spacing: LibraryMetrics.sectionSpacing) {
                header
                ForEach(library.entries.filter { $0.state.transfer != nil }) { entry in
                    DownloadBanner(entry: entry, library: library)
                }
                modelTable
                behaviour
            }
            .padding(.horizontal, LibraryMetrics.pagePadding.width)
            .padding(.vertical, LibraryMetrics.pagePadding.height)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .sheet(item: $details) { entry in
            ModelDetailsSheet(entry: entry)
        }
        .confirmationDialog(
            pendingDownload.map { "Download \($0.title)?" } ?? "",
            isPresented: Binding(get: { pendingDownload != nil }, set: { if !$0 { pendingDownload = nil } }),
            presenting: pendingDownload
        ) { entry in
            Button("Download") { library.download(entry.id) }
            Button("Not now", role: .cancel) {}
        } message: { entry in
            Text(Self.approvalMessage(for: entry))
        }
    }

    static func approvalMessage(for entry: ModelEntry) -> String {
        let size = entry.sizeBytes.map { " (\(ModelFormat.bytes($0)))" } ?? ""
        return "\(entry.productName) downloads\(size) to this Mac and runs here. Nothing else is sent."
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
            if !downloadable.isEmpty {
                Menu {
                    ForEach(downloadable) { entry in
                        Button("\(entry.title)\(entry.sizeBytes.map { " · " + ModelFormat.bytes($0) } ?? "")") {
                            pendingDownload = entry
                        }
                    }
                } label: {
                    Text("Browse models")
                }
                .menuStyle(.button)
                .buttonStyle(.mfPrimary)
                .fixedSize()
            }
        }
    }

    /// Models that are not installed and may be offered on this Mac.
    private var downloadable: [ModelEntry] {
        library.entries.filter { entry in
            guard case .notInstalled = entry.state else { return false }
            if case .tooLarge = entry.fit { return false }
            return true
        }
    }

    private var modelTable: some View {
        MFCard {
            HStack(spacing: MFSpacing.m) {
                MFSectionHeader("Model").frame(maxWidth: .infinity, alignment: .leading)
                MFSectionHeader("On disk").frame(width: LibraryMetrics.sizeColumn, alignment: .leading)
                MFSectionHeader("Status").frame(width: LibraryMetrics.statusColumn, alignment: .leading)
                Color.clear.frame(width: LibraryMetrics.actionColumn, height: 1)
            }
            .padding(.horizontal, MFSpacing.l)
            .padding(.vertical, MFSpacing.s + 1)
            .background(MFColor.canvas)
            ForEach(library.entries.filter { $0.state != .notInstalled || $0.state.transfer != nil }) { entry in
                ModelTableRow(entry: entry,
                              onDetails: { details = entry },
                              onDownload: { pendingDownload = entry })
            }
        }
    }

    private var behaviour: some View {
        MFCard {
            MFListRow("Default model", subtitle: "Used for new chats, mail drafts and scanning.") {
                Menu {
                    ForEach(library.entries.filter(\.state.isInstalled)) { entry in
                        Button(entry.title) { library.defaultModelID = entry.id }
                    }
                } label: {
                    Text(library.entries.first { $0.id == library.defaultModelID }?.title ?? "Choose")
                }
                .menuStyle(.button)
                .buttonStyle(.mfSecondary)
                .fixedSize()
                .accessibilityLabel("Default model")
            }
            MFListRow("Faster answers from Deep (MTP)",
                      subtitle: "Drafts several tokens per step and checks them. Same quality, less waiting.") {
                MFBadge(.custom("On", .success))
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
            }
        }
    }

    static func idleText(_ minutes: Int) -> String {
        minutes == 1 ? "1 minute" : minutes % 60 == 0 && minutes >= 60
            ? (minutes == 60 ? "1 hour" : "\(minutes / 60) hours") : "\(minutes) minutes"
    }

    static func idleShort(_ minutes: Int) -> String {
        minutes >= 60 && minutes % 60 == 0 ? "\(minutes / 60) h" : "\(minutes) min"
    }
}

enum LibraryMetrics {
    static let pagePadding = CGSize(width: 30, height: 22)
    static let sectionSpacing: CGFloat = 20
    static let sizeColumn: CGFloat = 170
    static let statusColumn: CGFloat = 200
    static let actionColumn: CGFloat = 96
    static let detailsWidth: CGFloat = 440
    static let idleChoices = [5, 10, 30, 60]
}

/// One installed or downloading model.
struct ModelTableRow: View {
    let entry: ModelEntry
    let onDetails: () -> Void
    let onDownload: () -> Void

    var body: some View {
        HStack(spacing: MFSpacing.m) {
            VStack(alignment: .leading, spacing: MFSpacing.xxs) {
                Text(entry.title)
                    .font(MFFont.calloutEmphasis)
                    .fontWeight(.semibold)
                    .foregroundStyle(MFColor.ink)
                Text(entry.summary)
                    .font(MFFont.caption)
                    .foregroundStyle(MFColor.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            Text(entry.sizeBytes.map(ModelFormat.bytes) ?? "—")
                .font(MFFont.callout)
                .foregroundStyle(MFColor.ink)
                .frame(width: LibraryMetrics.sizeColumn, alignment: .leading)
            HStack {
                ModelStatusBadge(state: entry.state)
            }
            .frame(width: LibraryMetrics.statusColumn, alignment: .leading)
            Group {
                if case .failed = entry.state, !entry.state.isInstalled {
                    Button("Retry", action: onDownload).buttonStyle(.mfSecondary)
                } else {
                    Button("Details", action: onDetails).buttonStyle(.mfSecondary)
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
        case .failed: .custom("Needs attention", .preview)
        }
    }
}

/// The active download, with pause/resume and cancel.
struct DownloadBanner: View {
    let entry: ModelEntry
    let library: any ModelLibraryModel

    private var paused: Bool {
        if case .paused = entry.state { return true }
        return false
    }

    var body: some View {
        let progress = entry.state.transfer ?? TransferProgress(completedBytes: 0, totalBytes: nil)
        HStack(spacing: MFSpacing.m + 2) {
            VStack(alignment: .leading, spacing: MFSpacing.xxs) {
                Text("\(paused ? "Paused" : "Downloading") \(entry.title)")
                    .font(MFFont.bodyEmphasis)
                    .fontWeight(.semibold)
                    .foregroundStyle(MFColor.ink)
                Text([entry.formatDetail, entry.sizeBytes.map(ModelFormat.bytes), "approved by you"]
                    .compactMap { $0 }.filter { !$0.isEmpty }.joined(separator: " · "))
                    .font(MFFont.callout)
                    .foregroundStyle(MFColor.secondary)
                MFProgressBar(progress.fraction ?? 0)
                    .padding(.top, MFSpacing.xs)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            Text(ModelFormat.transfer(progress))
                .font(MFFont.caption)
                .foregroundStyle(MFColor.secondary)
            Button(paused ? "Resume" : "Pause") {
                if paused { library.resume(entry.id) } else { library.pause(entry.id) }
            }
            .buttonStyle(.mfSecondary)
            Button("Cancel") { library.cancelDownload(entry.id) }
                .buttonStyle(.mfQuiet)
        }
        .padding(.horizontal, MFSpacing.l)
        .padding(.vertical, MFSpacing.m + 2)
        .background(RoundedRectangle(cornerRadius: MFRadius.l).fill(MFColor.accentFillFaint))
        .overlay(RoundedRectangle(cornerRadius: MFRadius.l).strokeBorder(MFColor.accentFill))
        .accessibilityElement(children: .contain)
    }
}

/// Details: the engine and files behind a model (the only place, with the
/// picker, where engine names show; L2).
struct ModelDetailsSheet: View {
    let entry: ModelEntry
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(alignment: .leading, spacing: MFSpacing.l) {
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
                MFListRow("Status") { ModelStatusBadge(state: entry.state) }
                if let location = entry.location {
                    MFListRow("Location", subtitle: location)
                }
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

    private func value(_ text: String) -> some View {
        Text(text)
            .font(MFFont.callout)
            .foregroundStyle(MFColor.ink)
            .textSelection(.enabled)
    }
}

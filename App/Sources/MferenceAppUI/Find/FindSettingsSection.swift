import AppKit
import MferenceAppCore
import MferenceDesign
import SwiftUI

/// Settings › Find (canvas board 5): status, folders, skipped things, what
/// Find reads, behaviour and storage.
struct FindSettingsSection: View {
    let find: FindService?
    @State private var confirmDelete = false
    @State private var skipped: [FindSkippedFile] = []
    @State private var showsSkipped = false

    var body: some View {
        VStack(alignment: .leading, spacing: MFSpacing.m) {
            VStack(alignment: .leading, spacing: MFSpacing.xs) {
                Text("Find").font(MFFont.sectionTitle).foregroundStyle(MFColor.ink).accessibilityAddTraits(.isHeader)
                Text("Search your files by what is in them. The index and the search model stay on this Mac.")
                    .font(MFFont.callout).foregroundStyle(MFColor.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            if let find {
                content(find)
            } else {
                MFCard { MFListRow("Find", subtitle: "Not available in this build yet.") }
            }
        }
    }

    @ViewBuilder
    private func content(_ find: FindService) -> some View {
        MFCard {
            MFListRow("Find", subtitle: find.settings.enabled ? statusLine(find) : "Off. Turn it on from the Find tab.") {
                if find.settings.enabled {
                    Toggle("Find", isOn: Binding(get: { true }, set: { if !$0 { Task { await find.turnOff() } } }))
                        .toggleStyle(.mf)
                } else {
                    Button("Turn on") {
                        Task { await find.turnOn(roots: find.settings.roots) }
                    }
                    .buttonStyle(.mfSecondary)
                }
            }
            if case .failed(let message) = find.modelState {
                MFListRow("Search model", subtitle: message) {
                    Button("Retry") { find.retryModelDownload() }.buttonStyle(.mfSecondary)
                }
            }
            if let error = find.lastError {
                MFListRow("Index", subtitle: error)
            }
        }
        MFCard {
            ForEach(find.settings.roots, id: \.self) { root in
                MFListRow(root.lastPathComponent, subtitle: FindResultRow.folder(of: root.appending(path: "x"))) {
                    Button("Remove") { Task { await find.removeFolder(root) } }
                        .buttonStyle(.mfSecondary)
                        .accessibilityLabel("Remove \(root.lastPathComponent) from Find")
                }
            }
            MFListRow("Add a folder", subtitle: "Find reads it the next time it looks for new files.") {
                Button("Add a folder…") { addFolder(find) }.buttonStyle(.mfSecondary)
            }
            MFListRow("Skipped", subtitle: FindCrawlRules.skippedSummary.joined(separator: " · "))
            if find.status.skippedFiles > 0 {
                MFListRow("Files Find could not read",
                          subtitle: "\(find.status.skippedFiles.formatted()) files are searched by name only.") {
                    Button("See the list") {
                        Task {
                            skipped = await find.skippedFiles()
                            showsSkipped = true
                        }
                    }
                    .buttonStyle(.mfSecondary)
                }
            }
        }
        MFCard {
            MFListRow("Documents, PDFs, slides and sheets", subtitle: "Text, by page and slide.") {
                MFBadge(.custom("Always", .neutral))
            }
            MFListRow("Text in screenshots and scans", subtitle: "Read with Vision on this Mac.") {
                Toggle("Text in screenshots and scans", isOn: Binding(
                    get: { find.settings.readImages }, set: { value in Task { await find.setReadImages(value) } }))
                    .toggleStyle(.mf)
            }
            MFListRow("Photos, video and audio by what is in them", subtitle: mediaLine(find)) {
                if case .failed = find.mediaModelState, find.settings.readMedia {
                    Button("Retry") { find.retryMediaModelDownload() }.buttonStyle(.mfSecondary)
                }
                Toggle("Photos, video and audio by what is in them", isOn: Binding(
                    get: { find.settings.readMedia }, set: { value in Task { await find.setReadMedia(value) } }))
                    .toggleStyle(.mf)
            }
            MFListRow("Photos library", subtitle: photosLine(find)) {
                if find.photosAccess == .needsSystemSettings || find.photosAccess == .denied {
                    Button("Open System Settings") { find.openPhotosSettings() }.buttonStyle(.mfSecondary)
                }
                Toggle("Photos library", isOn: Binding(
                    get: { find.settings.photosLibrary }, set: { value in Task { await find.setPhotosLibrary(value) } }))
                    .toggleStyle(.mf)
            }
            MFListRow("Code in git projects", subtitle: "Tracked files only. Off by default.") {
                Toggle("Code in git projects", isOn: Binding(
                    get: { find.settings.includeCode }, set: { value in Task { await find.setIncludeCode(value) } }))
                    .toggleStyle(.mf)
            }
        }
        MFCard {
            MFListRow("Learn meaning only on power", subtitle: "Words are always kept up to date. Learning also pauses while Chat is answering.") {
                Toggle("Learn meaning only on power", isOn: Binding(
                    get: { find.settings.learnOnlyOnPower },
                    set: { value in Task { await find.setLearnOnlyOnPower(value) } }))
                    .toggleStyle(.mf)
            }
            MFListRow("Quick find", subtitle: "Opens Find over any app.") {
                MFKeycap("⌥ Space")
            }
            MFListRow("On disk", subtitle: storageLine(find)) {
                Button("Rebuild") { Task { await find.rebuild() } }
                    .buttonStyle(.mfSecondary)
                    .disabled(!find.settings.enabled)
                Button("Delete index", role: .destructive) { confirmDelete = true }
                    .buttonStyle(.mfSecondary)
            }
        }
        .confirmationDialog("Delete Find's index?", isPresented: $confirmDelete) {
            Button("Delete index", role: .destructive) { Task { await find.deleteIndex() } }
            Button("Keep", role: .cancel) {}
        } message: {
            Text("Find turns off and forgets what it read. Your files are not touched. The search model stays downloaded.")
        }
        .sheet(isPresented: $showsSkipped) {
            FindSkippedList(files: skipped) { showsSkipped = false }
        }
        .task { await find.refreshPhotosAccess() }
        .onReceive(NotificationCenter.default.publisher(for: NSApplication.didBecomeActiveNotification)) { _ in
            Task { await find.refreshPhotosAccess() }
        }
    }

    private func statusLine(_ find: FindService) -> String {
        find.status.line ?? "Up to date · \(find.status.totalFiles.formatted()) files"
    }

    private func storageLine(_ find: FindService) -> String {
        let index = ByteCountFormatter.string(fromByteCount: Int64(find.status.indexBytes), countStyle: .file)
        let model = find.modelState.isReady ? "search model \(FindModel.sizeLabel)" : "search model not downloaded"
        let media = find.mediaModelState.isReady ? " · photo, video and audio model \(FindModelFile.media.sizeLabel)" : ""
        return "Index \(index) · \(model)\(media)"
    }

    private func photosLine(_ find: FindService) -> String {
        if find.settings.photosLibrary {
            let count = find.photosCount.map { "\($0.formatted()) photos and videos" } ?? "Your photos and videos"
            let hint = find.settings.readMedia ? "" : " Turn on Photos, video and audio to find them by what they show."
            return "\(count), read on this Mac. Nothing in Photos changes.\(hint)"
        }
        switch find.photosAccess {
        case .needsSystemSettings, .denied:
            return "Mference is not allowed to read Photos. Allow it in System Settings › Privacy & Security › Photos, then turn this on."
        case .granted, .notDetermined:
            return "Pictures and videos kept in the Photos app. Asks for permission first."
        }
    }

    private func mediaLine(_ find: FindService) -> String {
        let about = "“dog on a beach”, “the clip where we sing”. Seen and heard on this Mac."
        guard find.settings.readMedia else {
            return about + " Needs a further \(FindModelFile.media.sizeLabel) model."
        }
        switch find.mediaModelState {
        case .downloading(let fraction): return "Downloading the model: \(Int(fraction * 100))% of \(FindModelFile.media.sizeLabel)."
        case .verifying: return "Checking the model…"
        case .failed(let message): return "The model could not be downloaded: \(message)"
        case .waitingForOfflineMode: return "Downloads when Offline mode is off."
        case .ready, .notDownloaded: return about
        }
    }

    private func addFolder(_ find: FindService) {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.prompt = "Add"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        Task { await find.addFolder(url) }
    }
}

/// "See the list": files Find could not read, and why.
struct FindSkippedList: View {
    let files: [FindSkippedFile]
    let done: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: MFSpacing.m) {
            Text("Files Find could not read").font(MFFont.headline).foregroundStyle(MFColor.ink)
            List(files) { file in
                VStack(alignment: .leading, spacing: MFSpacing.xxs) {
                    Text(file.url.lastPathComponent).font(MFFont.bodySmall).foregroundStyle(MFColor.ink)
                    Text(file.reason).font(MFFont.caption).foregroundStyle(MFColor.secondary)
                }
            }
            HStack {
                Spacer()
                Button("Done", action: done).buttonStyle(.mfPrimary).keyboardShortcut(.defaultAction)
            }
        }
        .padding(MFSpacing.l)
        .frame(width: FindMetrics.setupSideWidth + FindMetrics.filterWidth, height: FindMetrics.setupMaxWidth)
    }
}

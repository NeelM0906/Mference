import MferenceAppCore
import MferenceDesign
import SwiftUI

/// Settings (`08-settings.png`): Models, Memory, Privacy, Connections on one
/// scrolling page with a section sidebar; the model library (`08b`) opens
/// from Models. Phase C and D rows (dictation, errands, payments, other mail
/// providers, iPhone) are shown as "Later".
public struct SettingsView: View {
    public enum Pane: String, CaseIterable, Identifiable, Sendable {
        case models, memory, privacy, connections, dictation, shortcuts, iPhone

        public var id: String { rawValue }

        public var title: String {
            switch self {
            case .models: "Models"
            case .memory: "Memory"
            case .privacy: "Privacy"
            case .connections: "Connections"
            case .dictation: "Dictation"
            case .shortcuts: "Shortcuts"
            case .iPhone: "iPhone"
            }
        }
    }

    let library: (any ModelLibraryModel)?
    let memory: (any MemoryListModel)?
    let settings: (any AppSettingsModel)?

    @State var pane: Pane
    @State var showsLibrary: Bool
    @State private var details: ModelEntry?
    @State private var pendingDownload: ModelEntry?
    @State private var editing: MemoryEditTarget?
    @State private var showsAllMemories = false

    /// The live models come from the app environment (U2, U4); a nil input
    /// shows that section's honest empty state.
    public init(library: (any ModelLibraryModel)? = nil, memory: (any MemoryListModel)? = nil,
                settings: (any AppSettingsModel)? = nil, pane: Pane = .models, showsLibrary: Bool = false) {
        self.library = library
        self.memory = memory
        self.settings = settings
        _pane = State(initialValue: pane)
        _showsLibrary = State(initialValue: showsLibrary)
    }

    public var body: some View {
        HStack(spacing: 0) {
            sidebar
            Rectangle().fill(MFColor.hairline).frame(width: 1)
            if showsLibrary, let library {
                VStack(alignment: .leading, spacing: 0) {
                    Button {
                        showsLibrary = false
                    } label: {
                        Label("Settings", systemImage: "chevron.left")
                    }
                    .buttonStyle(MFPillButtonStyle())
                    .padding(.horizontal, SettingsMetrics.pagePadding.width - MFSpacing.m)
                    .padding(.top, MFSpacing.m)
                    ModelLibraryView(library: library)
                }
            } else {
                page
            }
        }
        .sheet(item: $details) { ModelDetailsSheet(entry: $0) }
        .sheet(item: $editing) { target in
            MemoryEditor(target: target) { text in
                guard let memory else { return }
                switch target {
                case .new: memory.add(text)
                case .existing(let entry): memory.update(entry.id, text: text)
                }
            }
        }
        .confirmationDialog(
            pendingDownload.map { "Download \($0.title)?" } ?? "",
            isPresented: Binding(get: { pendingDownload != nil }, set: { if !$0 { pendingDownload = nil } }),
            presenting: pendingDownload
        ) { entry in
            Button("Download") { library?.download(entry.id) }
            Button("Not now", role: .cancel) {}
        } message: { entry in
            Text(ModelLibraryView.approvalMessage(for: entry))
        }
    }

    // MARK: Sidebar

    private var sidebar: some View {
        VStack(alignment: .leading, spacing: MFSpacing.xxs) {
            ForEach(Pane.allCases) { item in
                Button {
                    pane = item
                    showsLibrary = false
                } label: {
                    Text(item.title)
                        .font(item == pane ? MFFont.callout.weight(.medium) : MFFont.callout)
                        .foregroundStyle(MFColor.ink)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(.horizontal, MFSpacing.m - 2)
                        .padding(.vertical, MFSpacing.s - 1)
                        .background(RoundedRectangle(cornerRadius: MFRadius.s).fill(item == pane ? MFColor.fill : .clear))
                        .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .accessibilityAddTraits(item == pane ? .isSelected : [])
            }
            Spacer()
        }
        .padding(.horizontal, MFSpacing.m - 2)
        .padding(.vertical, MFSpacing.m + 2)
        .frame(width: SettingsMetrics.sidebarWidth)
        .frame(maxHeight: .infinity)
        .background(MFColor.canvas.opacity(0.6))
        .accessibilityElement(children: .contain)
        .accessibilityLabel("Settings sections")
    }

    // MARK: Page

    private var page: some View {
        ScrollViewReader { proxy in
            MFScroll {
                VStack(alignment: .leading, spacing: SettingsMetrics.sectionSpacing) {
                    modelsSection.id(Pane.models)
                    memorySection.id(Pane.memory)
                    privacySection.id(Pane.privacy)
                    connectionsSection.id(Pane.connections)
                    dictationSection.id(Pane.dictation)
                    shortcutsSection.id(Pane.shortcuts)
                    iPhoneSection.id(Pane.iPhone)
                }
                .padding(.horizontal, SettingsMetrics.pagePadding.width)
                .padding(.vertical, SettingsMetrics.pagePadding.height)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .onChange(of: pane, initial: true) { _, pane in
                withAnimation { proxy.scrollTo(pane, anchor: .top) }
            }
        }
    }

    private func section<Content: View>(_ title: String, _ subtitle: String,
                                        @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: MFSpacing.m) {
            VStack(alignment: .leading, spacing: MFSpacing.xs) {
                Text(title)
                    .font(MFFont.headline)
                    .foregroundStyle(MFColor.ink)
                    .accessibilityAddTraits(.isHeader)
                Text(subtitle)
                    .font(MFFont.callout)
                    .foregroundStyle(MFColor.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            content()
        }
    }

    // MARK: Models

    static let quickID = OnboardingView.quickID
    static let deepID = OnboardingView.deepID

    private var modelsSection: some View {
        section("Models", modelsSubtitle) {
            MFCard {
                if let library {
                    ForEach(library.entries.filter { $0.id == Self.quickID || $0.id == Self.deepID }) { entry in
                        modelRow(entry, isDefault: entry.id == library.defaultModelID)
                    }
                    MFListRow("All models", subtitle: "\(library.entries.filter(\.state.isInstalled).count) on this Mac. Default model, unload when idle (\(ModelLibraryView.idleText(library.idleUnloadMinutes))), details.") {
                        Button("Open model library") { showsLibrary = true }
                            .buttonStyle(.mfSecondary)
                    }
                } else {
                    MFListRow("Models", subtitle: "Looking for models on this Mac…")
                }
                MFListRow("Choose the model for me",
                          subtitle: "Coming later. For now you choose the model for each chat yourself.") {
                    MFBadge(.later)
                    Toggle("Choose the model for me", isOn: .constant(false))
                        .toggleStyle(.mf)
                        .disabled(true)
                }
                MFListRow("Speech · Cactus Whistle",
                          subtitle: "Dictation and meeting notes, transcribed on this Mac. Arrives with Notes in a later version.") {
                    MFBadge(.later)
                }
                MFListRow("Text recognition · Apple Vision",
                          subtitle: "Built into macOS. Reads receipts, documents and tables for Scan. Nothing to download.") {
                    MFBadge(.custom("Built in", .neutral))
                }
                MFListRow("Apple Intelligence model",
                          subtitle: "Apple's on-device model sorts mail, writes titles and short summaries, and fills scan fields when Apple Intelligence is on. Quick does these jobs when it is off.") {
                    MFBadge(.custom("For small jobs", .success))
                    settingToggle("Apple Intelligence model", .useAppleModel)
                }
            }
        }
    }

    private var modelsSubtitle: String {
        guard let library else { return "Everything runs on this computer." }
        return ModelFormat.machineLine(library.machine) + " Everything runs on this computer."
    }

    private func modelRow(_ entry: ModelEntry, isDefault: Bool) -> some View {
        MFListRow(entry.title, subtitle: modelRowSubtitle(entry)) {
            switch entry.state {
            case .downloading(let progress):
                MFProgressBadge("Downloading", fraction: progress.fraction)
                Button("Pause") { library?.pause(entry.id) }.buttonStyle(.mfSecondary)
            case .paused(let progress):
                MFProgressBadge("Paused", fraction: progress.fraction)
                Button("Resume") { library?.resume(entry.id) }.buttonStyle(.mfSecondary)
            case .notInstalled, .failed:
                if case .tooLarge(let reason) = entry.fit {
                    MFBadge(.custom(reason, .neutral))
                } else {
                    ModelStatusBadge(state: entry.state)
                    Button("Download…") { pendingDownload = entry }.buttonStyle(.mfSecondary)
                }
            case .loaded, .ready, .loading:
                MFBadge(.custom(isDefault ? "Installed · default" : "Installed", .success))
                Button("Details") { details = entry }.buttonStyle(.mfSecondary)
            }
        }
    }

    private func modelRowSubtitle(_ entry: ModelEntry) -> String {
        var parts = [entry.summary.hasSuffix(".") ? String(entry.summary.dropLast()) : entry.summary]
        if let size = entry.sizeBytes {
            parts.append("\(ModelFormat.bytes(size)) \(entry.state.isInstalled ? "on disk" : "download")")
        }
        if let licence = entry.licence { parts.append(licence) }
        return parts.joined(separator: " · ")
    }

    // MARK: Memory

    private var memorySection: some View {
        section("Memory", "What Mference remembers about you, kept on this Mac. Edit or delete anything.") {
            MFCard {
                if let memory {
                    let all = memory.memories
                    let shown = showsAllMemories ? all : Array(all.prefix(SettingsMetrics.memoryPreview))
                    if all.isEmpty {
                        MFListRow("Nothing remembered yet",
                                  subtitle: "When Mference learns something useful about you, it shows here with where it learned it.")
                    }
                    ForEach(shown) { entry in
                        MFListRow(entry.text, subtitle: entry.source) {
                            Button("Edit") { editing = .existing(entry) }.buttonStyle(.mfSecondary)
                            Button("Delete") { memory.delete(entry.id) }
                                .buttonStyle(.mfSecondary)
                                .accessibilityLabel("Delete “\(entry.text)”")
                        }
                    }
                    HStack(spacing: MFSpacing.m) {
                        Text(all.count > shown.count ? "\(all.count - shown.count) more" : "\(all.count) remembered")
                            .font(MFFont.caption)
                            .foregroundStyle(MFColor.secondary)
                            .frame(maxWidth: .infinity, alignment: .leading)
                        if all.count > SettingsMetrics.memoryPreview {
                            Button(showsAllMemories ? "Show fewer" : "Show all") { showsAllMemories.toggle() }
                                .buttonStyle(.mfSecondary)
                        }
                        Button("Add") { editing = .new }.buttonStyle(.mfSecondary)
                    }
                    .padding(.horizontal, MFSpacing.l)
                    .padding(.vertical, MFSpacing.m)
                } else {
                    MFListRow("Memory is loading", subtitle: "Your remembered items appear here.")
                }
            }
        }
    }

    // MARK: Privacy

    private var privacySection: some View {
        section("Privacy", "No account and no usage reporting. There is nothing to opt out of.") {
            MFCard {
                MFListRow("Web search through SearXNG",
                          subtitle: "Off until you turn it on. Only your search queries leave this Mac. Runs as a bundled SearXNG on this Mac.") {
                    settingToggle("Web search through SearXNG", .webSearchEnabled)
                }
                MFListRow("Offline mode",
                          subtitle: "Turns off web search and mail sync. Chat and scanning keep working.") {
                    settingToggle("Offline mode", .offlineMode)
                }
                MFListRow("Block remote images and trackers in mail",
                          subtitle: "Removed before a message is shown.") {
                    settingToggle("Block remote images and trackers in mail", .blockRemoteMailContent)
                }
                MFListRow("Ask before any purchase, booking, send or delete",
                          subtitle: "Always on. Errands arrive in a later version and will stop at an approval card.") {
                    MFBadge(.custom("Always", .neutral))
                }
                MFListRow("Payments",
                          subtitle: "You pay at checkout yourself. Arrives with Errands in a later version.") {
                    MFBadge(.later)
                }
            }
        }
    }

    // MARK: Connections

    private var connectionsSection: some View {
        section("Connections", "Mail and calendar come from the Apple apps on this Mac. Nothing is synced to a server.") {
            MFCard {
                connectionRow(.appleMail, title: "Apple Mail · every account in Mail on this Mac",
                              subtitle: "Read from Mail's local store (asks for Full Disk Access the first time); replies send through Mail when you click Send.")
                connectionRow(.appleCalendar, title: "Apple Calendar",
                              subtitle: "Shows your events and adds the ones you approve (asks for Calendar access the first time).")
                MFListRow("Sync to-dos to Reminders",
                          subtitle: "Off until you turn it on. Asks for Reminders access the first time.") {
                    settingToggle("Sync to-dos to Reminders", .remindersSyncEnabled)
                }
                MFListRow("Microsoft 365 and Google",
                          subtitle: "Mail and calendar from these providers arrive in a later version.") {
                    MFBadge(.later)
                }
            }
        }
    }

    private func connectionRow(_ kind: ConnectionKind, title: String, subtitle: String) -> some View {
        MFListRow(title, subtitle: subtitle) {
            switch settings?.connectionState(kind) ?? .later {
            case .connected:
                MFBadge(.custom("Connected", .success))
            case .notConnected:
                Button("Connect") { settings?.connect(kind) }.buttonStyle(.mfPrimary)
            case .needsSystemSettings:
                Button("Open System Settings") { settings?.connect(kind) }.buttonStyle(.mfSecondary)
            case .later:
                MFBadge(.later)
            }
        }
    }

    // MARK: Later sections

    private var dictationSection: some View {
        section("Dictation", "Dictate into any app with Cactus Whistle, on this Mac.") {
            MFCard {
                MFListRow("Dictation and meeting notes", subtitle: "Arrives in a later version. Nothing is recorded today.") {
                    MFBadge(.later)
                }
            }
        }
    }

    private var shortcutsSection: some View {
        section("Shortcuts", "Keyboard shortcuts in the Mference window.") {
            MFCard {
                ForEach(AppSection.pills + [.settings]) { section in
                    MFListRow(section == .settings ? "Open Settings" : "Go to \(section.title)") {
                        MFKeycap("⌘\(String(ShellModel.shortcut(for: section).character))")
                    }
                }
                MFListRow("Search") { MFKeycap("⌘K") }
            }
        }
    }

    private var iPhoneSection: some View {
        section("iPhone", "Mference on iPhone runs Quick itself and hands off to this Mac on the same Wi-Fi.") {
            MFCard {
                MFListRow("Pair an iPhone", subtitle: "Arrives with the iPhone app in a later version.") {
                    MFBadge(.later)
                }
            }
        }
    }

    // MARK: Helpers

    private func settingToggle(_ label: String, _ setting: SettingSwitch) -> some View {
        Toggle(label, isOn: Binding(
            get: { settings.map(setting.get) ?? false },
            set: { value in if let settings { setting.set(settings, value) } }))
            .toggleStyle(.mf)
            .disabled(settings == nil)
    }
}

/// The on/off settings, so one toggle helper can bind any of them.
enum SettingSwitch {
    case webSearchEnabled, offlineMode, blockRemoteMailContent, remindersSyncEnabled, useAppleModel

    func get(_ settings: any AppSettingsModel) -> Bool {
        switch self {
        case .webSearchEnabled: settings.webSearchEnabled
        case .offlineMode: settings.offlineMode
        case .blockRemoteMailContent: settings.blockRemoteMailContent
        case .remindersSyncEnabled: settings.remindersSyncEnabled
        case .useAppleModel: settings.useAppleModel
        }
    }

    func set(_ settings: any AppSettingsModel, _ value: Bool) {
        switch self {
        case .webSearchEnabled: settings.webSearchEnabled = value
        case .offlineMode: settings.offlineMode = value
        case .blockRemoteMailContent: settings.blockRemoteMailContent = value
        case .remindersSyncEnabled: settings.remindersSyncEnabled = value
        case .useAppleModel: settings.useAppleModel = value
        }
    }
}

enum SettingsMetrics {
    static let sidebarWidth: CGFloat = 220
    static let pagePadding = CGSize(width: 30, height: 22)
    static let sectionSpacing: CGFloat = 26
    static let memoryPreview = 3
    static let editorSize = CGSize(width: 440, height: 120)
}

enum MemoryEditTarget: Identifiable, Hashable {
    case new
    case existing(MemoryEntry)

    var id: String {
        switch self {
        case .new: "new"
        case .existing(let entry): entry.id.uuidString
        }
    }
}

/// Add or edit one remembered thing.
struct MemoryEditor: View {
    let target: MemoryEditTarget
    let onSave: (String) -> Void
    @State private var text: String
    @Environment(\.dismiss) private var dismiss

    init(target: MemoryEditTarget, onSave: @escaping (String) -> Void) {
        self.target = target
        self.onSave = onSave
        if case .existing(let entry) = target {
            _text = State(initialValue: entry.text)
        } else {
            _text = State(initialValue: "")
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: MFSpacing.m) {
            Text(target == .new ? "Remember something" : "Edit memory")
                .font(MFFont.headline)
                .foregroundStyle(MFColor.ink)
            if case .existing(let entry) = target {
                Text(entry.source)
                    .font(MFFont.caption)
                    .foregroundStyle(MFColor.secondary)
            }
            TextEditor(text: $text)
                .font(MFFont.body)
                .frame(width: SettingsMetrics.editorSize.width, height: SettingsMetrics.editorSize.height)
                .padding(MFSpacing.xs)
                .background(RoundedRectangle(cornerRadius: MFRadius.m).fill(MFColor.surface))
                .overlay(RoundedRectangle(cornerRadius: MFRadius.m).strokeBorder(MFColor.hairlineStrong))
                .accessibilityLabel("Memory text")
            HStack {
                Spacer()
                Button("Cancel") { dismiss() }
                    .buttonStyle(.mfSecondary)
                    .keyboardShortcut(.cancelAction)
                Button("Save") {
                    onSave(text.trimmingCharacters(in: .whitespacesAndNewlines))
                    dismiss()
                }
                .buttonStyle(.mfPrimary)
                .keyboardShortcut(.defaultAction)
                .disabled(text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
        }
        .padding(MFSpacing.xl)
    }
}

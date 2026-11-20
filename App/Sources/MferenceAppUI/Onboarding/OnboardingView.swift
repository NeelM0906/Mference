import MferenceAppCore
import MferenceDesign
import SwiftUI

/// First run (`01-first-run.png`): no account, no permission asked up front
/// (L5, L6). Quick installs on every Mac when the Models step shows, if the
/// disk has room; Deep is offered only on Macs with more than 16 GB and
/// downloads only after the person says yes (L4). Every status line is read
/// from the model's current state.
public struct OnboardingView: View {
    public enum Step: Int, CaseIterable, Identifiable, Sendable {
        case welcome, models, permissions, mail

        public var id: Int { rawValue }

        var title: String {
            switch self {
            case .welcome: "Welcome"
            case .models: "Models"
            case .permissions: "Permissions, as you need them"
            case .mail: "Connect mail (optional)"
            }
        }
    }

    let library: any ModelLibraryModel
    let onFinish: () -> Void
    @State var step: Step
    /// The person said "Not now" to Deep.
    @State var declinedDeep = false
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    public init(library: any ModelLibraryModel, step: Step = .welcome, onFinish: @escaping () -> Void) {
        self.library = library
        self.onFinish = onFinish
        _step = State(initialValue: step)
    }

    var quick: ModelEntry? { library.entries.first { $0.id == AppModel.quick.modelID } }
    var deep: ModelEntry? { library.entries.first { $0.id == AppModel.deep.modelID } }

    /// Deep is offered only where L4 allows it (`ModelEntry.isOffered`).
    var offersDeep: Bool { deep?.isOffered(on: library.machine) ?? false }

    /// "Quick installs now": starts Quick's download when it is missing and
    /// fits on disk with a margin. Returns whether it started.
    @discardableResult
    func startQuickIfNeeded() -> Bool {
        guard let quick, quick.state == .notInstalled, library.machine.hasRoom(for: quick.sizeBytes) else {
            return false
        }
        library.download(quick.id)
        return true
    }

    /// The "Download Deep" button (also "Download Deep again" after a
    /// failure): the person's explicit approval, the only path that starts
    /// Deep here. Checks the disk margin.
    func approveDeep() {
        guard let deep, offersDeep else { return }
        declinedDeep = false
        ModelLibraryView.approveDownload(deep, in: library)
    }

    public var body: some View {
        HStack(spacing: 0) {
            OnboardingSidebar(step: step)
                .frame(width: OnboardingMetrics.sidebarWidth)
                .background(LinearGradient(colors: [MFColor.accentFillFaint, MFColor.sidebar],
                                           startPoint: .top, endPoint: .bottom))
            Rectangle().fill(MFColor.hairline).frame(width: MFLayout.hairline)
            VStack(alignment: .leading, spacing: 0) {
                MFScroll {
                    VStack(alignment: .leading, spacing: MFSpacing.xl) {
                        OnboardingPicture(step: step, library: library)
                        content
                    }
                    .id(step)
                    .transition(.asymmetric(insertion: .move(edge: .trailing).combined(with: .opacity),
                                            removal: .opacity))
                    .padding(.top, OnboardingMetrics.contentTop)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                footer
            }
            .padding(.horizontal, OnboardingMetrics.contentPadding)
            .background(MFColor.surface)
        }
        .onChange(of: step, initial: true) { _, step in
            if step == .models { startQuickIfNeeded() }
        }
    }

    @ViewBuilder
    private var content: some View {
        switch step {
        case .welcome: welcome
        case .models: models
        case .permissions: permissions
        case .mail: mail
        }
    }

    // MARK: Steps

    private var welcome: some View {
        VStack(alignment: .leading, spacing: MFSpacing.l) {
            pageTitle("Welcome to Mference")
            pageText("An assistant that runs entirely on this Mac. Your chats, mail and files stay here.")
            MFCard {
                MFListRow("Runs on this Mac", subtitle: "The model runs on this computer. Nothing you type is sent to a server, unless you turn on web search.")
                MFListRow("No account", subtitle: "Nothing to sign up for, and no usage data sent anywhere.")
                MFListRow("You decide what it sees", subtitle: "Calendar, mail and the web stay off until you turn them on.")
            }
        }
    }

    private var models: some View {
        VStack(alignment: .leading, spacing: MFSpacing.l) {
            VStack(alignment: .leading, spacing: MFSpacing.s) {
                pageTitle("Choose your models")
                Text("We checked this Mac: \(Text(Self.machineFacts(library.machine)).fontWeight(.semibold).foregroundStyle(MFColor.ink)). \(modelsIntro)")
                    .font(MFFont.body)
                    .foregroundStyle(MFColor.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            if let quick {
                ModelChoiceCard(entry: quick, isSelected: true, badge: .custom("Recommended for every Mac", .success),
                                detail: "Handles most things: chat, web answers, mail drafts, sorting, scanning.") {
                    quickStatus(quick)
                }
            } else {
                MFCard {
                    MFListRow("Quick isn't available in this build yet",
                              subtitle: "You can add a model later in Settings > Models.")
                }
            }
            if offersDeep, let deep {
                ModelChoiceCard(entry: deep, isSelected: deep.state.isInstalled || deep.state.transfer != nil,
                                badge: .custom("Offered on Macs over 16 GB", .accent),
                                detail: "For harder reasoning, long documents and multi-step work. Slower, and uses more memory while it runs.") {
                    deepOffer(deep)
                }
            }
            Text(library.machine.canOfferDeep
                 ? "Macs with 16 GB or less get Quick only. You can add or remove models any time in Settings."
                 : "This Mac has \(ModelFormat.memory(library.machine.physicalMemoryBytes)) of memory, so Mference uses Quick. You can add or remove models any time in Settings.")
                .font(MFFont.preview)
                .foregroundStyle(MFColor.secondary)
        }
    }

    /// "Apple M1 Max, 32 GB memory, 582 GB free"
    static func machineFacts(_ machine: MachineSummary) -> String {
        String(ModelFormat.machineLine(machine).dropFirst("This Mac: ".count).dropLast())
    }

    var modelsIntro: String {
        let quickLine: String = switch quick?.state {
        case nil: "Quick isn't available in this build yet."
        case .loadFailed: "Quick is installed but didn't load."
        case .loaded, .ready, .loading: "Quick is already installed."
        case .downloading: "Quick is installing now."
        case .paused: "Quick's download is paused."
        case .downloadFailed: "Quick didn't download."
        case .notInstalled:
            library.machine.hasRoom(for: quick?.sizeBytes) ? "Quick installs now." : "Quick needs more free space."
        }
        // The offer stands only while Deep is untouched; once it downloads,
        // pauses or fails, its card says so.
        return quickLine + (offersDeep && deep?.state == .notInstalled
                            ? " Deep also fits, and only downloads if you say yes." : "")
    }

    /// Retry Quick's failed download (no approval needed); checks the disk
    /// margin again. Returns whether it restarted.
    @discardableResult
    func retryDownload(_ entry: ModelEntry) -> Bool {
        ModelLibraryView.retryDownload(entry, in: library)
    }

    /// Quick's line under its card, from its state.
    @ViewBuilder
    private func quickStatus(_ quick: ModelEntry) -> some View {
        let size = quick.sizeBytes.map(ModelFormat.bytes)
        switch quick.state {
        case .notInstalled:
            if library.machine.hasRoom(for: quick.sizeBytes) {
                statusLine("Installing Quick\(size.map { ", \($0)" } ?? "").")
            } else {
                retryLine("Quick needs \(size ?? "more space") plus some room to spare; \(library.machine.freeDiskBytes.map { ModelFormat.bytes($0) } ?? "less") is free. Free some space, then try again.",
                          label: "Try again", accessibility: "Try installing Quick again") {
                    startQuickIfNeeded()
                }
            }
        case .downloading(let progress):
            statusLine("Installing Quick\(size.map { ", \($0)" } ?? "")\(MFProgress.percent(progress.fraction).map { " · \($0)%" } ?? "").")
        case .paused:
            retryLine("Quick's download is paused.", label: "Resume", accessibility: "Resume installing Quick") {
                library.resume(quick.id)
            }
        case .downloadFailed(let message):
            if library.machine.hasRoom(for: quick.sizeBytes) {
                retryLine("Quick didn't download: \(message)", label: "Retry", accessibility: "Retry installing Quick") {
                    retryDownload(quick)
                }
            } else {
                statusLine("Quick didn't download: \(message) It needs \(size ?? "more space") plus some room to spare; free some space, then come back to Settings > Models.")
            }
        case .loadFailed(let message):
            retryLine("Quick is installed but didn't load: \(message)", label: "Retry", accessibility: "Retry loading Quick") {
                library.retryLoad(quick.id)
            }
        case .loaded, .ready, .loading:
            EmptyView()
        }
    }

    /// Deep's offer under its card, from its state and the person's answer.
    @ViewBuilder
    private func deepOffer(_ deep: ModelEntry) -> some View {
        switch deep.state {
        case .loaded, .ready, .loading, .loadFailed:
            EmptyView()
        case .downloading(let progress):
            statusLine("Deep is downloading\(MFProgress.percent(progress.fraction).map { " · \($0)%" } ?? ""). You can keep going; it is ready when the download finishes.")
        case .paused:
            retryLine("Deep's download is paused.", label: "Resume", accessibility: "Resume downloading Deep") {
                library.resume(deep.id)
            }
        case .downloadFailed(let message):
            retryLine("Deep didn't download: \(message)", label: "Download Deep again",
                      accessibility: "Download Deep again") {
                approveDeep()
            }
        case .notInstalled:
            if declinedDeep {
                HStack {
                    statusLine("Not now. You can download Deep later in Settings > Models.")
                    Spacer()
                    Button("Download Deep") { approveDeep() }
                        .buttonStyle(MFButtonStyle(.secondary, size: .medium))
                }
            } else {
                HStack(spacing: MFSpacing.m) {
                    Text("\(Text("Download Deep too?").fontWeight(.semibold).foregroundStyle(MFColor.ink)) \(deep.sizeBytes.map { ModelFormat.bytes($0) + " more. " } ?? "")Nothing downloads until you choose.")
                        .font(MFFont.bodySmall)
                        .foregroundStyle(MFColor.secondary)
                        .frame(maxWidth: .infinity, alignment: .leading)
                    Button("Not now") { declinedDeep = true }
                        .buttonStyle(MFButtonStyle(.secondary, size: .medium))
                    Button("Download Deep") { approveDeep() }
                        .buttonStyle(MFButtonStyle(.primary, size: .medium))
                }
                .padding(MFSpacing.comfortable)
                .background(RoundedRectangle(cornerRadius: MFRadius.m).fill(MFColor.accentFillFaint))
                .overlay(RoundedRectangle(cornerRadius: MFRadius.m).strokeBorder(MFColor.accentFill, lineWidth: MFLayout.hairline))
            }
        }
    }

    private func statusLine(_ text: String) -> some View {
        Text(text)
            .font(MFFont.callout)
            .foregroundStyle(MFColor.secondary)
            .fixedSize(horizontal: false, vertical: true)
    }

    private func retryLine(_ text: String, label: String, accessibility: String,
                           action: @escaping () -> Void) -> some View {
        HStack(spacing: MFSpacing.m) {
            statusLine(text)
                .frame(maxWidth: .infinity, alignment: .leading)
            Button(label, action: action)
                .buttonStyle(MFButtonStyle(.secondary, size: .medium))
                .accessibilityLabel(accessibility)
        }
    }

    private var permissions: some View {
        VStack(alignment: .leading, spacing: MFSpacing.l) {
            pageTitle("Permissions, as you need them")
            pageText("Nothing is asked now. The first time a feature needs access, Mference says why in one line, then macOS asks you.")
            MFCard {
                ForEach(OnboardingMetrics.permissionRows, id: \.0) { permission, title in
                    MFListRow(title, subtitle: permission.reason) {
                        MFBadge(.custom("When needed", .neutral))
                    }
                }
            }
        }
    }

    private var mail: some View {
        VStack(alignment: .leading, spacing: MFSpacing.l) {
            pageTitle("Connect mail (optional)")
            pageText("Mference can read Apple Mail on this Mac, sort it and draft replies that send only when you click Send. Mail never leaves this Mac.")
            MFCard {
                MFListRow("Apple Mail · every account in Mail on this Mac",
                          subtitle: "Connect it from the Mail tab or Settings > Connections when you want it.")
            }
        }
    }

    private func pageTitle(_ text: String) -> some View {
        Text(text)
            .font(MFFont.largeTitle)
            .foregroundStyle(MFColor.ink)
            .accessibilityAddTraits(.isHeader)
    }

    private func pageText(_ text: String) -> some View {
        Text(text)
            .font(MFFont.body)
            .foregroundStyle(MFColor.secondary)
            .fixedSize(horizontal: false, vertical: true)
    }

    // MARK: Footer

    private var footer: some View {
        VStack(spacing: 0) {
            Rectangle().fill(MFColor.hairline).frame(height: MFLayout.hairline)
            HStack(spacing: MFSpacing.m) {
                downloadStatus
                    .frame(maxWidth: .infinity, alignment: .leading)
                if step != .welcome {
                    Button("Back") { move(by: -1) }
                        .buttonStyle(MFButtonStyle(.secondary, size: .large))
                        .keyboardShortcut(.cancelAction)
                }
                Button(step == .mail ? "Start using Mference" : "Continue") {
                    if step == .mail { onFinish() } else { move(by: 1) }
                }
                .buttonStyle(MFButtonStyle(.primary, size: .large))
                .keyboardShortcut(.defaultAction)
            }
            .padding(.vertical, MFSpacing.l)
        }
    }

    /// "Downloading now: Quick, 37%", from the entries' states.
    static func downloadingSummary(_ entries: [ModelEntry]) -> String? {
        let parts = entries.compactMap { entry -> String? in
            guard case .downloading(let progress) = entry.state else { return nil }
            return [entry.productName, MFProgress.percent(progress.fraction).map { "\($0)%" }]
                .compactMap { $0 }.joined(separator: ", ")
        }
        return parts.isEmpty ? nil : parts.joined(separator: "; ")
    }

    @ViewBuilder
    private var downloadStatus: some View {
        if let summary = Self.downloadingSummary(library.entries) {
            Text("Downloading now: \(Text(summary).fontWeight(.semibold).foregroundStyle(MFColor.ink))")
                .font(MFFont.callout)
                .foregroundStyle(MFColor.secondary)
        }
    }

    private func move(by offset: Int) {
        guard let next = Step(rawValue: step.rawValue + offset) else { return }
        withAnimation(reduceMotion ? nil : .smooth(duration: 0.45)) { step = next }
    }
}

enum OnboardingMetrics {
    static let sidebarWidth: CGFloat = 280
    static let contentPadding: CGFloat = 48
    static let contentTop: CGFloat = 44
    static let markSide: CGFloat = 56
    static let markRadius: CGFloat = 15
    static let stepCircle: CGFloat = 22
    /// `.card` padding: 18 across, 16 down.
    static let cardPadding = CGSize(width: 18, height: 16)
    static let permissionRows: [(AppPermission, String)] = [
        (.calendar, "Calendar"),
        (.reminders, "Reminders"),
        (.camera, "Camera"),
        (.fullDiskAccess, "Full Disk Access, for Apple Mail"),
        (.mailAutomation, "Sending through Mail"),
    ]
}

/// The left column: mark, promise, steps, and the "No account" note.
struct OnboardingSidebar: View {
    let step: OnboardingView.Step
    @Environment(\.mfRendersStatically) private var rendersStatically

    var body: some View {
        VStack(alignment: .leading, spacing: MFSpacing.xl) {
            if rendersStatically {
                // Snapshot stand-in for the window buttons the real window draws here.
                ShellWindowButtonsPlaceholder()
            }
            Text("M")
                .font(MFFont.largeTitle)
                .foregroundStyle(MFColor.secondary)
                .frame(width: OnboardingMetrics.markSide, height: OnboardingMetrics.markSide)
                .background(RoundedRectangle(cornerRadius: OnboardingMetrics.markRadius)
                    .fill(LinearGradient(colors: [MFColor.surface, MFColor.hairlineStrong],
                                         startPoint: .topLeading, endPoint: .bottomTrailing)))
                .overlay(RoundedRectangle(cornerRadius: OnboardingMetrics.markRadius)
                    .strokeBorder(MFColor.hairlineStrong, lineWidth: MFLayout.hairline))
                .accessibilityHidden(true)
            VStack(alignment: .leading, spacing: MFSpacing.m) {
                Text("Your own AI,\non this Mac.")
                    .font(MFFont.largeTitle)
                    .foregroundStyle(MFColor.ink)
                Text("Mail, calendar, scanning and web answers, run by a model on your computer.")
                    .font(MFFont.body)
                    .foregroundStyle(MFColor.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            VStack(alignment: .leading, spacing: MFSpacing.comfortable) {
                ForEach(OnboardingView.Step.allCases) { item in
                    StepLabel(item: item, current: step)
                }
            }
            Spacer()
            HStack(alignment: .top, spacing: MFSpacing.compact) {
                Image(systemName: "lock")
                    .foregroundStyle(MFColor.success)
                    .accessibilityHidden(true)
                Text("\(Text("No account.").fontWeight(.semibold).foregroundStyle(MFColor.ink)) Nothing to sign up for, and no usage data sent anywhere.")
                    .font(MFFont.preview)
                    .foregroundStyle(MFColor.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .padding(MFSpacing.m)
            .background(RoundedRectangle(cornerRadius: MFRadius.m).fill(MFColor.surface))
            .overlay(RoundedRectangle(cornerRadius: MFRadius.m).strokeBorder(MFColor.hairline, lineWidth: MFLayout.hairline))
        }
        .padding(MFSpacing.xl)
        .padding(.top, rendersStatically ? 0 : MFSpacing.xxl)
        .frame(maxHeight: .infinity, alignment: .top)
    }
}

/// "✓ Welcome", "2 Models", ... with the current step emphasised.
struct StepLabel: View {
    let item: OnboardingView.Step
    let current: OnboardingView.Step

    var body: some View {
        HStack(spacing: MFSpacing.compact) {
            ZStack {
                if item.rawValue < current.rawValue {
                    Circle().fill(MFColor.ink)
                    Image(systemName: "checkmark")
                        .font(MFFont.chip.weight(.bold))
                        .foregroundStyle(MFColor.onInk)
                } else {
                    Circle().fill(MFColor.surface)
                    Circle().strokeBorder(item == current ? MFColor.ink : MFColor.hairlineStrong,
                                          lineWidth: item == current ? MFLayout.strokeEmphasis : MFLayout.hairline)
                    Text("\(item.rawValue + 1)")
                        .font(MFFont.chip.weight(.semibold))
                        .foregroundStyle(item == current ? MFColor.ink : MFColor.secondary)
                }
            }
            .frame(width: OnboardingMetrics.stepCircle, height: OnboardingMetrics.stepCircle)
            Text(item.title)
                .font(item == current ? MFFont.bodySmallEmphasis : MFFont.bodySmall)
                .foregroundStyle(item == current ? MFColor.ink : MFColor.secondary)
                .lineLimit(1)
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Step \(item.rawValue + 1), \(item.title)\(item.rawValue < current.rawValue ? ", done" : item == current ? ", current" : "")")
    }
}

/// A model card in the Models step.
struct ModelChoiceCard<Accessory: View>: View {
    let entry: ModelEntry
    let isSelected: Bool
    let badge: MFBadge.Kind
    let detail: String
    let accessory: Accessory

    init(entry: ModelEntry, isSelected: Bool, badge: MFBadge.Kind, detail: String,
         @ViewBuilder accessory: () -> Accessory) {
        self.entry = entry
        self.isSelected = isSelected
        self.badge = badge
        self.detail = detail
        self.accessory = accessory()
    }

    var body: some View {
        VStack(alignment: .leading, spacing: MFSpacing.compact) {
            HStack(spacing: MFSpacing.s) {
                if isSelected {
                    Image(systemName: "checkmark")
                        .font(MFFont.calloutSemibold)
                        .foregroundStyle(MFColor.ink)
                        .accessibilityHidden(true)
                }
                Text(entry.productName)
                    .font(MFFont.sectionTitle)
                    .foregroundStyle(MFColor.ink)
                if !entry.variantName.isEmpty {
                    MFChip(entry.variantName)
                }
                MFBadge(badge)
                if entry.state.isInstalled {
                    MFBadge(.installed)
                }
            }
            Text(detail)
                .font(MFFont.bodySmall)
                .foregroundStyle(MFColor.secondary)
                .fixedSize(horizontal: false, vertical: true)
            HStack(spacing: MFSpacing.snug) {
                if let size = entry.sizeBytes {
                    MFChip(entry.state.isInstalled ? "\(ModelFormat.bytes(size)) on disk" : "\(ModelFormat.bytes(size)) download")
                }
                if let licence = entry.licence {
                    MFChip(licence)
                }
            }
            accessory
        }
        .padding(.horizontal, OnboardingMetrics.cardPadding.width)
        .padding(.vertical, OnboardingMetrics.cardPadding.height)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(RoundedRectangle(cornerRadius: MFRadius.l).fill(MFColor.surface))
        .overlay(RoundedRectangle(cornerRadius: MFRadius.l)
            .strokeBorder(isSelected ? MFColor.ink : MFColor.hairline,
                          lineWidth: isSelected ? MFLayout.strokeEmphasis : MFLayout.hairline))
        .accessibilityElement(children: .contain)
        .accessibilityLabel("\(entry.productName)\(isSelected ? ", selected" : "")")
    }
}

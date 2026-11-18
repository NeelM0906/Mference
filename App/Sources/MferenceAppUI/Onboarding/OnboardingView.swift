import MferenceAppCore
import MferenceDesign
import SwiftUI

/// First run (`01-first-run.png`): no account, no permission asked up front
/// (L5, L6). Quick installs on every Mac; Deep is offered only on Macs with
/// more than 16 GB and downloads only after the person says yes (L4).
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

    /// Catalog ids of the two recommended models (spec 4.3.1).
    public static let quickID = "qwen3.5-4b"
    public static let deepID = "qwen3.8-27b"

    let library: any ModelLibraryModel
    let onFinish: () -> Void
    @State var step: Step
    /// The person's answer to "Download Deep too?": nil until they choose.
    @State var deepChoice: Bool?

    public init(library: any ModelLibraryModel, step: Step = .welcome, onFinish: @escaping () -> Void) {
        self.library = library
        self.onFinish = onFinish
        _step = State(initialValue: step)
    }

    var quick: ModelEntry? { library.entries.first { $0.id == Self.quickID } }
    var deep: ModelEntry? { library.entries.first { $0.id == Self.deepID } }

    /// Deep is offered only above 16 GB and when MemoryFit does not rule it out.
    var offersDeep: Bool {
        guard library.machine.canOfferDeep, let deep else { return false }
        if case .tooLarge = deep.fit { return false }
        return true
    }

    public var body: some View {
        HStack(spacing: 0) {
            OnboardingSidebar(step: step)
                .frame(width: OnboardingMetrics.sidebarWidth)
                .background(MFColor.accentFillFaint.opacity(0.5))
            Rectangle().fill(MFColor.hairline).frame(width: 1)
            VStack(alignment: .leading, spacing: 0) {
                MFScroll {
                    content
                        .padding(.top, OnboardingMetrics.contentTop)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                footer
            }
            .padding(.horizontal, OnboardingMetrics.contentPadding)
            .background(MFColor.surface)
        }
        .onChange(of: step, initial: true) { _, step in
            // "Quick installs now": the download starts when the Models step shows.
            if step == .models, let quick, quick.state == .notInstalled {
                library.download(quick.id)
            }
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
            Text("An assistant that runs entirely on this Mac. Your chats, mail and files stay here.")
                .font(MFFont.body)
                .foregroundStyle(MFColor.secondary)
            MFCard {
                MFListRow("Runs on this Mac", subtitle: "The model runs on this computer. Nothing you type is sent to a server.")
                MFListRow("No account", subtitle: "Nothing to sign up for, and no usage data sent anywhere.")
                MFListRow("You decide what it sees", subtitle: "Calendar, mail and the web stay off until you turn them on.")
            }
        }
    }

    private var models: some View {
        VStack(alignment: .leading, spacing: MFSpacing.l) {
            pageTitle("Choose your models")
            Text(modelsIntro)
                .font(MFFont.body)
                .foregroundStyle(MFColor.secondary)
                .fixedSize(horizontal: false, vertical: true)
            if let quick {
                ModelChoiceCard(entry: quick, isSelected: true, badge: .custom("Recommended for every Mac", .success),
                                detail: "Handles most things: chat, web answers, mail drafts, sorting, scanning.")
            }
            if offersDeep, let deep {
                ModelChoiceCard(entry: deep, isSelected: deepChoice == true || deep.state.isInstalled,
                                badge: .custom("Offered on Macs over 16 GB", .accent),
                                detail: "For harder reasoning, long documents and multi-step work. Slower, and uses more memory while it runs.") {
                    deepOffer(deep)
                }
            }
            Text(library.machine.canOfferDeep
                 ? "Macs with 16 GB or less get Quick only. You can add or remove models any time in Settings."
                 : "This Mac has \(ModelFormat.memory(library.machine.physicalMemoryBytes)) of memory, so Mference uses Quick. You can add or remove models any time in Settings.")
                .font(MFFont.callout)
                .foregroundStyle(MFColor.secondary)
        }
    }

    private var modelsIntro: String {
        let machine = ModelFormat.machineLine(library.machine)
            .replacingOccurrences(of: "This Mac: ", with: "We checked this Mac: ")
        let quickLine = quick?.state.isInstalled == true ? "Quick is already installed." : "Quick installs now."
        let deepLine = offersDeep ? " Deep also fits, and only downloads if you say yes." : ""
        return "\(machine) \(quickLine)\(deepLine)"
    }

    @ViewBuilder
    private func deepOffer(_ deep: ModelEntry) -> some View {
        if deep.state.isInstalled {
            EmptyView()
        } else if deep.state.transfer != nil || deepChoice == true {
            Text("Deep is downloading. You can keep going; it is ready when the download finishes.")
                .font(MFFont.callout)
                .foregroundStyle(MFColor.secondary)
        } else if deepChoice == false {
            HStack {
                Text("Not now. You can download Deep later in Settings > Models.")
                    .font(MFFont.callout)
                    .foregroundStyle(MFColor.secondary)
                Spacer()
                Button("Download Deep") { approveDeep(deep) }.buttonStyle(.mfSecondary)
            }
        } else {
            HStack(spacing: MFSpacing.m) {
                Text("\(Text("Download Deep too?").fontWeight(.semibold).foregroundStyle(MFColor.ink)) \(deep.sizeBytes.map { ModelFormat.bytes($0) + " more. " } ?? "")Nothing downloads until you choose.")
                    .font(MFFont.callout)
                    .foregroundStyle(MFColor.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
                Button("Not now") { deepChoice = false }.buttonStyle(.mfSecondary)
                Button("Download Deep") { approveDeep(deep) }.buttonStyle(.mfPrimary)
            }
            .padding(MFSpacing.m + 2)
            .background(RoundedRectangle(cornerRadius: MFRadius.m).fill(MFColor.accentFillFaint))
            .overlay(RoundedRectangle(cornerRadius: MFRadius.m).strokeBorder(MFColor.accentFill))
        }
    }

    private func approveDeep(_ deep: ModelEntry) {
        deepChoice = true
        library.download(deep.id)
    }

    private var permissions: some View {
        VStack(alignment: .leading, spacing: MFSpacing.l) {
            pageTitle("Permissions, as you need them")
            Text("Nothing is asked now. The first time a feature needs access, Mference says why in one line, then macOS asks you.")
                .font(MFFont.body)
                .foregroundStyle(MFColor.secondary)
                .fixedSize(horizontal: false, vertical: true)
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
            Text("Mference can read Apple Mail on this Mac, sort it and draft replies that send only when you click Send. Mail never leaves this Mac.")
                .font(MFFont.body)
                .foregroundStyle(MFColor.secondary)
                .fixedSize(horizontal: false, vertical: true)
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

    // MARK: Footer

    private var footer: some View {
        VStack(spacing: 0) {
            Rectangle().fill(MFColor.hairline).frame(height: 1)
            HStack(spacing: MFSpacing.m) {
                downloadStatus
                    .frame(maxWidth: .infinity, alignment: .leading)
                if step != .welcome {
                    Button("Back") { move(by: -1) }
                        .buttonStyle(.mfSecondary)
                        .keyboardShortcut(.cancelAction)
                }
                Button(step == .mail ? "Start using Mference" : "Continue") {
                    if step == .mail { onFinish() } else { move(by: 1) }
                }
                .buttonStyle(.mfPrimary)
                .keyboardShortcut(.defaultAction)
            }
            .padding(.vertical, MFSpacing.l + 2)
        }
    }

    @ViewBuilder
    private var downloadStatus: some View {
        let downloading = library.entries.filter { $0.state.transfer != nil }
        if !downloading.isEmpty {
            let names = downloading.map { entry in
                [entry.productName, entry.state.transfer?.fraction.map { "\(Int(($0 * 100).rounded(.down)))%" }]
                    .compactMap { $0 }.joined(separator: ", ")
            }.joined(separator: "; ")
            Text("Downloading now: \(Text(names).fontWeight(.semibold).foregroundStyle(MFColor.ink))")
                .font(MFFont.callout)
                .foregroundStyle(MFColor.secondary)
        }
    }

    private func move(by offset: Int) {
        if let next = Step(rawValue: step.rawValue + offset) { step = next }
    }
}

enum OnboardingMetrics {
    static let sidebarWidth: CGFloat = 300
    static let contentPadding: CGFloat = 48
    static let contentTop: CGFloat = 44
    static let markSide: CGFloat = 56
    static let markRadius: CGFloat = 14
    static let stepCircle: CGFloat = 22
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

    var body: some View {
        VStack(alignment: .leading, spacing: MFSpacing.xl) {
            Text("M")
                .font(MFFont.largeTitle)
                .fontWeight(.bold)
                .foregroundStyle(MFColor.secondary)
                .frame(width: OnboardingMetrics.markSide, height: OnboardingMetrics.markSide)
                .background(RoundedRectangle(cornerRadius: OnboardingMetrics.markRadius)
                    .fill(LinearGradient(colors: [MFColor.surface, MFColor.hairlineStrong],
                                         startPoint: .topLeading, endPoint: .bottomTrailing)))
                .overlay(RoundedRectangle(cornerRadius: OnboardingMetrics.markRadius).strokeBorder(MFColor.hairlineStrong))
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
            VStack(alignment: .leading, spacing: MFSpacing.m + 2) {
                ForEach(OnboardingView.Step.allCases) { item in
                    StepLabel(item: item, current: step)
                }
            }
            Spacer()
            HStack(alignment: .top, spacing: MFSpacing.m) {
                Image(systemName: "lock")
                    .foregroundStyle(MFColor.success)
                    .accessibilityHidden(true)
                Text("\(Text("No account.").fontWeight(.semibold).foregroundStyle(MFColor.ink)) Nothing to sign up for, and no usage data sent anywhere.")
                    .font(MFFont.callout)
                    .foregroundStyle(MFColor.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .padding(MFSpacing.l)
            .background(RoundedRectangle(cornerRadius: MFRadius.l).fill(MFColor.surface))
            .overlay(RoundedRectangle(cornerRadius: MFRadius.l).strokeBorder(MFColor.hairline))
        }
        .padding(MFSpacing.xl + 4)
        .padding(.top, MFSpacing.xxl)
        .frame(maxHeight: .infinity, alignment: .top)
    }
}

/// "✓ Welcome", "2 Models", ... with the current step in bold.
struct StepLabel: View {
    let item: OnboardingView.Step
    let current: OnboardingView.Step

    var body: some View {
        HStack(spacing: MFSpacing.m) {
            ZStack {
                if item.rawValue < current.rawValue {
                    Circle().fill(MFColor.ink)
                    Image(systemName: "checkmark")
                        .font(MFFont.caption.weight(.bold))
                        .foregroundStyle(MFColor.onInk)
                } else {
                    Circle().strokeBorder(item == current ? MFColor.ink : MFColor.hairlineStrong, lineWidth: 1.5)
                    Text("\(item.rawValue + 1)")
                        .font(MFFont.caption.weight(.semibold))
                        .foregroundStyle(item == current ? MFColor.ink : MFColor.secondary)
                }
            }
            .frame(width: OnboardingMetrics.stepCircle, height: OnboardingMetrics.stepCircle)
            Text(item.title)
                .font(item == current ? MFFont.bodyEmphasis.weight(.semibold) : MFFont.body)
                .foregroundStyle(item == current ? MFColor.ink : MFColor.secondary)
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
        VStack(alignment: .leading, spacing: MFSpacing.m) {
            HStack(spacing: MFSpacing.s) {
                if isSelected {
                    Image(systemName: "checkmark")
                        .font(MFFont.bodyEmphasis.weight(.semibold))
                        .foregroundStyle(MFColor.ink)
                        .accessibilityHidden(true)
                }
                Text(entry.productName)
                    .font(MFFont.title)
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
                .font(MFFont.body)
                .foregroundStyle(MFColor.secondary)
                .fixedSize(horizontal: false, vertical: true)
            HStack(spacing: MFSpacing.s) {
                if let size = entry.sizeBytes {
                    MFChip(entry.state.isInstalled ? "\(ModelFormat.bytes(size)) on disk" : "\(ModelFormat.bytes(size)) download")
                }
                if let licence = entry.licence {
                    MFChip(licence)
                }
            }
            accessory
        }
        .padding(MFSpacing.xl - 4)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(RoundedRectangle(cornerRadius: MFRadius.l).fill(MFColor.surface))
        .overlay(RoundedRectangle(cornerRadius: MFRadius.l)
            .strokeBorder(isSelected ? MFColor.ink : MFColor.hairline, lineWidth: isSelected ? 2 : 1))
        .accessibilityElement(children: .contain)
        .accessibilityLabel("\(entry.productName)\(isSelected ? ", selected" : "")")
    }
}

extension ModelChoiceCard where Accessory == EmptyView {
    init(entry: ModelEntry, isSelected: Bool, badge: MFBadge.Kind, detail: String) {
        self.init(entry: entry, isSelected: isSelected, badge: badge, detail: detail) { EmptyView() }
    }
}

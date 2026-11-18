import MferenceAppCore
import MferenceDesign
import SwiftUI

/// Mail (mockup 03): Apple Mail on this Mac, sorted into renamable buckets,
/// read as text with trackers removed, a reply draft sent only on click.
public struct MailView: View {
    @Environment(MailInboxModel.self) private var environmentModel: MailInboxModel?
    private let injected: MailInboxModel?
    private let loadsOnAppear: Bool

    /// The shell's slot: the model comes from the environment.
    public init() {
        injected = nil
        loadsOnAppear = true
    }

    /// Snapshots and previews; `loadsOnAppear` false keeps fixture state.
    public init(model: MailInboxModel, loadsOnAppear: Bool = true) {
        injected = model
        self.loadsOnAppear = loadsOnAppear
    }

    public var body: some View {
        if let model = injected ?? environmentModel {
            MailContent(model: model)
                .task { if loadsOnAppear { await model.load() } }
        } else {
            AppleEmptyState(systemImage: "envelope", title: "Mail", message: "Mail is starting. Try again in a moment.")
        }
    }
}

struct MailContent: View {
    @Bindable var model: MailInboxModel

    var body: some View {
        Group {
            switch model.status {
            case .available, nil:
                GeometryReader { proxy in
                    // Columns scale with the window: about 200 and 300 pt at
                    // the mockup's width, the reader takes the rest.
                    let sidebar = min(max(proxy.size.width * 0.17, 190), 220)
                    let list = min(max(proxy.size.width * 0.25, 280), 350)
                    HStack(spacing: 0) {
                        MailSidebar(model: model).frame(width: sidebar)
                        divider
                        MailList(model: model).frame(width: list)
                        divider
                        MailReader(model: model).frame(maxWidth: .infinity, maxHeight: .infinity)
                    }
                }
            case .needsFullDiskAccess:
                AppleEmptyState(systemImage: "lock.shield", title: "Read Apple Mail on this Mac",
                                message: AppPermission.fullDiskAccess.reason
                                    + " Turn on Mference under Privacy & Security > Full Disk Access.",
                                actionTitle: "Open System Settings") { Task { await model.requestAccess() } }
            case .notFound:
                AppleEmptyState(systemImage: "envelope", title: "Apple Mail is not set up",
                                message: "Add an account in Apple Mail first. Mference reads it here, on this Mac.")
            case .unsupported:
                AppleEmptyState(systemImage: "exclamationmark.triangle", title: "This Mail version is not supported yet",
                                message: model.status?.userMessage ?? "")
            }
        }
        .background(MFColor.surface)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var divider: some View { Rectangle().fill(MFColor.hairline).frame(width: 1) }
}

// MARK: Sidebar

struct MailSidebar: View {
    @Bindable var model: MailInboxModel
    @State private var renaming = false

    var body: some View {
        VStack(alignment: .leading, spacing: MFSpacing.xs) {
            HStack {
                AppleSectionHeader(text: "Sorted for you")
                Spacer()
                Button(renaming ? "Done" : "Rename") { renaming.toggle() }
                    .buttonStyle(.plain)
                    .font(MFFont.calloutEmphasis)
                    .foregroundStyle(MFColor.accent)
            }
            .padding(.horizontal, MFSpacing.s)
            .padding(.bottom, MFSpacing.xs)
            ForEach(model.buckets, id: \.key) { bucket in
                if renaming {
                    BucketNameField(name: bucket.name) { model.rename(bucket: bucket.key, to: $0) }
                } else {
                    folderRow(bucket.name, folder: .bucket(bucket.key))
                }
            }
            AppleSectionHeader(text: "Mail")
                .padding(.horizontal, MFSpacing.s)
                .padding(.top, MFSpacing.l)
                .padding(.bottom, MFSpacing.xs)
            folderRow("Done", folder: .done, showsZero: false)
            folderRow("Drafts", folder: .drafts)
            folderRow("Sent", folder: .sent, showsCount: false)
            if !model.accounts.isEmpty {
                AppleSectionHeader(text: "Accounts")
                    .padding(.horizontal, MFSpacing.s)
                    .padding(.top, MFSpacing.l)
                    .padding(.bottom, MFSpacing.xs)
                ForEach(Array(model.accounts.enumerated()), id: \.element.id) { index, account in
                    HStack(spacing: MFSpacing.s) {
                        Circle().fill(index.isMultiple(of: 2) ? MFColor.accent : MFColor.preview)
                            .frame(width: MFSpacing.s, height: MFSpacing.s)
                            .accessibilityHidden(true)
                        Text(account.name)
                            .font(MFFont.body)
                            .foregroundStyle(MFColor.ink)
                            .lineLimit(1)
                    }
                    .padding(.horizontal, MFSpacing.s)
                    .padding(.vertical, MFSpacing.xs + MFSpacing.xxs)
                }
            }
            Spacer()
        }
        .padding(MFSpacing.m + MFSpacing.xs)
        .background(MFColor.canvas)
    }

    private func folderRow(_ title: String, folder: MailInboxModel.Folder, showsCount: Bool = true,
                           showsZero: Bool = true) -> some View {
        let selected = model.folder == folder
        let count = model.count(in: folder)
        return Button { Task { await model.select(folder) } } label: {
            HStack {
                Text(title)
                    .font(MFFont.body)
                    .foregroundStyle(MFColor.ink)
                Spacer()
                if showsCount, count > 0 || !showsZero {
                    if count > 0 {
                        Text("\(count)")
                            .font(MFFont.callout)
                            .foregroundStyle(MFColor.secondary)
                    }
                }
            }
            .padding(.horizontal, MFSpacing.s + MFSpacing.xxs)
            .padding(.vertical, MFSpacing.s)
            .background(RoundedRectangle(cornerRadius: MFRadius.m).fill(selected ? MFColor.fill : .clear))
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityLabel("\(title), \(count) messages")
        .accessibilityAddTraits(selected ? .isSelected : [])
    }
}

struct BucketNameField: View {
    @State var name: String
    var commit: (String) -> Void

    var body: some View {
        AppleTextField(title: "Bucket name", text: $name)
            .font(MFFont.body)
            .padding(.horizontal, MFSpacing.s + MFSpacing.xxs)
            .padding(.vertical, MFSpacing.s)
            .background(RoundedRectangle(cornerRadius: MFRadius.m).strokeBorder(MFColor.hairlineStrong))
            .onSubmit { commit(name) }
            .onDisappear { commit(name) }
            .accessibilityLabel("Bucket name")
    }
}

// MARK: List

struct MailList: View {
    @Bindable var model: MailInboxModel

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: MFSpacing.s) {
                Text(model.folderTitle)
                    .font(MFFont.headline)
                    .foregroundStyle(MFColor.ink)
                    .accessibilityAddTraits(.isHeader)
                Text("\(model.messages.count)")
                    .font(MFFont.calloutEmphasis)
                    .foregroundStyle(MFColor.secondary)
                    .padding(.horizontal, MFSpacing.s)
                    .padding(.vertical, MFSpacing.xxs)
                    .background(Capsule().fill(MFColor.fill))
                if model.isSorting {
                    ProgressView().controlSize(.small).accessibilityLabel("Sorting")
                }
                Spacer()
            }
            .padding(.horizontal, MFSpacing.l + MFSpacing.xs)
            .padding(.vertical, MFSpacing.l)
            Rectangle().fill(MFColor.hairlineFaint).frame(height: 1)
            FitOrScroll {
                VStack(spacing: 0) {
                    ForEach(model.messages) { message in
                        MailRow(message: message, selected: model.selected?.messageID == message.messageID)
                            .onTapGesture { Task { await model.open(message) } }
                            .accessibilityAddTraits(.isButton)
                        Rectangle().fill(MFColor.hairlineFaint).frame(height: 1)
                    }
                    if model.messages.isEmpty {
                        Text("Nothing here.")
                            .font(MFFont.body)
                            .foregroundStyle(MFColor.secondary)
                            .padding(MFSpacing.xl)
                    }
                }
            }
            .frame(maxHeight: .infinity, alignment: .top)
            Rectangle().fill(MFColor.hairlineFaint).frame(height: 1)
            shortcuts
        }
    }

    private var shortcuts: some View {
        HStack(spacing: MFSpacing.m) {
            shortcut("E", "Done") { if let message = model.selected { model.markDone(message) } }
                .keyboardShortcut("e", modifiers: [])
            shortcut("Z", "Undo") { model.undo() }
                .keyboardShortcut("z", modifiers: [])
            shortcut("⌘J", "Draft") { Task { await model.writeDraft() } }
                .keyboardShortcut("j", modifiers: .command)
            Spacer()
        }
        .padding(.horizontal, MFSpacing.l)
        .padding(.vertical, MFSpacing.m)
    }

    private func shortcut(_ key: String, _ label: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            HStack(spacing: MFSpacing.xs + MFSpacing.xxs) {
                Text(key)
                    .font(MFFont.caption)
                    .foregroundStyle(MFColor.secondary)
                    .padding(.horizontal, MFSpacing.xs + MFSpacing.xxs)
                    .padding(.vertical, MFSpacing.xxs)
                    .background(RoundedRectangle(cornerRadius: MFRadius.xs).strokeBorder(MFColor.hairlineStrong))
                Text(label)
                    .font(MFFont.callout)
                    .foregroundStyle(MFColor.secondary)
            }
        }
        .buttonStyle(.plain)
        .accessibilityLabel(label)
    }
}

struct MailRow: View {
    var message: MailMessageSummary
    var selected: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: MFSpacing.xxs) {
            HStack(alignment: .firstTextBaseline) {
                Text(message.senderName.isEmpty ? message.senderAddress : message.senderName)
                    .font(MFFont.callout.weight(message.isRead ? .medium : .semibold))
                    .foregroundStyle(MFColor.ink)
                    .lineLimit(1)
                Spacer()
                Text(Self.timeLabel(message.date))
                    .font(MFFont.caption)
                    .foregroundStyle(MFColor.secondary)
            }
            Text(message.subject)
                .font(MFFont.calloutEmphasis)
                .foregroundStyle(MFColor.ink)
                .lineLimit(1)
            Text(message.snippet)
                .font(MFFont.caption)
                .foregroundStyle(MFColor.secondary)
                .lineSpacing(MFSpacing.xxs)
                .lineLimit(2)
        }
        .padding(.horizontal, MFSpacing.l + MFSpacing.xs)
        .padding(.vertical, MFSpacing.m + MFSpacing.xxs)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(selected ? MFColor.accentFillFaint : .clear)
        .overlay(alignment: .leading) {
            Rectangle().fill(selected ? MFColor.accent : .clear).frame(width: MFSpacing.xs)
        }
        .contentShape(Rectangle())
        .accessibilityElement(children: .combine)
        .accessibilityAddTraits(selected ? .isSelected : [])
    }

    static func timeLabel(_ date: Date, now: Date = .now) -> String {
        let calendar = Calendar.current
        if calendar.isDate(date, inSameDayAs: now) { return date.formatted(date: .omitted, time: .shortened) }
        if let yesterday = calendar.date(byAdding: .day, value: -1, to: now), calendar.isDate(date, inSameDayAs: yesterday) {
            return "Yesterday"
        }
        return date.formatted(.dateTime.month(.abbreviated).day())
    }
}

// MARK: Reader

struct MailReader: View {
    @Bindable var model: MailInboxModel

    var body: some View {
        if let message = model.selected {
            FitOrScroll {
                VStack(alignment: .leading, spacing: MFSpacing.l) {
                    Text(message.subject)
                        .font(MFFont.title)
                        .foregroundStyle(MFColor.ink)
                        .accessibilityAddTraits(.isHeader)
                    senderLine(message)
                    if let body = model.body {
                        Text(body.text)
                            .font(MFFont.body)
                            .foregroundStyle(MFColor.ink)
                            .lineSpacing(MFSpacing.xs + MFSpacing.xxs)
                            .textSelection(.enabled)
                            .fixedSize(horizontal: false, vertical: true)
                    } else {
                        ProgressView().controlSize(.small)
                    }
                    if let card = model.eventCard { eventCard(card) }
                    DraftCard(model: model)
                }
                .padding(.horizontal, MFSpacing.xl + MFSpacing.s)
                .padding(.vertical, MFSpacing.xl)
            }
        } else {
            AppleEmptyState(systemImage: "tray", title: "No message selected",
                            message: "Pick a message to read it here.")
        }
    }

    private func senderLine(_ message: MailMessageSummary) -> some View {
        HStack(spacing: MFSpacing.s) {
            Text(message.initials)
                .font(MFFont.calloutEmphasis)
                .foregroundStyle(MFColor.accentStrong)
                .frame(width: MFSpacing.xxl, height: MFSpacing.xxl)
                .background(Circle().fill(MFColor.accentFill))
                .accessibilityHidden(true)
            Text("\(Text(message.senderName.isEmpty ? message.senderAddress : message.senderName).font(MFFont.bodyEmphasis).foregroundStyle(MFColor.ink)) \(Text("to you · \(MailRow.timeLabel(message.date))").foregroundStyle(MFColor.secondary))")
                .font(MFFont.body)
            if let trackers = model.body?.trackersRemoved, trackers > 0 {
                AppleChip(text: trackers == 1 ? "1 tracker removed" : "\(trackers) trackers removed", tone: .success)
            }
            if let blocked = model.body?.remoteImagesBlocked, blocked > 0 {
                AppleChip(text: blocked == 1 ? "1 remote image not loaded" : "\(blocked) remote images not loaded")
            }
        }
    }

    private func eventCard(_ card: ProposedEvent) -> some View {
        AppleActionCard {
            HStack(spacing: MFSpacing.m) {
                Image(systemName: "calendar")
                    .font(MFFont.headline)
                    .foregroundStyle(MFColor.accent)
                    .frame(width: MFSpacing.xxl + MFSpacing.s, height: MFSpacing.xxl + MFSpacing.s)
                    .background(RoundedRectangle(cornerRadius: MFRadius.m).fill(MFColor.surface))
                    .overlay(RoundedRectangle(cornerRadius: MFRadius.m).strokeBorder(MFColor.accentFill))
                    .accessibilityHidden(true)
                VStack(alignment: .leading, spacing: MFSpacing.xxs) {
                    Text("\(card.title) · \(CalendarTimeFormat.dayRange(card.start, card.end))")
                        .font(MFFont.bodyEmphasis)
                        .foregroundStyle(MFColor.ink)
                        .lineLimit(1)
                    Text("Nothing changes until you add it.")
                        .font(MFFont.body)
                        .foregroundStyle(MFColor.secondary)
                }
                Spacer(minLength: MFSpacing.m)
                Button("Add to calendar") { model.addEventCardToCalendar() }
                    .buttonStyle(.applePrimary)
            }
        }
    }
}

struct DraftCard: View {
    @Bindable var model: MailInboxModel
    @State private var editing = false

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(spacing: MFSpacing.xs) {
                Text("Draft reply").font(MFFont.bodyEmphasis).foregroundStyle(MFColor.ink)
                Text("· in your voice\(model.draftModelLabel.map { " · \($0)" } ?? "")")
                    .font(MFFont.body)
                    .foregroundStyle(MFColor.secondary)
                Spacer()
                if model.draftState == .ready || model.draftState == .idle && !model.draft.isEmpty {
                    Button("Shorter") { Task { await model.writeDraft(tone: .shorter) } }.buttonStyle(.appleQuiet)
                    Button("Warmer") { Task { await model.writeDraft(tone: .warmer) } }.buttonStyle(.appleQuiet)
                }
            }
            .padding(.horizontal, MFSpacing.l + MFSpacing.xs)
            .padding(.vertical, MFSpacing.m)
            .background(MFColor.canvas)
            Rectangle().fill(MFColor.hairline).frame(height: 1)
            content
                .padding(.horizontal, MFSpacing.l + MFSpacing.xs)
                .padding(.vertical, MFSpacing.l)
            Rectangle().fill(MFColor.hairline).frame(height: 1)
            HStack(spacing: MFSpacing.m) {
                Button("Send") { Task { await model.send() } }
                    .buttonStyle(.applePrimary)
                    .disabled(model.draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                              || model.draftState == .sending || model.draftState == .sent)
                Button(editing ? "Done" : "Edit") { editing.toggle() }
                    .buttonStyle(.appleSecondary)
                    .disabled(model.draft.isEmpty)
                Text(footer)
                    .font(MFFont.callout)
                    .foregroundStyle(MFColor.secondary)
            }
            .padding(.horizontal, MFSpacing.l + MFSpacing.xs)
            .padding(.vertical, MFSpacing.m)
        }
        .background(RoundedRectangle(cornerRadius: MFRadius.l).fill(MFColor.surface))
        .clipShape(RoundedRectangle(cornerRadius: MFRadius.l))
        .overlay(RoundedRectangle(cornerRadius: MFRadius.l).strokeBorder(MFColor.hairlineStrong))
    }

    @ViewBuilder
    private var content: some View {
        switch model.draftState {
        case .writing:
            HStack(spacing: MFSpacing.s) {
                ProgressView().controlSize(.small)
                Text("Writing a draft on this Mac…").font(MFFont.body).foregroundStyle(MFColor.secondary)
            }
            .frame(maxWidth: .infinity, minHeight: 100, alignment: .topLeading)
        case .idle where model.draft.isEmpty:
            Button("Write a draft") { Task { await model.writeDraft() } }
                .buttonStyle(.appleSecondary)
                .frame(maxWidth: .infinity, minHeight: 100, alignment: .topLeading)
        case .failed(let message) where model.draft.isEmpty:
            Text(message).font(MFFont.body).foregroundStyle(MFColor.danger)
                .frame(maxWidth: .infinity, minHeight: 100, alignment: .topLeading)
        default:
            if editing {
                TextEditor(text: $model.draft)
                    .font(MFFont.body)
                    .scrollContentBackground(.hidden)
                    .frame(minHeight: 120)
                    .accessibilityLabel("Draft reply")
            } else {
                Text(model.draft)
                    .font(MFFont.body)
                    .foregroundStyle(MFColor.ink)
                    .lineSpacing(MFSpacing.xs)
                    .frame(maxWidth: .infinity, minHeight: 100, alignment: .topLeading)
                    .textSelection(.enabled)
            }
        }
    }

    private var footer: String {
        switch model.draftState {
        case .sending: "Sending through Mail…"
        case .sent: "Sent through Mail."
        case .failed(let message): message
        default: "Not sent until you press Send."
        }
    }
}

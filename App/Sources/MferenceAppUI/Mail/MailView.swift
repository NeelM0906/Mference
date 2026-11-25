import AppKit
import MferenceAppCore
import MferenceDesign
import SwiftUI

/// Mail (mockup 03): Apple Mail on this Mac with Mail's own mailboxes,
/// followed as Mail saves changes, read as text with trackers removed, a
/// reply draft sent only on click.
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
                .task {
                    guard loadsOnAppear else { return }
                    await model.load()
                    await model.watch()
                }
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
                // Compose docks over the bottom left, not modal: the list
                // and the reader stay usable. Only once Mail can be read.
                .overlay(alignment: .bottomLeading) {
                    if case .available = model.status, model.composer.isOpen {
                        MailComposeCard(composer: model.composer)
                            .padding(MFSpacing.m)
                    }
                }
            case .needsFullDiskAccess:
                MailAccessState(model: model)
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
        // Coming back from System Settings: look again while access is
        // missing; otherwise look for new mail at once.
        .onReceive(NotificationCenter.default.publisher(for: NSApplication.didBecomeActiveNotification)) { _ in
            Task { await model.appBecameActive() }
        }
        .onReceive(NotificationCenter.default.publisher(for: NSApplication.didResignActiveNotification)) { _ in
            model.appResignedActive()
        }
        .shellToolbar(id: "mail.compose") { ComposeToolbarButton(model: model) }
    }

    private var divider: some View { Rectangle().fill(MFColor.hairline).frame(width: 1) }
}

// MARK: Sidebar

/// Layout sizes of the Mail screen.
enum MailMetrics {
    /// The mailbox icon's column in the sidebar.
    static let iconWidth: CGFloat = 16
    /// The unread dot in the sidebar's sync note and the message list.
    static let dot: CGFloat = 7
    /// The docked compose card, sized to fit the minimum window (960×640).
    static let composeWidth: CGFloat = 480
    static let composeMaxHeight: CGFloat = 460
    static let composeBodyMinHeight: CGFloat = 96
    static let composeMinimizedWidth: CGFloat = 280
    static let composeLabelWidth: CGFloat = 52
    static let composeShadowOpacity: Double = 0.14
    static let composeShadowRadius: CGFloat = 20
    static let composeShadowY: CGFloat = 10
}

struct MailSidebar: View {
    @Bindable var model: MailInboxModel

    var body: some View {
        VStack(spacing: 0) {
            // Scrolls when many accounts or a tall window's text size need it.
            MFScroll {
                VStack(alignment: .leading, spacing: MFSpacing.l) {
                    section("Favorites") {
                        folderRow("Inbox", icon: icon(for: .inbox), folder: .favorite(.inbox))
                        if model.offersFlagged {
                            folderRow("Flagged", icon: "flag", folder: .flagged)
                        }
                        folderRow("Drafts", icon: icon(for: .drafts), folder: .favorite(.drafts))
                        folderRow("Sent", icon: icon(for: .sent), folder: .favorite(.sent))
                    }
                    if !model.localMailboxes.isEmpty {
                        section("On My Mac") {
                            ForEach(model.localMailboxes) { mailbox in mailboxRow(mailbox) }
                        }
                    }
                    ForEach(model.accountSections) { section in accountSection(section) }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(MFSpacing.m + MFSpacing.xs)
            }
            MailSyncNote(model: model)
                .padding(.horizontal, MFSpacing.m)
                .padding(.bottom, MFSpacing.m)
        }
        .background(MFColor.canvas)
    }

    private func section<Content: View>(_ title: String, @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: MFSpacing.xxs) {
            AppleSectionHeader(text: title)
                .padding(.horizontal, MFSpacing.s)
                .padding(.bottom, MFSpacing.xs)
            content()
        }
    }

    private func accountSection(_ section: MailInboxModel.AccountSection) -> some View {
        let collapsed = model.collapsedAccounts.contains(section.id)
        return VStack(alignment: .leading, spacing: MFSpacing.xxs) {
            Button {
                if collapsed { model.collapsedAccounts.remove(section.id) } else { model.collapsedAccounts.insert(section.id) }
            } label: {
                HStack(spacing: MFSpacing.xs) {
                    AppleSectionHeader(text: section.account.title)
                        .lineLimit(1)
                    Spacer(minLength: MFSpacing.xs)
                    Image(systemName: collapsed ? "chevron.right" : "chevron.down")
                        .font(MFFont.caption)
                        .foregroundStyle(MFColor.tertiary)
                        .accessibilityHidden(true)
                }
                .padding(.horizontal, MFSpacing.s)
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .accessibilityLabel("\(section.account.title), \(collapsed ? "collapsed" : "expanded")")
            .accessibilityHint(collapsed ? "Shows this account's mailboxes" : "Hides this account's mailboxes")
            if let subtitle = accountSubtitle(section, collapsed: collapsed) {
                Text(subtitle)
                    .font(MFFont.caption)
                    .foregroundStyle(MFColor.tertiary)
                    .lineLimit(1)
                    .padding(.horizontal, MFSpacing.s)
                    .padding(.bottom, MFSpacing.xs)
            }
            if !collapsed {
                ForEach(section.mailboxes) { mailbox in mailboxRow(mailbox) }
            }
        }
    }

    private func accountSubtitle(_ section: MailInboxModel.AccountSection, collapsed: Bool) -> String? {
        let count = section.mailboxes.count
        let mailboxes = count == 1 ? "1 mailbox" : "\(count) mailboxes"
        switch (section.account.subtitle, collapsed) {
        case (let address?, true): return "\(address) · \(mailboxes)"
        case (let address?, false): return address
        case (nil, true): return mailboxes
        case (nil, false): return nil
        }
    }

    private func mailboxRow(_ mailbox: Mailbox) -> some View {
        folderRow(mailbox.name, icon: mailbox.kind == .other && mailbox.isLocal && mailbox.name == "Send Later"
                      ? "clock" : icon(for: mailbox.kind),
                  folder: .mailbox(mailbox.ref.url))
    }

    private func icon(for kind: MailboxRef.Kind) -> String {
        switch kind {
        case .inbox: "tray"
        case .important: "tag"
        case .drafts: "doc"
        case .sent: "paperplane"
        case .junk: "xmark.bin"
        case .trash: "trash"
        case .archive: "archivebox"
        case .other: "folder"
        }
    }

    private func folderRow(_ title: String, icon: String, folder: MailInboxModel.Folder) -> some View {
        let selected = model.folder == folder
        let count = model.count(in: folder)
        return Button { Task { await model.select(folder) } } label: {
            HStack(spacing: MFSpacing.s) {
                Image(systemName: icon)
                    .font(MFFont.callout)
                    .foregroundStyle(selected ? MFColor.ink : MFColor.tertiary)
                    .frame(width: MailMetrics.iconWidth)
                    .accessibilityHidden(true)
                Text(title)
                    .font(selected ? MFFont.calloutEmphasis : MFFont.callout)
                    .foregroundStyle(MFColor.ink)
                    .lineLimit(1)
                Spacer(minLength: MFSpacing.xs)
                if count > 0 {
                    Text(count.formatted())
                        .font(MFFont.caption)
                        .foregroundStyle(selected ? MFColor.ink : MFColor.secondary)
                }
            }
            .padding(.horizontal, MFSpacing.s + MFSpacing.xxs)
            .padding(.vertical, MFSpacing.xs + MFSpacing.xxs)
            .background(RoundedRectangle(cornerRadius: MFRadius.s).fill(selected ? MFColor.sidebarSelected : .clear))
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityLabel(count > 0 ? "\(title), \(count)" : title)
        .accessibilityAddTraits(selected ? .isSelected : [])
    }
}

/// The bottom of the sidebar: whether new mail is still coming in.
struct MailSyncNote: View {
    var model: MailInboxModel

    private var closed: Bool { model.syncState == .mailClosed }

    var body: some View {
        HStack(alignment: .top, spacing: MFSpacing.s) {
            Circle()
                .fill(closed ? MFColor.preview : MFColor.success)
                .frame(width: MailMetrics.dot, height: MailMetrics.dot)
                .padding(.top, MFSpacing.xs + MFSpacing.xxs)
                .accessibilityHidden(true)
            VStack(alignment: .leading, spacing: MFSpacing.xs) {
                Text(closed ? "Apple Mail is closed" : "Up to date with Apple Mail")
                    .font(MFFont.calloutSemibold)
                    .foregroundStyle(MFColor.ink)
                Text(detail)
                    .font(MFFont.caption)
                    .foregroundStyle(MFColor.secondary)
                    .lineSpacing(MFLineSpacing.tight)
                    .fixedSize(horizontal: false, vertical: true)
                if closed {
                    Button("Open Mail") { model.openMail() }
                        .buttonStyle(.appleSecondary)
                        .padding(.top, MFSpacing.xxs)
                }
            }
            Spacer(minLength: 0)
        }
        .padding(MFSpacing.m)
        .background(RoundedRectangle(cornerRadius: MFRadius.m).fill(closed ? MFColor.approvalFill : MFColor.surface))
        .overlay(RoundedRectangle(cornerRadius: MFRadius.m)
            .strokeBorder(closed ? MFColor.approvalBorder : MFColor.hairline))
        .accessibilityElement(children: .contain)
    }

    private var detail: String {
        guard closed else { return "New mail shows here a few seconds after Mail gets it." }
        let time = model.lastRefreshed.map { " as of \($0.formatted(date: .omitted, time: .shortened))" } ?? ""
        return "Showing mail\(time). New mail arrives when Mail is open."
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
                if model.unreadInFolder > 0 {
                    Text("\(model.unreadInFolder.formatted()) unread")
                        .font(MFFont.chip)
                        .foregroundStyle(MFColor.secondary)
                        .padding(.horizontal, MFSpacing.s)
                        .padding(.vertical, MFSpacing.xxs)
                        .background(Capsule().fill(MFColor.fill))
                        .lineLimit(1)
                }
                Spacer(minLength: MFSpacing.xs)
                Text(model.folderScope)
                    .font(MFFont.caption)
                    .foregroundStyle(MFColor.secondary)
                    .lineLimit(1)
            }
            .padding(.horizontal, MFSpacing.l + MFSpacing.xs)
            .padding(.vertical, MFSpacing.l)
            Rectangle().fill(MFColor.hairlineFaint).frame(height: 1)
            MFScroll {
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
            shortcut("⌘J", "Draft") { Task { await model.writeDraft() } }
                .keyboardShortcut("j", modifiers: .command)
                // Not while the compose card is up: ⌘J drafts a reply to the
                // selected message, not the new one.
                .disabled(model.composer.isOpen && !model.composer.isMinimized)
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
            HStack(alignment: .firstTextBaseline, spacing: MFSpacing.xs) {
                Group {
                    if message.isOutgoing {
                        Text("\(Text("To:").foregroundStyle(MFColor.secondary).font(MFFont.callout)) \(message.counterpart)")
                    } else {
                        Text(message.counterpart)
                    }
                }
                .font(message.isRead ? MFFont.bodySmallEmphasis : MFFont.rowTitle)
                .foregroundStyle(MFColor.ink)
                .lineLimit(1)
                Spacer(minLength: MFSpacing.xs)
                if message.isFlagged {
                    Image(systemName: "flag.fill")
                        .font(MFFont.caption)
                        .foregroundStyle(MFColor.preview)
                        .accessibilityLabel("Flagged")
                }
                Text(Self.timeLabel(message.date))
                    .font(MFFont.caption)
                    .foregroundStyle(MFColor.secondary)
            }
            Text(message.subject)
                .font(message.isRead ? MFFont.callout : MFFont.calloutEmphasis)
                .foregroundStyle(MFColor.ink)
                .lineLimit(1)
            Text(message.snippet)
                .font(MFFont.preview)
                .foregroundStyle(MFColor.secondary)
                .lineSpacing(MFLineSpacing.tight)
                .lineLimit(2)
        }
        .padding(.horizontal, MFSpacing.l + MFSpacing.xs)
        .padding(.vertical, MFSpacing.m + MFSpacing.xxs)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(selected ? MFColor.accentFillFaint : .clear)
        .overlay(alignment: .topLeading) {
            if !message.isRead {
                Circle().fill(MFColor.accent)
                    .frame(width: MailMetrics.dot, height: MailMetrics.dot)
                    .padding(.leading, MFSpacing.s)
                    .padding(.top, MFSpacing.m + MFSpacing.s)
                    .accessibilityLabel("Unread")
            }
        }
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
            MFScroll {
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
                            .lineSpacing(MFLineSpacing.relaxed)
                            .textSelection(.enabled)
                            .fixedSize(horizontal: false, vertical: true)
                    } else {
                        ProgressView().controlSize(.small)
                    }
                    if let card = model.eventCard { eventCard(card) }
                    if model.canReply { DraftCard(model: model) }
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
            Text("\(Text(from(message)).font(MFFont.bodyEmphasis).foregroundStyle(MFColor.ink)) \(Text("to \(to(message)) · \(MailRow.timeLabel(message.date)) · \(model.location(of: message))").foregroundStyle(MFColor.secondary))")
                .font(MFFont.body)
            if let trackers = model.body?.trackersRemoved, trackers > 0 {
                AppleChip(text: trackers == 1 ? "1 tracker removed" : "\(trackers) trackers removed", tone: .success)
            }
            if let blocked = model.body?.remoteImagesBlocked, blocked > 0 {
                AppleChip(text: blocked == 1 ? "1 remote image not loaded" : "\(blocked) remote images not loaded")
            }
        }
    }

    private func from(_ message: MailMessageSummary) -> String {
        message.isOutgoing ? "You" : message.senderDisplay
    }

    private func to(_ message: MailMessageSummary) -> String {
        message.isOutgoing && !message.recipients.isEmpty ? message.counterpart : "you"
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
    private var editing: Bool { model.isEditingDraft }

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
            if let reply = model.replyDraft {
                VStack(alignment: .leading, spacing: MFSpacing.xxs) {
                    addressLine("To", reply.to)
                    if let from = reply.from { addressLine("From", from) }
                    addressLine("Subject", reply.subject)
                }
                .padding(.horizontal, MFSpacing.l + MFSpacing.xs)
                .padding(.top, MFSpacing.m)
            }
            content
                .padding(.horizontal, MFSpacing.l + MFSpacing.xs)
                .padding(.vertical, MFSpacing.l)
            Rectangle().fill(MFColor.hairline).frame(height: 1)
            HStack(spacing: MFSpacing.m) {
                Button("Send") {
                    model.isEditingDraft = false
                    Task { await model.send() }
                }
                    .buttonStyle(.applePrimary)
                    .disabled(!model.canSend)
                Button(editing ? "Done" : "Edit") { model.isEditingDraft.toggle() }
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
                    .lineSpacing(MFLineSpacing.normal)
                    .frame(maxWidth: .infinity, minHeight: 100, alignment: .topLeading)
                    .textSelection(.enabled)
            }
        }
    }

    private func addressLine(_ label: String, _ value: String) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: MFSpacing.s) {
            Text(label)
                .font(MFFont.caption)
                .foregroundStyle(MFColor.tertiary)
                .frame(width: MFSpacing.xxl * 2, alignment: .leading)
            Text(value)
                .font(MFFont.callout)
                .foregroundStyle(MFColor.ink)
                .lineLimit(1)
                .textSelection(.enabled)
        }
        .accessibilityElement(children: .combine)
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

/// Compose (C) in the shell toolbar. The shell keeps a toolbar item by id,
/// so this view reads the model live in its own body.
struct ComposeToolbarButton: View {
    @Bindable var model: MailInboxModel

    private var isMailAvailable: Bool {
        if case .available = model.status { return true }
        return false
    }

    var body: some View {
        Button { model.composer.open() } label: {
            HStack(spacing: MFSpacing.s) {
                Text("Compose")
                Text("C")
                    .font(MFFont.caption)
                    .padding(.horizontal, MFSpacing.xs + MFSpacing.xxs)
                    .padding(.vertical, MFSpacing.xxs)
                    .background(RoundedRectangle(cornerRadius: MFRadius.xs).fill(MFColor.onInk.opacity(0.18)))
                    .accessibilityHidden(true)
            }
        }
        .buttonStyle(.applePrimary)
        .keyboardShortcut("c", modifiers: [])
        // Off while the card is expanded (C types a letter there); a
        // minimized card comes back with C.
        .disabled(model.isEditingText || !isMailAvailable)
        .accessibilityLabel(model.composer.isMinimized ? "Show the message being written" : "Compose a new message")
    }
}

/// Full Disk Access is missing: the reason and System Settings first; once
/// the pane was opened and access still fails, Quit and Reopen (macOS applies
/// the grant only from the next launch).
struct MailAccessState: View {
    var model: MailInboxModel

    var body: some View {
        VStack(spacing: MFSpacing.m) {
            Image(systemName: "lock.shield")
                .font(.system(size: 30, weight: .regular))
                .foregroundStyle(MFColor.tertiary)
                .accessibilityHidden(true)
            Text("Read Apple Mail on this Mac")
                .font(MFFont.headline)
                .foregroundStyle(MFColor.ink)
            Text(message)
                .font(MFFont.body)
                .foregroundStyle(MFColor.secondary)
                .lineSpacing(MFLineSpacing.normal)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 440)
            HStack(spacing: MFSpacing.s) {
                if model.needsRelaunchForAccess {
                    Button("Quit and Reopen") { AppRelaunch.quitAndReopen() }
                        .buttonStyle(.applePrimary)
                    Button("Open System Settings") { Task { await model.requestAccess() } }
                        .buttonStyle(.appleSecondary)
                } else {
                    Button("Open System Settings") { Task { await model.requestAccess() } }
                        .buttonStyle(.applePrimary)
                }
                Button("Check again") { Task { await model.recheckAccess() } }
                    .buttonStyle(.appleSecondary)
            }
            .padding(.top, MFSpacing.xs)
        }
        .padding(MFSpacing.xxl)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .accessibilityElement(children: .contain)
    }

    private var message: String {
        if model.needsRelaunchForAccess {
            return "If Mference is already on in Full Disk Access, quit and reopen Mference so macOS applies it."
        }
        return AppPermission.fullDiskAccess.reason + " Turn on Mference under Privacy & Security > Full Disk Access."
    }
}

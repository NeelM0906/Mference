import MferenceAppCore
import MferenceDesign
import SwiftUI

/// The conversation list: New chat, Today / earlier groups, Private, and the
/// Memory card. (The mockup's "Skills" section has no Phase A data source and
/// is left out.)
struct ChatSidebar: View {
    @Bindable var model: ChatViewModel
    let actions: ChatActions

    var body: some View {
        VStack(alignment: .leading, spacing: MFSpacing.l) {
            HStack(spacing: MFSpacing.s) {
                newChatButton
                newPrivateChatButton
            }
            ScrollView {
                VStack(alignment: .leading, spacing: MFSpacing.l) {
                    if model.groups.groups.isEmpty {
                        Text("Your chats appear here. They stay on this Mac.")
                            .font(MFFont.callout)
                            .foregroundStyle(MFColor.tertiary)
                            .padding(.horizontal, MFSpacing.s)
                    }
                    ForEach(model.groups.groups) { group in
                        section(group)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .scrollIndicators(.never)
            memoryCard
        }
        .padding(.horizontal, MFSpacing.m)
        .padding(.vertical, MFSpacing.l)
        .background(MFColor.canvas)
        .accessibilityElement(children: .contain)
        .accessibilityLabel("Chats")
    }

    private var newChatButton: some View {
        Button {
            model.newChat()
        } label: {
            HStack(spacing: MFSpacing.s) {
                Text("New chat")
                    .font(MFFont.calloutEmphasis)
                Text("⌘N")
                    .font(MFFont.keycap)
                    .frame(minWidth: ChatMetrics.keycap, minHeight: ChatMetrics.keycap)
                    .padding(.horizontal, MFSpacing.xs)
                    .background(MFColor.onInk.opacity(0.12), in: RoundedRectangle(cornerRadius: MFRadius.xs))
                    .accessibilityHidden(true)
            }
            .foregroundStyle(MFColor.onInk)
            .frame(maxWidth: .infinity)
            .padding(.vertical, MFSpacing.s + MFSpacing.xxs)
            .background(MFColor.ink, in: Capsule())
            .contentShape(Capsule())
        }
        .buttonStyle(.plain)
        .keyboardShortcut("n", modifiers: .command)
        .accessibilityLabel("New chat")
        .accessibilityHint("Command N")
    }

    /// A private chat: never used for memory, listed under Private.
    private var newPrivateChatButton: some View {
        Button {
            model.newChat(isPrivate: true)
        } label: {
            Image(systemName: "lock")
                .font(MFFont.calloutEmphasis)
                .foregroundStyle(MFColor.ink)
                .frame(width: ChatMetrics.sidebarButton, height: ChatMetrics.sidebarButton)
                .background(MFColor.fill, in: Circle())
                .contentShape(Circle())
        }
        .buttonStyle(.plain)
        .keyboardShortcut("n", modifiers: [.command, .shift])
        .help("New private chat (⇧⌘N)")
        .accessibilityLabel("New private chat")
        .accessibilityHint("Shift Command N. Private chats are never used for memory")
    }

    private func section(_ group: ConversationGroups.Group) -> some View {
        VStack(alignment: .leading, spacing: MFSpacing.xxs) {
            Text(group.title.uppercased())
                .font(MFFont.sectionHeader)
                .tracking(MFFont.sectionHeaderTracking)
                .foregroundStyle(MFColor.tertiary)
                .padding(.horizontal, MFSpacing.s)
                .padding(.bottom, MFSpacing.xs)
                .accessibilityAddTraits(.isHeader)
            ForEach(group.rows) { row in
                ConversationRow(row: row, isSelected: row.id == model.selectedID) {
                    model.select(row.id)
                } onDelete: {
                    Task { await model.delete(row.id) }
                }
            }
        }
    }

    private var memoryCard: some View {
        VStack(alignment: .leading, spacing: MFSpacing.xxs) {
            Text("Memory")
                .font(MFFont.caption.weight(.semibold))
                .foregroundStyle(MFColor.ink)
            Text(memoryLine)
                .font(MFFont.caption)
                .foregroundStyle(MFColor.secondary)
                .fixedSize(horizontal: false, vertical: true)
            Button("Review") { actions.openMemorySettings() }
                .buttonStyle(.plain)
                .font(MFFont.caption.weight(.medium))
                .foregroundStyle(MFColor.accent)
                .padding(.top, MFSpacing.xxs)
                .accessibilityLabel("Review memory in Settings")
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(MFSpacing.m)
        .background(MFColor.surface, in: RoundedRectangle(cornerRadius: MFRadius.m))
        .overlay(RoundedRectangle(cornerRadius: MFRadius.m).strokeBorder(MFColor.hairlineFaint))
        .accessibilityElement(children: .contain)
    }

    private var memoryLine: String {
        switch model.rememberedCount {
        case 0: "Nothing remembered yet. What it learns stays on this Mac."
        case 1: "1 thing remembered, all on this Mac."
        case let n: "\(n) things remembered, all on this Mac."
        }
    }
}

struct ConversationRow: View {
    let row: ConversationGroups.Row
    let isSelected: Bool
    let onSelect: () -> Void
    let onDelete: () -> Void
    @State private var isHovered = false

    var body: some View {
        Button(action: onSelect) {
            HStack(spacing: MFSpacing.s) {
                if row.isPrivate {
                    Image(systemName: "lock")
                        .imageScale(.small)
                        .accessibilityHidden(true)
                }
                Text(row.title)
                    .lineLimit(1)
                    .truncationMode(.tail)
                Spacer(minLength: 0)
            }
            .font(isSelected ? MFFont.calloutEmphasis : MFFont.callout)
            .foregroundStyle(MFColor.ink)
            .padding(.horizontal, MFSpacing.s + MFSpacing.xxs)
            .padding(.vertical, MFSpacing.s)
            .background(isSelected ? MFColor.fill : (isHovered ? MFColor.fillSubtle : .clear),
                        in: RoundedRectangle(cornerRadius: MFRadius.s))
            .contentShape(RoundedRectangle(cornerRadius: MFRadius.s))
        }
        .buttonStyle(.plain)
        .onHover { isHovered = $0 }
        .contextMenu {
            Button("Delete chat", role: .destructive, action: onDelete)
        }
        .accessibilityLabel(row.isPrivate ? "\(row.title), private" : row.title)
        .accessibilityAddTraits(isSelected ? .isSelected : [])
    }
}

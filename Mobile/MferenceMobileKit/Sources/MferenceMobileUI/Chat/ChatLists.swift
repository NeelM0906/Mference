import MferenceDesign
import SwiftUI

/// The model picker sheet: On this iPhone, then On your Mac.
struct ModelPickerSheet: View {
    @Bindable var model: MobileAppModel
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: MFSpacing.xs) {
                HStack {
                    Text("Model for this chat").font(MFPhoneFont.headline).foregroundStyle(MFColor.ink)
                    Spacer()
                    Button("Done") { dismiss() }
                        .buttonStyle(PhoneButtonStyle(kind: .quiet, size: .compact))
                }
                .padding(.bottom, MFSpacing.s)

                PhoneSectionHeader("On this iPhone")
                pickerRow(.quick, badge: "Loaded", tone: .success)
                if model.appleModelEnabled {
                    pickerRow(.apple, badge: "Built in", tone: .neutral)
                }

                Divider().overlay(MFColor.hairlineFaint).padding(.vertical, MFSpacing.s)

                HStack {
                    PhoneSectionHeader("On your Mac")
                    if let mac = model.pairedMac {
                        PhoneChip(title: "\(mac.name) · same Wi-Fi", systemImage: "laptopcomputer", tone: .success)
                    }
                }
                if model.macModels.isEmpty {
                    Text("Pair your Mac in Settings to use Deep and the other Mac models. Your Mac must be on the same Wi-Fi.")
                        .font(MFPhoneFont.preview)
                        .foregroundStyle(MFColor.secondary)
                        .padding(MFSpacing.m)
                } else {
                    ForEach(model.macModels) { macModel in
                        pickerRow(macModel, badge: "Ready", tone: .neutral)
                    }
                }

                HStack(spacing: MFSpacing.s) {
                    VStack(alignment: .leading, spacing: 2) {
                        HStack(spacing: MFSpacing.snug) {
                            Text("Auto").font(MFPhoneFont.rowTitle).foregroundStyle(MFColor.ink)
                            PhoneChip(title: "Later", tone: .preview)
                        }
                        Text("For now you choose the model yourself.")
                            .font(MFPhoneFont.preview).foregroundStyle(MFColor.secondary)
                    }
                    Spacer()
                }
                .padding(MFSpacing.m)
                .background(RoundedRectangle(cornerRadius: MFRadius.l).fill(MFColor.sidebar))
                .padding(.top, MFSpacing.m)
            }
            .padding(.horizontal, MFSpacing.l)
            .padding(.top, MFSpacing.xl)
        }
        .background(MFColor.surface)
    }

    private func pickerRow(_ option: PhoneModel, badge: String, tone: MFTone) -> some View {
        let isSelected = option == model.selectedModel
        return Button {
            model.selectedModel = option
            dismiss()
        } label: {
            HStack(alignment: .top, spacing: MFSpacing.m) {
                ZStack {
                    Circle().strokeBorder(isSelected ? MFColor.accent : MFColor.controlBorder, lineWidth: 1.5)
                    if isSelected { Circle().fill(MFColor.accent).frame(width: 10, height: 10) }
                }
                .frame(width: MFPhoneMetrics.radio, height: MFPhoneMetrics.radio)
                .padding(.top, 1)
                VStack(alignment: .leading, spacing: 3) {
                    HStack(alignment: .firstTextBaseline, spacing: MFSpacing.snug) {
                        Text(option.name).font(MFPhoneFont.rowTitle).foregroundStyle(MFColor.ink)
                        if option.isPreview {
                            Text("Preview").font(MFPhoneFont.control).foregroundStyle(MFColor.preview)
                        } else if !option.detail.isEmpty {
                            Text(option.detail).font(MFPhoneFont.control.weight(.regular)).foregroundStyle(MFColor.secondary)
                        }
                    }
                    Text(option.summary).font(MFPhoneFont.preview).foregroundStyle(MFColor.secondary)
                        .multilineTextAlignment(.leading)
                        .fixedSize(horizontal: false, vertical: true)
                }
                Spacer(minLength: 0)
                PhoneChip(title: badge, tone: tone)
            }
            .padding(MFSpacing.m)
            .background(RoundedRectangle(cornerRadius: MFRadius.l).fill(isSelected ? MFColor.accentFillFaint : .clear))
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityAddTraits(isSelected ? .isSelected : [])
    }
}

/// The chat history: groups, search and the memory card. A pushed screen on
/// iPhone; the left pane on Duo open.
struct ChatsListContent: View {
    @Bindable var model: MobileAppModel
    /// Called after a conversation or New chat is chosen (pops on iPhone).
    var onOpen: () -> Void = {}
    /// Duo open draws rows as sidebar rows instead of cards.
    var style: Style = .cards

    enum Style { case cards, sidebar }

    var body: some View {
        VStack(alignment: .leading, spacing: MFSpacing.m) {
            if style == .sidebar {
                HStack(spacing: MFSpacing.s) {
                    Button {
                        model.startNewChat()
                        onOpen()
                    } label: {
                        Label("New chat", systemImage: "square.and.pencil")
                    }
                    .buttonStyle(PhoneButtonStyle(kind: .primary))
                    HStack(spacing: MFSpacing.s) {
                        Image(systemName: "magnifyingglass").foregroundStyle(MFColor.tertiary)
                        TextField("Search", text: $model.searchText)
                            .font(MFPhoneFont.callout)
                    }
                    .padding(.horizontal, 14)
                    .frame(height: MFPhoneMetrics.buttonHeight)
                    .background(Capsule().fill(MFColor.surface))
                    .overlay(Capsule().strokeBorder(MFColor.hairline, lineWidth: MFLayout.hairline))
                }
            }
            ForEach(Conversation.ListGroup.allCases, id: \.self) { group in
                let items = model.conversations(in: group)
                if !items.isEmpty {
                    PhoneSectionHeader(group.rawValue)
                        .padding(.top, MFSpacing.xs)
                    if style == .cards {
                        PhoneCard {
                            ForEach(items) { conversation in cardRow(conversation) }
                        }
                    } else {
                        VStack(spacing: 2) {
                            ForEach(items) { conversation in sidebarRow(conversation) }
                        }
                    }
                }
            }
            memoryCard
                .padding(.top, MFSpacing.s)
        }
    }

    private func cardRow(_ conversation: Conversation) -> some View {
        Button {
            model.open(conversation)
            onOpen()
        } label: {
            VStack(alignment: .leading, spacing: 2) {
                HStack {
                    if conversation.group == .privateChats {
                        Image(systemName: "lock").font(.system(size: 13)).foregroundStyle(MFColor.secondary)
                    }
                    Text(conversation.title).font(MFPhoneFont.rowTitle).foregroundStyle(MFColor.ink)
                    Spacer()
                    if conversation.isRunningOnMac {
                        PhoneChip(title: "On your Mac", tone: .accent)
                    } else {
                        Text(conversation.updatedLabel).font(MFPhoneFont.caption).foregroundStyle(MFColor.tertiary)
                    }
                }
                if !conversation.preview.isEmpty {
                    Text(conversation.preview).font(MFPhoneFont.preview).foregroundStyle(MFColor.secondary)
                        .lineLimit(1)
                }
            }
            .padding(.horizontal, MFSpacing.l)
            .padding(.vertical, 11)
            .frame(minHeight: MFPhoneMetrics.rowHeight)
            .background(conversation.id == model.currentConversationID ? MFColor.sidebar : .clear)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }

    private func sidebarRow(_ conversation: Conversation) -> some View {
        Button {
            model.open(conversation)
            onOpen()
        } label: {
            HStack(spacing: MFSpacing.s) {
                if conversation.group == .privateChats {
                    Image(systemName: "lock").font(.system(size: 13)).foregroundStyle(MFColor.secondary)
                }
                Text(conversation.title)
                    .font(conversation.id == model.currentConversationID ? MFPhoneFont.button : MFPhoneFont.callout)
                    .foregroundStyle(MFColor.ink)
                    .lineLimit(1)
                Spacer()
                if conversation.isRunningOnMac {
                    PhoneChip(title: "On your Mac", tone: .accent)
                }
            }
            .padding(.horizontal, MFSpacing.m)
            .frame(minHeight: MFPhoneMetrics.touchTarget)
            .background(RoundedRectangle(cornerRadius: MFRadius.m)
                .fill(conversation.id == model.currentConversationID ? MFColor.sidebarSelected : .clear))
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }

    private var memoryCard: some View {
        HStack(spacing: MFSpacing.m) {
            VStack(alignment: .leading, spacing: 2) {
                Text("Memory").font(MFPhoneFont.rowTitle).foregroundStyle(MFColor.ink)
                Text("\(model.memoryCount) things remembered, all on this iPhone.")
                    .font(MFPhoneFont.preview).foregroundStyle(MFColor.secondary)
            }
            Spacer()
            Button("Review") { model.selection = .settings }
                .buttonStyle(PhoneButtonStyle(kind: .secondary, size: .compact))
        }
        .padding(.horizontal, MFSpacing.l)
        .padding(.vertical, MFSpacing.m)
        .background(RoundedRectangle(cornerRadius: MFRadius.l).fill(MFColor.surface))
        .overlay(RoundedRectangle(cornerRadius: MFRadius.l).strokeBorder(MFColor.hairline, lineWidth: MFLayout.hairline))
    }
}

/// Chat on iPhone and Duo closed: the conversation, with Chats pushed from
/// the leading button and New chat on the trailing one.
struct ChatScreen: View {
    @Bindable var model: MobileAppModel
    @State private var showsChats = false

    var body: some View {
        NavigationStack {
            ChatConversationView(model: model)
                .navigationTitle(model.chatTitle)
                .navigationBarTitleDisplayMode(.inline)
                .toolbar {
                    ToolbarItem(placement: .topBarLeading) {
                        Button { showsChats = true } label: { Image(systemName: "list.bullet") }
                            .accessibilityLabel("Chats")
                    }
                    ToolbarItem(placement: .topBarTrailing) {
                        Button { model.startNewChat() } label: { Image(systemName: "square.and.pencil") }
                            .accessibilityLabel("New chat")
                    }
                }
                .navigationDestination(isPresented: $showsChats) {
                    ScrollView {
                        ChatsListContent(model: model) { showsChats = false }
                            .padding(.horizontal, MFPhoneMetrics.screenMargin)
                            .padding(.bottom, MFSpacing.xl)
                    }
                    .background(MFColor.canvas)
                    .navigationTitle("Chats")
                    .navigationBarTitleDisplayMode(.large)
                    .searchable(text: $model.searchText, prompt: "Search chats")
                    .toolbar {
                        ToolbarItem(placement: .topBarTrailing) {
                            Button {
                                model.startNewChat()
                                showsChats = false
                            } label: {
                                Label("New chat", systemImage: "square.and.pencil").labelStyle(.titleAndIcon)
                            }
                            .buttonStyle(.borderedProminent)
                            .tint(MFColor.ink)
                        }
                    }
                }
        }
    }
}

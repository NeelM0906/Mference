import MferenceDesign
import SwiftUI

/// The conversation and its composer. Used full screen on iPhone and Duo
/// closed, and as the right pane on Duo open.
struct ChatConversationView: View {
    @Bindable var model: MobileAppModel
    var margin: CGFloat = MFPhoneMetrics.screenMargin

    var body: some View {
        ScrollView {
            if model.currentMessages.isEmpty {
                EmptyChatView(model: model)
                    .padding(.horizontal, margin)
            } else {
                LazyVStack(alignment: .leading, spacing: MFSpacing.m) {
                    ForEach(model.currentMessages) { message in
                        MessageView(message: message, model: model)
                            .id(message.id)
                    }
                }
                .padding(.horizontal, margin)
                .padding(.top, MFSpacing.s)
                .padding(.bottom, MFSpacing.l)
            }
        }
        .defaultScrollAnchor(.bottom)
        .scrollDismissesKeyboard(.interactively)
        .safeAreaInset(edge: .bottom, spacing: 0) {
            Composer(model: model)
                .padding(.horizontal, margin - 4)
                .padding(.bottom, MFSpacing.s)
        }
        .background(MFColor.canvas)
        .sheet(isPresented: $model.isModelPickerPresented) {
            ModelPickerSheet(model: model)
                .presentationDetents([.medium, .large])
                .presentationDragIndicator(.visible)
        }
    }
}

/// A new chat: a short greeting and four starters, as on the Mac.
struct EmptyChatView: View {
    @Bindable var model: MobileAppModel

    private let starters = ["Answer from the web", "Plan my week", "What did Priya ask for?", "Compare standing desk warranties"]

    var body: some View {
        VStack(alignment: .leading, spacing: MFSpacing.l) {
            Spacer(minLength: 80)
            Text("What can I help with?")
                .font(MFPhoneFont.largeTitle)
                .foregroundStyle(MFColor.ink)
            Text("Answers run on this iPhone. Bigger jobs can go to your Mac when you say so.")
                .font(MFPhoneFont.callout)
                .foregroundStyle(MFColor.secondary)
            FlowLayout(spacing: MFSpacing.s) {
                ForEach(starters, id: \.self) { starter in
                    Button(starter) { model.send(starter) }
                        .buttonStyle(PhoneButtonStyle(kind: .secondary, size: .compact))
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

/// One message: the user's grey bubble, or an answer on the page.
struct MessageView: View {
    let message: ChatMessage
    @Bindable var model: MobileAppModel
    @State private var showsSteps = false

    var body: some View {
        switch message.role {
        case .user:
            Text(message.text)
                .font(MFPhoneFont.body)
                .foregroundStyle(MFColor.ink)
                .lineSpacing(MFLineSpacing.tight)
                .padding(.horizontal, 14)
                .padding(.vertical, 11)
                .background(RoundedRectangle(cornerRadius: MFPhoneMetrics.bubbleRadius).fill(MFColor.fill))
                .frame(maxWidth: MFPhoneMetrics.bubbleMaxWidth, alignment: .trailing)
                .frame(maxWidth: .infinity, alignment: .trailing)
        case .assistant:
            answer
        }
    }

    @ViewBuilder private var answer: some View {
        VStack(alignment: .leading, spacing: MFSpacing.m) {
            if let handoff = message.handoff {
                HandoffCard(offer: handoff, macName: model.pairedMac?.name ?? "your Mac",
                            modelName: model.pairedMac?.models.first?.shortName ?? "Deep 27B") { sendToMac in
                    model.resolveHandoff(message.id, sendToMac: sendToMac)
                }
            }
            if let steps = message.webSteps {
                Button {
                    withAnimation(.snappy(duration: 0.2)) { showsSteps.toggle() }
                } label: {
                    HStack(spacing: MFSpacing.snug) {
                        Image(systemName: "globe").accessibilityHidden(true)
                        Text(steps)
                        Image(systemName: showsSteps ? "chevron.up" : "chevron.down")
                            .font(.system(size: 11, weight: .semibold)).accessibilityHidden(true)
                    }
                    .font(MFPhoneFont.control)
                    .foregroundStyle(MFColor.secondary)
                    .padding(.horizontal, MFSpacing.m)
                    .frame(minHeight: 32)
                    .background(Capsule().fill(MFColor.fill))
                }
                .buttonStyle(.plain)
                if showsSteps {
                    Text("Only these search queries left your iPhone, through SearXNG on your paired Mac. The model and everything it read stay on your devices.")
                        .font(MFPhoneFont.caption)
                        .foregroundStyle(MFColor.secondary)
                        .padding(MFSpacing.m)
                        .background(RoundedRectangle(cornerRadius: MFRadius.l).fill(MFColor.sidebar))
                }
            }
            if !message.text.isEmpty || message.isStreaming {
                CitedText(text: message.text + (message.isStreaming ? " ▍" : ""))
            }
            if let table = message.table {
                AnswerTableView(table: table)
                    .padding(.trailing, -MFPhoneMetrics.screenMargin)
            }
            if let closing = message.closing {
                CitedText(text: closing)
            }
            if !message.sources.isEmpty {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: MFSpacing.s) {
                        ForEach(Array(message.sources.enumerated()), id: \.offset) { index, source in
                            SourceChip(number: index + 1, title: source)
                        }
                    }
                }
                .scrollClipDisabled()
            }
            if !message.actions.isEmpty {
                FlowLayout(spacing: MFSpacing.s) {
                    ForEach(message.actions, id: \.self) { action in
                        Button(action) { model.send(action) }
                            .buttonStyle(PhoneButtonStyle(kind: .secondary, size: .compact))
                    }
                }
            }
            if let label = message.engineLabel, !message.isStreaming {
                Text(label)
                    .font(MFPhoneFont.caption)
                    .foregroundStyle(MFColor.tertiary)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

/// A numbered source under an answer.
struct SourceChip: View {
    let number: Int
    let title: String

    var body: some View {
        HStack(spacing: MFSpacing.snug) {
            Text("\(number)").font(MFPhoneFont.previewEmphasis).foregroundStyle(MFColor.accent)
            Text(title).font(MFPhoneFont.preview).foregroundStyle(MFColor.ink).lineLimit(1)
        }
        .padding(.horizontal, MFSpacing.m)
        .frame(height: MFPhoneMetrics.sourceChipHeight)
        .background(Capsule().fill(MFColor.surface))
        .overlay(Capsule().strokeBorder(MFColor.hairline, lineWidth: MFLayout.hairline))
        .accessibilityElement(children: .combine)
        .accessibilityLabel("Source \(number), \(title)")
    }
}

/// The bordered table card; scrolls sideways when wider than the screen.
struct AnswerTableView: View {
    let table: AnswerTable

    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            Grid(alignment: .leading, horizontalSpacing: 0, verticalSpacing: 0) {
                GridRow {
                    ForEach(table.headers, id: \.self) { header in
                        Text(header)
                            .font(MFFont.geist(15, .semibold, relativeTo: .subheadline))
                            .foregroundStyle(MFColor.ink)
                            .padding(.horizontal, MFSpacing.m)
                            .padding(.vertical, MFSpacing.compact)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                }
                .background(MFColor.sidebar)
                ForEach(Array(table.rows.enumerated()), id: \.offset) { _, row in
                    Divider().overlay(MFColor.hairlineFaint).gridCellUnsizedAxes(.horizontal)
                    GridRow {
                        ForEach(Array(row.enumerated()), id: \.offset) { _, cell in
                            CitedText(text: cell, font: MFPhoneFont.preview)
                                .lineLimit(1)
                                .fixedSize()
                                .padding(.horizontal, MFSpacing.m)
                                .padding(.vertical, 11)
                        }
                    }
                }
            }
            .background(MFColor.surface)
            .clipShape(RoundedRectangle(cornerRadius: MFRadius.l))
            .overlay(RoundedRectangle(cornerRadius: MFRadius.l).strokeBorder(MFColor.hairline, lineWidth: MFLayout.hairline))
            .padding(.trailing, MFPhoneMetrics.screenMargin)
        }
        .scrollClipDisabled()
    }
}

/// Lays children out in rows, wrapping when a row is full.
struct FlowLayout: Layout {
    var spacing: CGFloat = 8

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let width = proposal.width ?? .infinity
        var x: CGFloat = 0
        var y: CGFloat = 0
        var rowHeight: CGFloat = 0
        var maxX: CGFloat = 0
        for subview in subviews {
            let size = subview.sizeThatFits(.unspecified)
            if x > 0, x + size.width > width {
                x = 0
                y += rowHeight + spacing
                rowHeight = 0
            }
            x += size.width + spacing
            maxX = max(maxX, x - spacing)
            rowHeight = max(rowHeight, size.height)
        }
        return CGSize(width: proposal.width ?? maxX, height: y + rowHeight)
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        var x = bounds.minX
        var y = bounds.minY
        var rowHeight: CGFloat = 0
        for subview in subviews {
            let size = subview.sizeThatFits(.unspecified)
            if x > bounds.minX, x + size.width > bounds.maxX {
                x = bounds.minX
                y += rowHeight + spacing
                rowHeight = 0
            }
            subview.place(at: CGPoint(x: x, y: y), proposal: ProposedViewSize(size))
            x += size.width + spacing
            rowHeight = max(rowHeight, size.height)
        }
    }
}

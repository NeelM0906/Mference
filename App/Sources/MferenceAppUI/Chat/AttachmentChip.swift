import MferenceAppCore
import MferenceDesign
import SwiftUI

/// A file in the composer: its kind, name and state (reading, ready, or
/// why it can't be read), with a button to remove it.
struct AttachmentChip: View {
    let pending: PendingAttachment
    let onRemove: () -> Void

    var body: some View {
        HStack(spacing: MFSpacing.xs + 1) {
            icon
            VStack(alignment: .leading, spacing: 0) {
                Text(pending.filename)
                    .font(MFFont.chip)
                    .foregroundStyle(MFColor.ink)
                    .lineLimit(1)
                    .truncationMode(.middle)
                Text(detail)
                    .font(MFFont.preview)
                    .foregroundStyle(isFailed ? MFColor.danger : MFColor.tertiary)
                    .lineLimit(1)
            }
            Button(action: onRemove) {
                Image(systemName: "xmark")
                    .imageScale(.small)
                    .font(MFFont.chip)
                    .foregroundStyle(MFColor.secondary)
                    .frame(width: ChatMetrics.attachmentRemove, height: ChatMetrics.attachmentRemove)
                    .contentShape(Circle())
            }
            .buttonStyle(.plain)
            .accessibilityLabel("Remove \(pending.filename)")
        }
        .padding(.leading, MFSpacing.s)
        .padding(.trailing, MFSpacing.xs)
        .padding(.vertical, MFSpacing.xs)
        .frame(maxWidth: ChatMetrics.attachmentChipWidth, alignment: .leading)
        .background(MFColor.fill, in: RoundedRectangle(cornerRadius: MFRadius.m))
        .accessibilityElement(children: .combine)
        .accessibilityLabel("\(pending.filename), \(detail)")
    }

    @ViewBuilder
    private var icon: some View {
        switch pending.state {
        case .reading:
            ProgressView()
                .controlSize(.small)
                .frame(width: ChatMetrics.attachmentRemove, height: ChatMetrics.attachmentRemove)
        case .failed:
            Image(systemName: "exclamationmark.triangle")
                .foregroundStyle(MFColor.danger)
        case .ready:
            Image(systemName: pending.kind == .image ? "photo" : "doc.text")
                .foregroundStyle(MFColor.secondary)
        }
    }

    private var isFailed: Bool {
        if case .failed = pending.state { return true }
        return false
    }

    private var detail: String {
        switch pending.state {
        case .reading: "Reading on this Mac…"
        case .failed(let reason): reason
        case .ready(let attachment): attachment.detail
        }
    }
}

/// A file sent with a message, above its bubble. Clicking shows the start
/// of the text the model read from it.
struct SentAttachmentChip: View {
    let attachment: ChatAttachment
    @State private var showingText = false

    var body: some View {
        Button {
            showingText.toggle()
        } label: {
            HStack(spacing: MFSpacing.xs + 1) {
                Image(systemName: attachment.kind == .image ? "photo" : "doc.text")
                    .foregroundStyle(MFColor.secondary)
                Text(attachment.filename)
                    .font(MFFont.chip)
                    .foregroundStyle(MFColor.ink)
                    .lineLimit(1)
                    .truncationMode(.middle)
                Text(attachment.detail)
                    .font(MFFont.preview)
                    .foregroundStyle(MFColor.tertiary)
                    .lineLimit(1)
            }
            .padding(.horizontal, MFSpacing.s)
            .padding(.vertical, MFSpacing.xs + 1)
            .frame(maxWidth: ChatMetrics.attachmentChipWidth, alignment: .leading)
            .background(MFColor.surface, in: RoundedRectangle(cornerRadius: MFRadius.m))
            .overlay(RoundedRectangle(cornerRadius: MFRadius.m).strokeBorder(MFColor.hairline))
            .contentShape(RoundedRectangle(cornerRadius: MFRadius.m))
        }
        .buttonStyle(.plain)
        .popover(isPresented: $showingText, arrowEdge: .bottom) {
            ScrollView {
                Text(preview)
                    .font(MFFont.callout)
                    .foregroundStyle(attachment.hasText ? MFColor.ink : MFColor.tertiary)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(MFSpacing.m)
            }
            .frame(width: ChatMetrics.hoverCardWidth + ChatMetrics.attachmentChipWidth,
                   height: ChatMetrics.pickerMaxHeight)
        }
        .accessibilityLabel("\(attachment.filename), \(attachment.detail)")
        .accessibilityHint("Shows the text read from it")
    }

    private var preview: String {
        guard attachment.hasText else { return "No text was found in this image." }
        let start = attachment.text.prefix(ChatMetrics.attachmentPreviewCharacters)
        return start.count < attachment.text.count ? String(start) + "\n…" : String(start)
    }
}

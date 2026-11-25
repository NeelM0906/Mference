import AppKit
import MferenceAppCore
import MferenceDesign
import SwiftUI
import UniformTypeIdentifiers

/// Scan (mockup 07): drop, paste or open an image or PDF. Apple's text
/// recognition reads it on this Mac; the full text shows for any scan, and
/// what it is decides the rest: a receipt's fields (Add row, CSV), an event
/// for Calendar, a business card for Contacts, a table as CSV.
public struct ScanView: View {
    @Environment(ScanSession.self) private var environmentSession: ScanSession?
    private let injected: ScanSession?

    /// The shell's slot: the session comes from the environment.
    public init() { injected = nil }

    /// Snapshots and previews.
    public init(session: ScanSession) { injected = session }

    public var body: some View {
        if let session = injected ?? environmentSession {
            ScanContent(session: session)
        } else {
            AppleEmptyState(systemImage: "viewfinder", title: "Scan",
                            message: "Scanning is starting. Try again in a moment.")
        }
    }
}

struct ScanContent: View {
    @Bindable var session: ScanSession
    @State private var isDropTargeted = false
    @State private var exportMessage: String?
    /// The page shown, decoded once per file and page (bounded), not on
    /// every keystroke.
    @State private var preview: (url: URL, page: Int, image: CGImage)?

    var body: some View {
        HStack(spacing: 0) {
            imagePane
                .frame(width: ScanMetrics.imagePaneWidth)
                .clipped()
            Rectangle().fill(MFColor.hairline).frame(width: 1)
            detailPane
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        }
        .background(MFColor.surface)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .task(id: PreviewKey(url: session.imageURL, page: session.page)) { loadPreview() }
        // Cleared: no export or paste message is left on the empty screen.
        .onChange(of: session.phase) { _, phase in
            if phase == .empty { exportMessage = nil }
        }
        .shellToolbar(id: "scan.actions") { ScanToolbarActions(session: session) }
    }

    private func loadPreview() {
        guard let url = session.imageURL else { preview = nil; return }
        if preview?.url == url, preview?.page == session.page { return }
        // Decoded once per page and bounded to 2048 px, so it is cheap here.
        if let image = VisionScanner.loadImage(at: url, page: session.page, maximumPixelSize: 2_048) {
            preview = (url, session.page, image)
        }
    }

    // MARK: Image

    private var imagePane: some View {
        ZStack {
            StripedBackground()
            if let preview, preview.url == session.imageURL, preview.page == session.page {
                Image(decorative: preview.image, scale: 1)
                    .resizable()
                    .interpolation(.high)
                    .scaledToFit()
                    .overlay { highlightBoxes }
                    .shadow(color: MFColor.ink.opacity(0.12), radius: MFSpacing.m, y: MFSpacing.xs)
                    .padding(MFSpacing.xxl)
                    .accessibilityLabel("Scanned image")
            } else {
                VStack(spacing: MFSpacing.m) {
                    Image(systemName: "doc.viewfinder")
                        .font(MFFont.largeTitle)
                        .foregroundStyle(MFColor.tertiary)
                        .accessibilityHidden(true)
                    Text("Drop a document or photo here")
                        .font(MFFont.bodyEmphasis)
                        .foregroundStyle(MFColor.ink)
                    Text("Receipts, letters, flyers, business cards, tables. PNG, JPEG, HEIC or PDF, read on this Mac.")
                        .multilineTextAlignment(.center)
                        .font(MFFont.callout)
                        .foregroundStyle(MFColor.secondary)
                    Button("Open file", action: openFile)
                        .buttonStyle(.applePrimary)
                    Text("or paste an image with ⌘V")
                        .font(MFFont.caption)
                        .foregroundStyle(MFColor.tertiary)
                }
                .padding(MFSpacing.xl)
            }
            if session.pageCount > 1, session.imageURL != nil { pageControl }
            if isDropTargeted {
                RoundedRectangle(cornerRadius: MFRadius.l)
                    .strokeBorder(MFColor.accent, style: StrokeStyle(lineWidth: 2, dash: [6, 4]))
                    .padding(MFSpacing.l)
                    .accessibilityHidden(true)
            }
        }
        .onPasteCommand(of: [.fileURL, .png, .tiff, .image]) { providers in paste(providers) }
        .onDrop(of: [.fileURL], isTargeted: $isDropTargeted) { providers in
            guard let provider = providers.first else { return false }
            _ = provider.loadObject(ofClass: URL.self) { url, _ in
                guard let url else { return }
                Task { @MainActor in
                    if ScanSession.accepts(url) { session.open(url) }
                }
            }
            return true
        }
        .accessibilityElement(children: .contain)
        .accessibilityLabel("Drop area for documents and photos")
    }

    /// "Page 2 of 5" under a PDF's page.
    private var pageControl: some View {
        VStack {
            Spacer()
            HStack(spacing: MFSpacing.s) {
                Button { session.showPage(session.page - 1) } label: { Image(systemName: "chevron.left") }
                    .disabled(session.page == 0)
                    .accessibilityLabel("Previous page")
                Text("Page \(session.page + 1) of \(session.pageCount)")
                    .font(MFFont.callout)
                    .foregroundStyle(MFColor.ink)
                    .monospacedDigit()
                Button { session.showPage(session.page + 1) } label: { Image(systemName: "chevron.right") }
                    .disabled(session.page >= session.pageCount - 1)
                    .accessibilityLabel("Next page")
            }
            .buttonStyle(.appleQuiet)
            .padding(.horizontal, MFSpacing.s)
            .padding(.vertical, MFSpacing.xxs)
            .background(Capsule().fill(MFColor.surface))
            .overlay(Capsule().strokeBorder(MFColor.hairlineStrong))
            .padding(.bottom, MFSpacing.m)
        }
    }

    /// ⌘V: an image file's URL, or image data, from the clipboard.
    private func paste(_ providers: [NSItemProvider]) {
        guard let provider = providers.first else { return }
        if provider.hasItemConformingToTypeIdentifier(UTType.fileURL.identifier) {
            _ = provider.loadObject(ofClass: URL.self) { url, _ in
                guard let url else { return }
                Task { @MainActor in if ScanSession.accepts(url) { session.open(url) } }
            }
            return
        }
        for type in [UTType.png, .tiff, .image] where provider.hasItemConformingToTypeIdentifier(type.identifier) {
            provider.loadDataRepresentation(forTypeIdentifier: type.identifier) { data, _ in
                guard let data else { return }
                let fileExtension = type == .tiff ? "tiff" : "png"
                Task { @MainActor in
                    do { try session.openPasted(data, fileExtension: fileExtension) } catch {
                        exportMessage = "The pasted image could not be read: \(error.localizedDescription)"
                    }
                }
            }
            return
        }
    }

    /// Outlines the fields that were read on the image (mockup 07).
    private var highlightBoxes: some View {
        let boxes = session.highlights
        return GeometryReader { proxy in
            ForEach(ScanHighlights.keys.filter { boxes[$0] != nil }, id: \.self) { key in
                let box = boxes[key] ?? .zero
                let inset = MFSpacing.xxs
                RoundedRectangle(cornerRadius: MFRadius.xs)
                    .fill(MFColor.accent.opacity(0.12))
                    .overlay(RoundedRectangle(cornerRadius: MFRadius.xs).strokeBorder(MFColor.accent.opacity(0.55), lineWidth: 1.5))
                    .frame(width: box.width * proxy.size.width + inset * 2,
                           height: box.height * proxy.size.height + inset * 2)
                    .offset(x: box.minX * proxy.size.width - inset, y: box.minY * proxy.size.height - inset)
                    .accessibilityHidden(true)
            }
        }
    }

    // MARK: Fields

    @ViewBuilder
    private var detailPane: some View {
        switch session.phase {
        case .empty:
            AppleEmptyState(systemImage: "doc.text.viewfinder", title: "Scan a document",
                            message: "Apple's built-in text recognition reads it on this Mac and shows all of its text. "
                                + "Receipts, events, business cards and tables get their own next step; "
                                + "you check everything first.")
        case .reading:
            VStack(spacing: MFSpacing.m) {
                ProgressView().controlSize(.regular)
                Text("Reading on this Mac…")
                    .font(MFFont.body)
                    .foregroundStyle(MFColor.secondary)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .accessibilityElement(children: .combine)
        case .failed(let message):
            VStack(spacing: MFSpacing.m) {
                AppleEmptyState(systemImage: "exclamationmark.triangle", title: "This scan could not be read",
                                message: message, actionTitle: "Open another file", action: openFile)
                    .frame(maxHeight: .infinity)
                Button("Clear") { session.reset() }
                    .buttonStyle(.appleQuiet)
                    .padding(.bottom, MFSpacing.xl)
            }
        case .review, .saved:
            MFScroll { review.padding(.horizontal, MFSpacing.xl + MFSpacing.s).padding(.vertical, MFSpacing.xl) }
        }
    }

    private var review: some View {
        VStack(alignment: .leading, spacing: MFSpacing.l) {
            HStack(spacing: MFSpacing.m) {
                Text(session.title)
                    .font(MFFont.title)
                    .foregroundStyle(MFColor.ink)
                    .lineLimit(2)
                    .accessibilityAddTraits(.isHeader)
                AppleChip(text: "Read on this Mac", tone: .success)
                if session.isPreparing { ProgressView().controlSize(.small) }
                Spacer(minLength: MFSpacing.s)
                ScanReadAsMenu(session: session)
            }
            Text(explanation)
                .font(MFFont.body)
                .foregroundStyle(MFColor.secondary)
                .lineSpacing(MFLineSpacing.normal)
                .fixedSize(horizontal: false, vertical: true)
            switch session.kind {
            case .receipt:
                VStack(spacing: 0) {
                    ForEach(ScanFieldKey.allCases) { key in
                        fieldRow(key)
                        Rectangle().fill(MFColor.hairlineFaint).frame(height: 1)
                    }
                }
                saveCard
            case .event:
                ScanEventCard(session: session)
            case .contact:
                ScanContactCard(session: session)
            case .table:
                ScanTablesCard(session: session) { exportMessage = $0 }
            case .document:
                EmptyView()
            }
            ForEach([exportMessage, session.notice].compactMap { $0 }, id: \.self) { message in
                Text(message)
                    .font(MFFont.callout)
                    .foregroundStyle(MFColor.secondary)
            }
            ScanFullTextCard(session: session) { exportMessage = $0 }
        }
    }

    private var explanation: String {
        let read = "Apple’s built-in text recognition read it on this Mac"
        switch session.kind {
        case .receipt:
            let filler = session.filledByLabel.map { "; \($0 == "Apple on-device" ? "the on-device model" : $0) filled the fields" }
                ?? "; the fields come from the printed text"
            return "Apple’s built-in text recognition read the receipt" + filler
                + ". Check them, then save the row. Fields marked Check need a look."
        case .event:
            return read + " and found an event. Check it, then send it to Calendar, where you add it."
        case .contact:
            return read + " and found a business card. Check the details, then open it in Contacts."
        case .table:
            return read + " and found a table. Export it as CSV for Numbers or Excel."
        case .document:
            return read + ". Copy the text, save it to Notes or export it."
        }
    }

    private func fieldRow(_ key: ScanFieldKey) -> some View {
        let field = session.fields[key]
        let needsCheck = field?.confidence != .sure
        return HStack(spacing: MFSpacing.l) {
            Text(key.label)
                .font(MFFont.body)
                .foregroundStyle(MFColor.secondary)
                .frame(width: ScanMetrics.fieldLabelWidth, alignment: .leading)
            AppleTextField(title: key.label, text: Binding(
                get: { session.fields[key]?.value ?? "" },
                set: { session.setValue($0, for: key) }))
                .font(MFFont.body)
                .foregroundStyle(MFColor.ink)
                .padding(.horizontal, MFSpacing.m)
                .padding(.vertical, MFSpacing.s)
                .background(RoundedRectangle(cornerRadius: MFRadius.m)
                    .fill(needsCheck ? MFColor.previewFill.opacity(0.5) : MFColor.surface))
                .overlay(RoundedRectangle(cornerRadius: MFRadius.m)
                    .strokeBorder(needsCheck ? MFColor.preview.opacity(0.45) : MFColor.hairlineStrong))
                .disabled(session.phase == .saved)
                .accessibilityLabel(key.label)
                .accessibilityHint(needsCheck ? "Needs checking" : "Read with confidence")
            AppleChip(text: needsCheck ? ScanFieldConfidence.check.label : ScanFieldConfidence.sure.label,
                      tone: needsCheck ? .preview : .success)
                .frame(width: ScanMetrics.chipWidth, alignment: .leading)
                .accessibilityHidden(true)
        }
        .padding(.vertical, MFSpacing.s + MFSpacing.xxs)
    }

    private var saveCard: some View {
        AppleActionCard {
            HStack(spacing: MFSpacing.m) {
                VStack(alignment: .leading, spacing: MFSpacing.xxs) {
                    Text(session.phase == .saved ? "Saved to your scans" : "Add as a row to your expenses")
                        .font(MFFont.bodyEmphasis)
                        .foregroundStyle(MFColor.ink)
                    Text(session.phase == .saved ? "Export every saved receipt as a CSV file."
                         : "Saved on this Mac · export as CSV for Numbers or Excel")
                        .font(MFFont.callout)
                        .foregroundStyle(MFColor.secondary)
                }
                Spacer(minLength: MFSpacing.m)
                Button("Export CSV", action: exportCSV)
                    .buttonStyle(.appleSecondary)
                if session.phase == .saved {
                    Button("Scan another", action: { session.reset() })
                        .buttonStyle(.applePrimary)
                } else {
                    Button("Add row", action: confirm)
                        .buttonStyle(.applePrimary)
                        .keyboardShortcut(.return, modifiers: .command)
                }
            }
        }
        .padding(.top, MFSpacing.s)
    }

    // MARK: Actions

    private func openFile() {
        ScanToolbarActions.chooseFile(for: session)
    }

    private func confirm() {
        do {
            try session.confirm()
            exportMessage = nil
        } catch {
            exportMessage = "The scan could not be saved: \(error.localizedDescription)"
        }
    }

    /// Exports the receipts already saved; a scan in review stays unsaved
    /// until the person clicks Add row.
    private func exportCSV() {
        do {
            let file = try session.exportCSVFile()
            let panel = NSSavePanel()
            panel.allowedContentTypes = [.commaSeparatedText]
            panel.nameFieldStringValue = "Receipts.csv"
            guard panel.runModal() == .OK, let url = panel.url else { return }
            try file.write(to: url, options: .atomic)
            exportMessage = session.phase == .review
                ? "Exported your saved receipts to \(url.lastPathComponent). This one is not saved yet: click Add row."
                : "Exported to \(url.lastPathComponent)."
        } catch {
            exportMessage = "The CSV could not be written: \(error.localizedDescription)"
        }
    }
}

/// Capture (Later) and Open file in the shell toolbar, as their own view so
/// the retained toolbar item acts on the live session.
struct ScanToolbarActions: View {
    var session: ScanSession

    var body: some View {
        HStack(spacing: MFSpacing.s) {
            HStack(spacing: MFSpacing.s) {
                Button("Capture screen area") {}
                    .buttonStyle(.appleSecondary)
                    .disabled(true)
                AppleChip(text: "Later", tone: .preview)
            }
            .accessibilityElement(children: .combine)
            .accessibilityLabel("Capture screen area, coming later")
            // An icon, so the section pills keep their names at the
            // window's usual width.
            Button { session.reset() } label: { Image(systemName: "xmark.circle") }
                .buttonStyle(.appleSecondary)
                .disabled(session.phase == .empty)
                .help("Clear this scan and start again")
                .accessibilityLabel("Clear this scan")
            Button("Open file") { ScanToolbarActions.chooseFile(for: session) }
                .buttonStyle(.applePrimary)
                .keyboardShortcut("o", modifiers: .command)
                .help("Read a document, receipt, card or photo on this Mac")
        }
    }

    static func chooseFile(for session: ScanSession) {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.image, .pdf]
        panel.allowsMultipleSelection = false
        panel.message = "Choose a receipt or document to read on this Mac."
        guard panel.runModal() == .OK, let url = panel.url else { return }
        session.open(url)
    }
}

/// What the preview shows: a file and its page.
private struct PreviewKey: Hashable {
    var url: URL?
    var page: Int
}

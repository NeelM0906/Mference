import AppKit
import MferenceAppCore
import MferenceDesign
import SwiftUI
import UniformTypeIdentifiers

/// Scan (mockup 07): drop or open a receipt, check the fields Apple's text
/// recognition and the on-device model filled, save it, export CSV.
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
    /// The image decoded once per file (bounded), not on every keystroke.
    @State private var preview: (url: URL, image: CGImage)?

    var body: some View {
        HStack(spacing: 0) {
            imagePane
                .frame(width: 380)
                .clipped()
            Rectangle().fill(MFColor.hairline).frame(width: 1)
            detailPane
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        }
        .background(MFColor.surface)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .task(id: session.imageURL) { loadPreview() }
    }

    private func loadPreview() {
        guard let url = session.imageURL else { preview = nil; return }
        if preview?.url == url { return }
        // Decoded once per file and bounded to 2048 px, so it is cheap here.
        if let image = VisionScanner.loadImage(at: url, maximumPixelSize: 2_048) { preview = (url, image) }
    }

    // MARK: Image

    private var imagePane: some View {
        ZStack {
            StripedBackground()
            if let preview, preview.url == session.imageURL {
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
                    Text("Drop a receipt or document here")
                        .font(MFFont.bodyEmphasis)
                        .foregroundStyle(MFColor.ink)
                    Text("PNG, JPEG, HEIC or PDF. Read on this Mac.")
                        .font(MFFont.callout)
                        .foregroundStyle(MFColor.secondary)
                    Button("Open file", action: openFile)
                        .buttonStyle(.applePrimary)
                        .keyboardShortcut("o", modifiers: .command)
                }
                .padding(MFSpacing.xl)
            }
            if isDropTargeted {
                RoundedRectangle(cornerRadius: MFRadius.l)
                    .strokeBorder(MFColor.accent, style: StrokeStyle(lineWidth: 2, dash: [6, 4]))
                    .padding(MFSpacing.l)
                    .accessibilityHidden(true)
            }
        }
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
        .accessibilityLabel("Drop area for receipts and documents")
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
                    .overlay(RoundedRectangle(cornerRadius: MFRadius.xs).strokeBorder(MFColor.accent, lineWidth: 1.5))
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
            AppleEmptyState(systemImage: "receipt", title: "Scan a receipt",
                            message: "Apple's built-in text recognition reads it on this Mac and fills the fields. "
                                + "You check them before anything is saved.",
                            actionTitle: "Open file", action: openFile)
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
            AppleEmptyState(systemImage: "exclamationmark.triangle", title: "This scan could not be read",
                            message: message, actionTitle: "Open another file", action: openFile)
        case .review, .saved:
            FitOrScroll { review.padding(.horizontal, MFSpacing.xl + MFSpacing.s).padding(.vertical, MFSpacing.xl) }
        }
    }

    private var review: some View {
        VStack(alignment: .leading, spacing: MFSpacing.l) {
            HStack(spacing: MFSpacing.m) {
                Text(session.title)
                    .font(MFFont.title)
                    .foregroundStyle(MFColor.ink)
                    .accessibilityAddTraits(.isHeader)
                AppleChip(text: "Read on this Mac", tone: .success)
                Spacer(minLength: MFSpacing.m)
                Button("Open file", action: openFile)
                    .buttonStyle(.appleSecondary)
                    .keyboardShortcut("o", modifiers: .command)
                    .help("Read another receipt or document")
            }
            Text(explanation)
                .font(MFFont.body)
                .foregroundStyle(MFColor.secondary)
                .fixedSize(horizontal: false, vertical: true)
            VStack(spacing: 0) {
                ForEach(ScanFieldKey.allCases) { key in
                    fieldRow(key)
                    Rectangle().fill(MFColor.hairlineFaint).frame(height: 1)
                }
            }
            saveCard
            if let exportMessage {
                Text(exportMessage)
                    .font(MFFont.callout)
                    .foregroundStyle(MFColor.secondary)
            }
        }
    }

    private var explanation: String {
        let reader = "Apple’s built-in text recognition read the receipt"
        let filler = session.filledByLabel.map { "; \($0 == "Apple on-device" ? "the on-device model" : $0) filled the fields" }
            ?? "; the fields come from the printed text"
        return reader + filler + ". Check them, then save the row. Fields marked Check need a look."
    }

    private func fieldRow(_ key: ScanFieldKey) -> some View {
        let field = session.fields[key]
        let needsCheck = field?.confidence != .sure
        return HStack(spacing: MFSpacing.l) {
            Text(key.label)
                .font(MFFont.body)
                .foregroundStyle(MFColor.secondary)
                .frame(width: 140, alignment: .leading)
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
                .frame(width: 84, alignment: .leading)
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
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.image, .pdf]
        panel.allowsMultipleSelection = false
        panel.message = "Choose a receipt or document to read on this Mac."
        guard panel.runModal() == .OK, let url = panel.url else { return }
        session.open(url)
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

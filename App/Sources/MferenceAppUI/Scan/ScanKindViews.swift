import AppKit
import MferenceAppCore
import MferenceDesign
import SwiftUI
import UniformTypeIdentifiers

/// Layout sizes of the Scan screen.
enum ScanMetrics {
    static let imagePaneWidth: CGFloat = 380
    static let fieldLabelWidth: CGFloat = 140
    static let chipWidth: CGFloat = 84
    /// Rows of a table shown before "and N more rows".
    static let tablePreviewRows = 8
    static let tableCellMinWidth: CGFloat = 72
}

/// "Read as": the person chooses what the scan is when the guess was wrong.
struct ScanReadAsMenu: View {
    var session: ScanSession

    var body: some View {
        Menu {
            ForEach(ScanKind.allCases) { kind in
                Button {
                    session.read(as: kind)
                } label: {
                    if kind == session.kind { Label(kind.label, systemImage: "checkmark") } else { Text(kind.label) }
                }
            }
        } label: {
            HStack(spacing: MFSpacing.xs) {
                Text("Read as \(session.kind.label)")
                Image(systemName: "chevron.down").font(MFFont.micro)
            }
            .font(MFFont.callout)
            .foregroundStyle(MFColor.ink)
            .padding(.horizontal, MFSpacing.m)
            .padding(.vertical, MFSpacing.xs + MFSpacing.xxs)
            .background(Capsule().fill(MFColor.fill))
        }
        .menuStyle(.button)
        .buttonStyle(.plain)
        .menuIndicator(.hidden)
        .fixedSize()
        .disabled(session.phase != .review || session.isPreparing)
        .accessibilityLabel("Read as \(session.kind.label). Choose what this scan is.")
    }
}

/// Everything that was read, with Copy, Save to Notes and Export .txt.
struct ScanFullTextCard: View {
    var session: ScanSession
    var onExport: (String) -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: MFSpacing.s) {
            HStack(spacing: MFSpacing.s) {
                AppleSectionHeader(text: "Full text")
                if session.pageCount > 1 {
                    Text("\(session.pageCount) pages")
                        .font(MFFont.caption)
                        .foregroundStyle(MFColor.tertiary)
                }
                Spacer()
                Button {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(session.fullText, forType: .string)
                } label: { Label("Copy", systemImage: "doc.on.doc") }
                    .buttonStyle(.appleQuiet)
                if session.canSaveToNotes {
                    Button { session.saveTextToNotes() } label: { Label("Save to Notes", systemImage: "note.text") }
                        .buttonStyle(.appleQuiet)
                }
                Button { exportText() } label: { Label("Export .txt", systemImage: "square.and.arrow.up") }
                    .buttonStyle(.appleQuiet)
            }
            // In full: the review pane around it already scrolls.
            Text(session.fullText)
                .font(MFFont.callout)
                .foregroundStyle(MFColor.ink)
                .lineSpacing(MFLineSpacing.normal)
                .textSelection(.enabled)
                .fixedSize(horizontal: false, vertical: true)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(MFSpacing.m)
            .background(RoundedRectangle(cornerRadius: MFRadius.m).fill(MFColor.canvas))
            .overlay(RoundedRectangle(cornerRadius: MFRadius.m).strokeBorder(MFColor.hairlineStrong))
            .accessibilityLabel("Full text of the scan")
        }
    }

    private func exportText() {
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.plainText]
        panel.nameFieldStringValue = session.exportName + ".txt"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do {
            try session.textFile.write(to: url, options: .atomic)
            session.noteExported(to: url.lastPathComponent)
        } catch {
            onExport("The text could not be written: \(error.localizedDescription)")
        }
    }
}

/// A flyer's event, checked here, then handed to Calendar's proposals.
struct ScanEventCard: View {
    @Bindable var session: ScanSession
    @Environment(\.mfRendersStatically) private var staticRendering

    var body: some View {
        AppleActionCard {
            if let event = session.event {
                VStack(alignment: .leading, spacing: MFSpacing.m) {
                    ScanFieldRow(label: "Event", text: Binding(
                        get: { session.event?.title ?? "" },
                        set: { session.event?.title = $0 }))
                    HStack(spacing: MFSpacing.l) {
                        Text("When")
                            .font(MFFont.body)
                            .foregroundStyle(MFColor.secondary)
                            .frame(width: ScanMetrics.fieldLabelWidth, alignment: .leading)
                        if staticRendering {
                            Text(CalendarTimeFormat.dayRange(event.start, event.end))
                                .font(MFFont.body)
                                .foregroundStyle(MFColor.ink)
                        } else {
                            DatePicker("Starts", selection: Binding(
                                get: { session.event?.start ?? event.start },
                                set: { start in
                                    let length = (session.event.map { $0.end.timeIntervalSince($0.start) }) ?? 3_600
                                    session.event?.start = start
                                    session.event?.end = start.addingTimeInterval(length)
                                }))
                                .labelsHidden()
                            Text("to").foregroundStyle(MFColor.secondary)
                            DatePicker("Ends", selection: Binding(
                                get: { session.event?.end ?? event.end },
                                set: { session.event?.end = max($0, session.event?.start ?? $0) }),
                                       displayedComponents: .hourAndMinute)
                                .labelsHidden()
                        }
                        Spacer(minLength: 0)
                    }
                    HStack {
                        Text("Nothing changes until you add it in Calendar.")
                            .font(MFFont.callout)
                            .foregroundStyle(MFColor.secondary)
                        Spacer()
                        Button("Add to Calendar") { session.addEventToCalendar() }
                            .buttonStyle(.applePrimary)
                            .disabled(event.title.trimmingCharacters(in: .whitespaces).isEmpty)
                    }
                }
            } else {
                Text("No date and time were found in this text. Read it as a document, or add the event in Calendar.")
                    .font(MFFont.body)
                    .foregroundStyle(MFColor.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }
}

/// A business card's details, checked here, then opened in Contacts.
struct ScanContactCard: View {
    @Bindable var session: ScanSession

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            ScanFieldRow(label: "Name", text: $session.contact.name)
            ScanFieldRow(label: "Company", text: $session.contact.organization)
            ScanFieldRow(label: "Job title", text: $session.contact.jobTitle)
            ScanFieldRow(label: "Phone", text: list(\.phones))
            ScanFieldRow(label: "Email", text: list(\.emails))
            ScanFieldRow(label: "Address", text: $session.contact.address)
            ScanFieldRow(label: "Website", text: $session.contact.website)
            AppleActionCard {
                HStack {
                    Text("Contacts shows the card and asks before adding it.")
                        .font(MFFont.callout)
                        .foregroundStyle(MFColor.secondary)
                    Spacer()
                    Button("Add to Contacts") { session.addContact() }
                        .buttonStyle(.applePrimary)
                        .disabled(session.contact.isEmpty)
                }
            }
            .padding(.top, MFSpacing.m)
        }
    }

    /// Several numbers or addresses, edited as one comma-separated line.
    private func list(_ path: WritableKeyPath<ContactDraft, [String]>) -> Binding<String> {
        Binding(get: { session.contact[keyPath: path].joined(separator: ", ") },
                set: { session.contact[keyPath: path] = $0.split(separator: ",")
                    .map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty } })
    }
}

/// The tables Vision found, each with Export CSV.
struct ScanTablesCard: View {
    var session: ScanSession
    var onMessage: (String) -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: MFSpacing.m) {
            if session.tables.isEmpty {
                Text("No table was found in this text. Read it as a document to copy or export the text.")
                    .font(MFFont.body)
                    .foregroundStyle(MFColor.secondary)
            }
            ForEach(Array(session.tables.enumerated()), id: \.offset) { index, table in
                VStack(alignment: .leading, spacing: MFSpacing.s) {
                    HStack {
                        Text(tableTitle(index, table))
                            .font(MFFont.bodyEmphasis)
                            .foregroundStyle(MFColor.ink)
                        Spacer()
                        Button("Export CSV") { export(index) }
                            .buttonStyle(.appleSecondary)
                    }
                    grid(table)
                }
            }
        }
    }

    private func tableTitle(_ index: Int, _ table: ScanTable) -> String {
        let size = "\(table.rows.count) rows, \(table.columnCount) columns"
        let page = session.pageCount > 1 ? " · page \(table.page + 1)" : ""
        return session.tables.count > 1 ? "Table \(index + 1) · \(size)\(page)" : "\(size)\(page)"
    }

    private func grid(_ table: ScanTable) -> some View {
        let shown = Array(table.rows.prefix(ScanMetrics.tablePreviewRows))
        return VStack(alignment: .leading, spacing: 0) {
            Grid(alignment: .leading, horizontalSpacing: 0, verticalSpacing: 0) {
                ForEach(Array(shown.enumerated()), id: \.offset) { rowIndex, row in
                    GridRow {
                        ForEach(0..<table.columnCount, id: \.self) { column in
                            Text(column < row.count ? row[column] : "")
                                .font(rowIndex == 0 ? MFFont.calloutEmphasis : MFFont.callout)
                                .foregroundStyle(MFColor.ink)
                                .lineLimit(2)
                                .frame(minWidth: ScanMetrics.tableCellMinWidth, alignment: .leading)
                                .padding(.horizontal, MFSpacing.s)
                                .padding(.vertical, MFSpacing.xs)
                        }
                    }
                    if rowIndex < shown.count - 1 {
                        Rectangle().fill(MFColor.hairlineFaint).frame(height: MFLayout.hairline)
                    }
                }
            }
            if table.rows.count > shown.count {
                Text("and \(table.rows.count - shown.count) more rows")
                    .font(MFFont.caption)
                    .foregroundStyle(MFColor.tertiary)
                    .padding(MFSpacing.s)
            }
        }
        .background(RoundedRectangle(cornerRadius: MFRadius.m).fill(MFColor.canvas))
        .overlay(RoundedRectangle(cornerRadius: MFRadius.m).strokeBorder(MFColor.hairlineStrong))
    }

    private func export(_ index: Int) {
        guard let data = session.tableFile(index) else { return }
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.commaSeparatedText]
        panel.nameFieldStringValue = session.exportName + (session.tables.count > 1 ? " \(index + 1)" : "") + ".csv"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do {
            try data.write(to: url, options: .atomic)
            session.noteExported(to: url.lastPathComponent)
        } catch {
            onMessage("The CSV could not be written: \(error.localizedDescription)")
        }
    }
}

/// A labeled, editable line (contact and event details).
struct ScanFieldRow: View {
    var label: String
    @Binding var text: String

    var body: some View {
        HStack(spacing: MFSpacing.l) {
            Text(label)
                .font(MFFont.body)
                .foregroundStyle(MFColor.secondary)
                .frame(width: ScanMetrics.fieldLabelWidth, alignment: .leading)
            AppleTextField(title: label, text: $text)
                .font(MFFont.body)
                .foregroundStyle(MFColor.ink)
                .padding(.horizontal, MFSpacing.m)
                .padding(.vertical, MFSpacing.s)
                .background(RoundedRectangle(cornerRadius: MFRadius.m).fill(MFColor.surface))
                .overlay(RoundedRectangle(cornerRadius: MFRadius.m).strokeBorder(MFColor.hairlineStrong))
                .accessibilityLabel(label)
        }
        .padding(.vertical, MFSpacing.xs + MFSpacing.xxs)
    }
}

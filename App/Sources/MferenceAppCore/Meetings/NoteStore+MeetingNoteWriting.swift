import Foundation

/// Meeting and file notes write through the Notes store, so the summary,
/// action items and meeting lines are ordinary note content: saved with the
/// body, searchable, editable, and still there after a relaunch (spec 4.5).
extension NoteStore: MeetingNoteWriting {
    public func createNote(id: UUID, kind: NoteKind, title: String, createdAt: Date) throws {
        // The id was chosen before: it names the recordings folder.
        try createNote(id: id, kind: kind, title: title, now: createdAt)
    }

    public func setTitle(_ title: String, of noteID: UUID) throws {
        guard let note = try self.note(id: noteID) else { return }
        try setTitle(title, of: note)
    }

    public func replaceMeetingSections(_ sections: MeetingSections, in noteID: UUID) throws {
        try edit(noteID) { $0 = MeetingBlocks.replacingSections(in: $0, with: sections) }
    }

    public func actionItems(in noteID: UUID) -> [MeetingActionItemRow] {
        guard let note = try? self.note(id: noteID) else { return [] }
        return MeetingBlocks.actionItems(in: document(of: note).blocks)
    }

    public func personalNotes(in noteID: UUID) -> String {
        guard let note = try? self.note(id: noteID) else { return "" }
        return MeetingBlocks.personalNotes(in: document(of: note).blocks)
    }

    public func appendParagraph(_ text: String, to noteID: UUID) throws {
        try edit(noteID) { $0.append(.paragraph(runs: [TextRun(text)])) }
    }

    public func appendMarker(at time: TimeInterval, to noteID: UUID) throws {
        try edit(noteID) { $0.append(.marker(time: time)) }
    }

    public func transcriptDidChange(in noteID: UUID) {
        guard let note = try? self.note(id: noteID) else { return }
        try? transcriptDidChange(in: note)
    }

    /// Reads the body, changes its blocks, saves (refused for a body this
    /// build must not overwrite).
    private func edit(_ noteID: UUID, _ change: (inout [NoteDocument.Block]) -> Void) throws {
        guard let note = try self.note(id: noteID) else { return }
        var document = document(of: note)
        change(&document.blocks)
        try save(document, to: note)
    }
}

/// How the meeting sections look as note blocks, right after the title line:
/// heading "Summary" + bullets, heading "Decisions" + bullets (when there are
/// some), heading "Action items" + checklist rows ("text · owner").
enum MeetingBlocks {
    static let summaryHeading = "Summary"
    static let decisionsHeading = "Decisions"
    static let actionItemsHeading = "Action items"
    static let ownerSeparator = " · "

    static func blocks(for sections: MeetingSections) -> [NoteDocument.Block] {
        var blocks: [NoteDocument.Block] = [heading(summaryHeading)]
        blocks += sections.summary.map { .bullet(runs: [TextRun($0)], indent: 0) }
        if !sections.decisions.isEmpty {
            blocks.append(heading(decisionsHeading))
            blocks += sections.decisions.map { .bullet(runs: [TextRun($0)], indent: 0) }
        }
        if !sections.actionItems.isEmpty {
            blocks.append(heading(actionItemsHeading))
            blocks += sections.actionItems.map { item in
                var runs = [TextRun(item.text)]
                if let owner = item.owner { runs.append(TextRun(ownerSeparator + owner)) }
                return .check(runs: runs, done: false, indent: 0)
            }
        }
        return blocks
    }

    /// The written sections' blocks, if the body starts (after its title
    /// line) with them: the Summary heading and everything that still has
    /// the sections' shape after it.
    static func sectionRange(in blocks: [NoteDocument.Block]) -> Range<Int>? {
        let start = firstBodyIndex(blocks)
        guard start < blocks.count, isHeading(blocks[start], summaryHeading) else { return nil }
        var index = start + 1
        var section = summaryHeading
        while index < blocks.count {
            let block = blocks[index]
            if section != decisionsHeading, section != actionItemsHeading, isHeading(block, decisionsHeading) {
                section = decisionsHeading
            } else if section != actionItemsHeading, isHeading(block, actionItemsHeading) {
                section = actionItemsHeading
            } else if section == actionItemsHeading, case .check = block {
            } else if section != actionItemsHeading, case .bullet = block {
            } else {
                break
            }
            index += 1
        }
        return start..<index
    }

    /// Replaces the sections written before (Redo), or puts them after the title line.
    static func replacingSections(in blocks: [NoteDocument.Block], with sections: MeetingSections)
        -> [NoteDocument.Block] {
        var result = blocks
        let fresh = Self.blocks(for: sections)
        if let range = sectionRange(in: blocks) {
            result.replaceSubrange(range, with: fresh)
        } else {
            result.insert(contentsOf: fresh, at: firstBodyIndex(blocks))
        }
        return result
    }

    /// The checklist rows under "Action items" as they stand now.
    static func actionItems(in blocks: [NoteDocument.Block]) -> [MeetingActionItemRow] {
        guard let range = sectionRange(in: blocks) else { return [] }
        var inActionItems = false
        var rows: [MeetingActionItemRow] = []
        for block in blocks[range] {
            if isHeading(block, actionItemsHeading) { inActionItems = true; continue }
            guard inActionItems, case .check(let runs, let done, _) = block else { continue }
            let text = runs.map(\.text).joined().trimmingCharacters(in: .whitespacesAndNewlines)
            guard !text.isEmpty else { continue }
            if let split = text.range(of: ownerSeparator, options: .backwards) {
                let owner = text[split.upperBound...].trimmingCharacters(in: .whitespaces)
                let task = text[..<split.lowerBound].trimmingCharacters(in: .whitespaces)
                if !owner.isEmpty, !task.isEmpty {
                    rows.append(MeetingActionItemRow(item: MeetingActionItem(text: task, owner: owner), isDone: done))
                    continue
                }
            }
            rows.append(MeetingActionItemRow(item: MeetingActionItem(text: text), isDone: done))
        }
        return rows
    }

    /// What the person typed: the body's text outside the title line and the sections.
    static func personalNotes(in blocks: [NoteDocument.Block]) -> String {
        let skipped = sectionRange(in: blocks)
        let start = firstBodyIndex(blocks)
        return blocks.indices.filter { $0 >= start && !(skipped?.contains($0) ?? false) }
            .map { blocks[$0] }
            .filter(\.isText)
            .map { $0.text.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
            .joined(separator: "\n")
    }

    private static func heading(_ text: String) -> NoteDocument.Block {
        .heading(level: .heading, runs: [TextRun(text)])
    }

    private static func isHeading(_ block: NoteDocument.Block, _ text: String) -> Bool {
        guard case .heading(.heading, let runs) = block else { return false }
        return runs.map(\.text).joined().trimmingCharacters(in: .whitespaces) == text
    }

    private static func firstBodyIndex(_ blocks: [NoteDocument.Block]) -> Int {
        if case .heading(.title, _)? = blocks.first { return 1 }
        return 0
    }
}

import AppKit
import MferenceAppCore
import UniformTypeIdentifiers

/// The note editor's text view (TextKit 2). It keeps the text in the
/// canonical form `NoteTextMapping` reads (block styles and run styles as
/// attributes) and re-derives fonts and paragraph styles after every edit.
///
/// Editing behaviour (Notes spec 4.1): Markdown shortcuts while typing
/// (`# `, `- ` / `* `, `1. `, `[] `), Return continues a list and ends it on
/// an empty item, Tab / Shift-Tab indent list items, ⌘B ⌘I ⌘U and
/// strikethrough toggle run styles, ⇧⌘L toggles a checklist, clicking a box
/// ticks it, paste keeps the supported formats only, links open in the
/// default browser. Undo covers all of it.
final class NoteTextView: NSTextView, NSTextLayoutManagerDelegate, NSTextStorageDelegate {
    /// Copy and paste between notes keeps every block.
    static let blocksPasteboardType = NSPasteboard.PasteboardType("ai.mference.note-blocks")

    /// Called after every change to the text (typing, styles, ticks).
    var onTextChange: (() -> Void)?
    /// Called with the time of a clicked transcript chip or marker.
    var onTimeChipClick: ((TimeInterval) -> Void)?
    /// Stores pasted or dropped image data as an attachment; returns its id.
    var importImage: ((Data, String) -> String?)?
    /// Copies a dropped or pasted image or audio file into the note; returns
    /// its block.
    var importFile: ((URL) -> NoteDocument.Block?)?
    /// Copies another note's attachment into this one (pasting blocks
    /// between notes); returns the new id.
    var copyAttachment: ((String, UUID) -> String?)?
    /// The note shown, stamped on copied blocks.
    var noteID: UUID?
    var mappingContext = NoteTextMapping.Context()

    /// The block style of an empty last line (it has no character to carry it).
    var emptyLastLineStyle: NoteBlockStyle = .title

    private var isRestyling = false
    private var isLoading = false

    /// The date line centered above the title; it scrolls with the text.
    var dateLine = "" {
        didSet {
            dateLabel.stringValue = dateLine
            dateLabel.isHidden = dateLine.isEmpty
            updateInsets()
        }
    }

    /// One plain line under the date: why a note opens read-only.
    var notice = "" {
        didSet {
            noticeLabel.stringValue = notice
            noticeLabel.isHidden = notice.isEmpty
            updateInsets()
        }
    }

    private lazy var noticeLabel: NSTextField = {
        let label = NSTextField(wrappingLabelWithString: "")
        label.font = NoteEditorStyle.geist(NoteEditorMetrics.dateSize, .regular)
        label.textColor = NoteEditorStyle.secondary
        label.alignment = .center
        label.isHidden = true
        label.autoresizingMask = [.width]
        return label
    }()

    private lazy var dateLabel: NSTextField = {
        let label = NSTextField(labelWithString: "")
        label.font = NoteEditorStyle.geist(NoteEditorMetrics.dateSize, .regular)
        label.textColor = NoteEditorStyle.tertiary
        label.alignment = .center
        label.isHidden = true
        label.autoresizingMask = [.width]
        label.setAccessibilityLabel("Edited")
        return label
    }()

    private func updateInsets() {
        if dateLabel.superview == nil { addSubview(dateLabel) }
        if noticeLabel.superview == nil { addSubview(noticeLabel) }
        dateLabel.frame = CGRect(x: 0, y: NoteEditorMetrics.topInset, width: bounds.width,
                                 height: NoteEditorMetrics.dateLineHeight)
        let noticeTop = NoteEditorMetrics.topInset + (dateLine.isEmpty ? 0 : NoteEditorMetrics.dateLineHeight)
        noticeLabel.frame = CGRect(x: NoteEditorMetrics.horizontalInset, y: noticeTop,
                                   width: max(0, bounds.width - 2 * NoteEditorMetrics.horizontalInset),
                                   height: notice.isEmpty ? 0 : NoteEditorMetrics.noticeHeight)
        let noticeHeight = notice.isEmpty ? 0 : NoteEditorMetrics.noticeHeight
        let dateHeight = (dateLine.isEmpty ? 0 : NoteEditorMetrics.dateLineHeight + NoteEditorMetrics.titleSpacingBefore)
            + noticeHeight
        textContainerInset = NSSize(width: NoteEditorMetrics.horizontalInset - NoteEditorMetrics.lineFragmentPadding,
                                    height: NoteEditorMetrics.topInset + dateHeight)
    }

    // MARK: Setup

    /// Configures a view made with `NoteTextView(usingTextLayoutManager: true)`.
    func configure() {
        isRichText = true
        importsGraphics = false
        allowsUndo = true
        isAutomaticLinkDetectionEnabled = true
        isAutomaticQuoteSubstitutionEnabled = false
        isAutomaticDashSubstitutionEnabled = false
        isAutomaticSpellingCorrectionEnabled = false
        usesFindBar = true
        isIncrementalSearchingEnabled = true
        drawsBackground = false
        updateInsets()
        textContainer?.lineFragmentPadding = NoteEditorMetrics.lineFragmentPadding
        linkTextAttributes = [.foregroundColor: NoteEditorStyle.accent, .cursor: NSCursor.pointingHand]
        textLayoutManager?.delegate = self
        textStorage?.delegate = self
        setAccessibilityLabel("Note")
        setAccessibilityRoleDescription("note editor")
    }

    // MARK: Loading and reading

    /// Shows `document`; clears undo (a different note's history never applies).
    func load(_ document: NoteDocument) {
        let (text, lastStyle) = NoteTextMapping.attributedString(from: document, context: mappingContext)
        isRestyling = true
        isLoading = true
        textStorage?.setAttributedString(text)
        isLoading = false
        isRestyling = false
        // Phrases kept across a reload stay where they were.
        let length = (string as NSString).length
        for index in pendingPhrases.indices { pendingPhrases[index].location = min(pendingPhrases[index].location, length) }
        emptyLastLineStyle = lastStyle
        undoManager?.removeAllActions(withTarget: self)
        undoManager?.removeAllActions()
        let end = (string as NSString).length
        setSelectedRange(NSRange(location: end, length: 0))
        typingAttributes = typingAttributesForCaret()
    }

    /// The document as it is now (pending dictation markers left out).
    var document: NoteDocument {
        guard let textStorage else { return .empty }
        return NoteTextMapping.document(from: textStorage, lastLineStyle: lastLineStyle)
    }

    /// The style of the last line, whether or not it has characters.
    private var lastLineStyle: NoteBlockStyle {
        let text = string as NSString
        if text.length > 0, text.character(at: text.length - 1) != 0x0A {
            let line = text.lineRange(for: NSRange(location: text.length, length: 0))
            return textStorage?.attribute(.noteBlock, at: line.location, effectiveRange: nil) as? NoteBlockStyle ?? .body
        }
        return emptyLastLineStyle
    }

    // MARK: Line helpers

    private var nsString: NSString { string as NSString }

    /// Ranges of the lines the selection touches (each with its line break).
    private var selectedLineRanges: [NSRange] {
        let selection = selectedRange()
        var ranges: [NSRange] = []
        var location = nsString.lineRange(for: NSRange(location: selection.location, length: 0)).location
        let end = NSMaxRange(selection)
        repeat {
            let line = nsString.lineRange(for: NSRange(location: location, length: 0))
            ranges.append(line)
            location = NSMaxRange(line)
        } while location < end && location < nsString.length
        return ranges
    }

    /// The style of the line at `location`.
    func blockStyle(at location: Int) -> NoteBlockStyle {
        let line = nsString.lineRange(for: NSRange(location: min(location, nsString.length), length: 0))
        if line.length == 0 { return emptyLastLineStyle }
        return textStorage?.attribute(.noteBlock, at: line.location, effectiveRange: nil) as? NoteBlockStyle ?? .body
    }

    /// The style of the line holding the caret (the Aa menu's tick).
    var currentBlockStyle: NoteBlockStyle { blockStyle(at: selectedRange().location) }

    private func typingAttributesForCaret() -> [NSAttributedString.Key: Any] {
        let location = selectedRange().location
        var attributes = NoteEditorStyle.typingAttributes(blockStyle(at: location))
        if location > 0, let storage = textStorage, location <= storage.length {
            let previous = storage.attributes(at: location - 1, effectiveRange: nil)
            if nsString.character(at: location - 1) != 0x0A {
                for key in [NSAttributedString.Key.noteBold, .noteItalic, .underlineStyle, .strikethroughStyle] {
                    attributes[key] = previous[key]
                }
            }
        }
        return restyledTyping(attributes)
    }

    private func restyledTyping(_ attributes: [NSAttributedString.Key: Any]) -> [NSAttributedString.Key: Any] {
        let style = attributes[.noteBlock] as? NoteBlockStyle ?? .body
        let run = TextRun("", bold: attributes[.noteBold] != nil, italic: attributes[.noteItalic] != nil,
                          underline: (attributes[.underlineStyle] as? Int ?? 0) != 0,
                          strikethrough: (attributes[.strikethroughStyle] as? Int ?? 0) != 0)
        return NoteEditorStyle.attributes(block: style, next: nil, run: run)
    }

    // MARK: Changing lines

    /// Sets the style of every line in `ranges` as one undoable change.
    func setBlockStyle(_ make: (NoteBlockStyle) -> NoteBlockStyle, lines ranges: [NSRange]) {
        guard isEditable, let storage = textStorage else { return }
        var changed = false
        for line in ranges {
            if line.length == 0 {
                emptyLastLineStyle = make(emptyLastLineStyle)
                changed = true
                continue
            }
            let old = storage.attribute(.noteBlock, at: line.location, effectiveRange: nil) as? NoteBlockStyle ?? .body
            let new = make(old)
            guard new != old, shouldChangeText(in: line, replacementString: nil) else { continue }
            storage.addAttribute(.noteBlock, value: new, range: line)
            didChangeText()
            changed = true
        }
        if changed {
            typingAttributes = typingAttributesForCaret()
            invalidateListLayout()
            onTextChange?()
        }
    }

    /// Text style menu: title, heading, subheading, body; or a list kind.
    func applyBlockKind(_ kind: NoteBlockStyle.Kind) {
        setBlockStyle({ NoteBlockStyle(kind, indent: kind == $0.kind || NoteBlockStyle(kind).isList ? $0.indent : 0) },
                      lines: selectedLineRanges)
    }

    /// ⇧⌘L: lines become a checklist, or back to body when they all are one.
    func toggleChecklist() {
        let allChecks = selectedLineRanges.allSatisfy { blockStyle(at: $0.location).kind == .check }
        applyBlockKind(allChecks ? .body : .check)
    }

    func toggleDone(lineAt location: Int) {
        let line = nsString.lineRange(for: NSRange(location: location, length: 0))
        guard blockStyle(at: line.location).kind == .check else { return }
        setBlockStyle({ $0.kind == .check ? $0.with(done: !$0.done) : $0 }, lines: [line])
        announceCheck(at: line.location)
    }

    /// ⇧⌘U ("Mark as checked"): ticks or unticks the checklist items the
    /// selection touches, from the keyboard.
    func toggleDoneForSelection() {
        let lines = selectedLineRanges.filter { blockStyle(at: $0.location).kind == .check }
        guard let first = lines.first else { return }
        let tick = !blockStyle(at: first.location).done
        setBlockStyle({ $0.kind == .check ? $0.with(done: tick) : $0 }, lines: lines)
        announceCheck(at: first.location)
    }

    /// VoiceOver hears the new state (the box is drawn, not text).
    private func announceCheck(at location: Int) {
        let done = blockStyle(at: location).done
        NSAccessibility.post(element: self, notification: .announcementRequested,
                             userInfo: [.announcement: done ? "Checked" : "Unchecked",
                                        .priority: NSAccessibilityPriorityLevel.high.rawValue])
    }

    override func accessibilityCustomActions() -> [NSAccessibilityCustomAction]? {
        var actions = super.accessibilityCustomActions() ?? []
        let style = currentBlockStyle
        if style.kind == .check {
            actions.append(NSAccessibilityCustomAction(name: style.done ? "Uncheck item" : "Check item") { [weak self] in
                self?.toggleDoneForSelection()
                return true
            })
        }
        return actions
    }

    override func accessibilityValue() -> String? {
        // Read the caret line's checklist state with the text.
        let style = currentBlockStyle
        guard style.kind == .check, let value = super.accessibilityValue() else { return super.accessibilityValue() }
        return (style.done ? "Checked item. " : "Unchecked item. ") + value
    }

    private func indentLists(by delta: Int) -> Bool {
        let lines = selectedLineRanges
        guard lines.allSatisfy({ blockStyle(at: $0.location).isList }) else { return false }
        setBlockStyle({ $0.with(indent: min(NoteEditorMetrics.maxIndent, max(0, $0.indent + delta))) }, lines: lines)
        return true
    }

    // MARK: Run styles

    enum RunStyle { case bold, italic, underline, strikethrough }

    /// Toggles a run style over the selection, or for what is typed next.
    func toggle(_ style: RunStyle) {
        let key: NSAttributedString.Key
        let on: Any
        switch style {
        case .bold: key = .noteBold; on = true
        case .italic: key = .noteItalic; on = true
        case .underline: key = .underlineStyle; on = NSUnderlineStyle.single.rawValue
        case .strikethrough: key = .strikethroughStyle; on = NSUnderlineStyle.single.rawValue
        }
        guard isEditable else { return }
        let selection = selectedRange()
        guard selection.length > 0, let storage = textStorage else {
            var typing = typingAttributes
            typing[key] = isOn(typing[key]) ? nil : on
            typingAttributes = restyledTyping(typing)
            return
        }
        var allOn = true
        storage.enumerateAttribute(key, in: selection) { value, _, stop in
            if !isOn(value) { allOn = false; stop.pointee = true }
        }
        guard shouldChangeText(in: selection, replacementString: nil) else { return }
        if allOn { storage.removeAttribute(key, range: selection) } else { storage.addAttribute(key, value: on, range: selection) }
        didChangeText()
        onTextChange?()
    }

    /// Whether the caret or the whole selection has `style` (toolbar state).
    func has(_ style: RunStyle) -> Bool {
        let key: NSAttributedString.Key = switch style {
        case .bold: .noteBold
        case .italic: .noteItalic
        case .underline: .underlineStyle
        case .strikethrough: .strikethroughStyle
        }
        let selection = selectedRange()
        guard selection.length > 0, let storage = textStorage else { return isOn(typingAttributes[key]) }
        var allOn = true
        storage.enumerateAttribute(key, in: selection) { value, _, stop in
            if !isOn(value) { allOn = false; stop.pointee = true }
        }
        return allOn
    }

    private func isOn(_ value: Any?) -> Bool {
        if let number = value as? NSNumber { return number.intValue != 0 }
        if let bool = value as? Bool { return bool }
        return value != nil
    }

    override func underline(_ sender: Any?) { toggle(.underline) }

    override func performKeyEquivalent(with event: NSEvent) -> Bool {
        guard window?.firstResponder === self else { return super.performKeyEquivalent(with: event) }
        let flags = event.modifierFlags.intersection(.deviceIndependentFlagsMask)
        let key = event.charactersIgnoringModifiers?.lowercased()
        switch (flags, key) {
        case ([.command], "b"): toggle(.bold); return true
        case ([.command], "i"): toggle(.italic); return true
        case ([.command], "u"): toggle(.underline); return true
        case ([.command, .shift], "x"): toggle(.strikethrough); return true
        case ([.command, .shift], "l"): toggleChecklist(); return true
        case ([.command, .shift], "u"): toggleDoneForSelection(); return true
        default: return super.performKeyEquivalent(with: event)
        }
    }

    // MARK: Typing

    override func insertText(_ insertString: Any, replacementRange: NSRange) {
        typingAttributes[.link] = nil
        // One kind of line break in the text: \n.
        if let text = insertString as? String, text.contains(where: { "\r\u{2028}\u{2029}".contains($0) }) {
            insertText(Self.normalizingBreaks(text), replacementRange: replacementRange)
            return
        }
        if let text = insertString as? String, text == " ", replacementRange.location == NSNotFound,
           applyMarkdownShortcut() {
            return
        }
        super.insertText(insertString, replacementRange: replacementRange)
    }

    /// The line whose shortcut was just undone: its next space stays a space.
    private var suppressedShortcutLine: Int?

    private func suppressShortcut(onLineAt location: Int) {
        suppressedShortcutLine = location
        undoManager?.registerUndo(withTarget: self) { view in view.suppressedShortcutLine = nil }
    }

    /// `# `, `- `, `* `, `1. `, `[] ` at the start of a body line.
    private func applyMarkdownShortcut() -> Bool {
        let selection = selectedRange()
        guard selection.length == 0, let storage = textStorage else { return false }
        let line = nsString.lineRange(for: NSRange(location: selection.location, length: 0))
        if let suppressed = suppressedShortcutLine {
            suppressedShortcutLine = nil
            if suppressed == line.location { return false }
        }
        let prefixRange = NSRange(location: line.location, length: selection.location - line.location)
        guard prefixRange.length > 0, prefixRange.length <= 3 else { return false }
        let style = blockStyle(at: line.location)
        guard style.kind == .body else { return false }
        let prefix = nsString.substring(with: prefixRange)
        let kind: NoteBlockStyle.Kind
        switch prefix {
        case "#": kind = .heading
        case "##": kind = .subheading
        case "-", "*": kind = .bullet
        case "[]": kind = .check
        default:
            guard prefix.hasSuffix("."), Int(prefix.dropLast()) != nil else { return false }
            kind = .numbered
        }
        guard shouldChangeText(in: prefixRange, replacementString: "") else { return false }
        storage.replaceCharacters(in: prefixRange, with: "")
        didChangeText()
        // ⌘Z brings the typed "-" back, and then the next space on that line
        // is just a space: a literal "- " can always be typed.
        undoManager?.registerUndo(withTarget: self) { view in view.suppressShortcut(onLineAt: line.location) }
        let newLine = nsString.lineRange(for: NSRange(location: line.location, length: 0))
        setBlockStyle({ _ in NoteBlockStyle(kind) }, lines: [newLine])
        setSelectedRange(NSRange(location: line.location, length: 0))
        typingAttributes = typingAttributesForCaret()
        undoManager?.setActionName("Format")
        return true
    }

    override func insertNewline(_ sender: Any?) {
        let selection = selectedRange()
        let style = blockStyle(at: selection.location)
        let line = nsString.lineRange(for: NSRange(location: selection.location, length: 0))
        var content = line
        if content.length > 0, nsString.character(at: NSMaxRange(content) - 1) == 0x0A { content.length -= 1 }
        // Return on an empty list item ends the list.
        if style.isList, content.length == 0 || nsString.substring(with: content).trimmingCharacters(in: .whitespaces).isEmpty {
            setBlockStyle({ _ in .body }, lines: [line])
            return
        }
        super.insertNewline(sender)
        let next: NoteBlockStyle? = switch style.kind {
        case .title, .heading, .subheading: .body
        case .check: style.with(done: false)
        default: nil
        }
        if let next {
            let newLine = nsString.lineRange(for: NSRange(location: selectedRange().location, length: 0))
            setBlockStyle({ _ in next }, lines: [newLine])
        }
        typingAttributes = typingAttributesForCaret()
    }

    static func normalizingBreaks(_ text: String) -> String {
        text.replacingOccurrences(of: "\r\n", with: "\n").replacingOccurrences(of: "\r", with: "\n")
            .replacingOccurrences(of: "\u{2028}", with: "\n").replacingOccurrences(of: "\u{2029}", with: "\n")
    }

    /// Shift-Return makes a new line too (no line separators in notes).
    override func insertLineBreak(_ sender: Any?) { insertNewline(sender) }

    override func insertTab(_ sender: Any?) {
        if !indentLists(by: 1) { super.insertTab(sender) }
    }

    override func insertBacktab(_ sender: Any?) {
        if !indentLists(by: -1) { super.insertBacktab(sender) }
    }

    override func deleteBackward(_ sender: Any?) {
        // Backspace at the start of a list item or heading makes it body first.
        let selection = selectedRange()
        let line = nsString.lineRange(for: NSRange(location: selection.location, length: 0))
        let style = blockStyle(at: selection.location)
        if selection.length == 0, selection.location == line.location, style.isList || (style.isHeading && style.kind != .title) {
            if style.indent > 0 {
                setBlockStyle({ $0.with(indent: $0.indent - 1) }, lines: [line])
            } else {
                setBlockStyle({ _ in .body }, lines: [line])
            }
            return
        }
        super.deleteBackward(sender)
    }

    override func setSelectedRanges(_ ranges: [NSValue], affinity: NSSelectionAffinity, stillSelecting: Bool) {
        super.setSelectedRanges(ranges, affinity: affinity, stillSelecting: stillSelecting)
        if !stillSelecting, !isRestyling {
            typingAttributes = typingAttributesForCaret()
        }
    }

    // MARK: Clicks

    override func mouseDown(with event: NSEvent) {
        let point = convert(event.locationInWindow, from: nil)
        let containerPoint = CGPoint(x: point.x - textContainerOrigin.x, y: point.y - textContainerOrigin.y)
        if let fragment = textLayoutManager?.textLayoutFragment(for: containerPoint) as? NoteParagraphFragment,
           let box = fragment.checkBoxRect,
           box.insetBy(dx: -NoteEditorMetrics.checkHitSlop, dy: -NoteEditorMetrics.checkHitSlop)
               .contains(CGPoint(x: containerPoint.x - fragment.layoutFragmentFrame.minX,
                                 y: containerPoint.y - fragment.layoutFragmentFrame.minY)),
           let manager = textLayoutManager {
            let location = manager.offset(from: manager.documentRange.location, to: fragment.rangeInElement.location)
            toggleDone(lineAt: location)
            return
        }
        if let time = chipTime(at: point) {
            onTimeChipClick?(time)
            return
        }
        super.mouseDown(with: event)
    }

    /// The time of a transcript chip or marker under `point`, if any.
    func chipTime(at point: CGPoint) -> TimeInterval? {
        guard onTimeChipClick != nil, let storage = textStorage, storage.length > 0 else { return nil }
        let index = characterIndexForInsertion(at: point)
        for candidate in [index, index - 1] where candidate >= 0 && candidate < storage.length {
            guard let attachment = storage.attribute(.attachment, at: candidate, effectiveRange: nil) as? NoteBlockAttachment,
                  let rect = firstRect(forCharacterRange: NSRange(location: candidate, length: 1), actualRange: nil) as CGRect?,
                  let window, convert(window.convertFromScreen(rect), from: nil).insetBy(dx: -NoteEditorMetrics.checkHitSlop, dy: -NoteEditorMetrics.checkHitSlop).contains(point)
            else { continue }
            switch attachment.block {
            case .transcriptAnchor: return attachment.time
            case .marker(let time): return time
            default: continue
            }
        }
        return nil
    }

    // MARK: Paste and copy

    /// What paste and drop accept; anything else is refused rather than
    /// handed to AppKit (which would keep formats a note cannot store).
    static let acceptedPasteboardTypes: [NSPasteboard.PasteboardType] = [
        blocksPasteboardType, .fileURL, .png, .tiff, .rtfd, .rtf, .html, .string,
    ]

    override var readablePasteboardTypes: [NSPasteboard.PasteboardType] { Self.acceptedPasteboardTypes }

    override var acceptableDragTypes: [NSPasteboard.PasteboardType] { Self.acceptedPasteboardTypes }

    override var writablePasteboardTypes: [NSPasteboard.PasteboardType] {
        [Self.blocksPasteboardType, .string]
    }

    override func writeSelection(to pboard: NSPasteboard, type: NSPasteboard.PasteboardType) -> Bool {
        guard let storage = textStorage else { return false }
        let selection = selectedRange()
        guard selection.length > 0 else { return false }
        let piece = storage.attributedSubstring(from: selection)
        switch type {
        case Self.blocksPasteboardType:
            let style = blockStyle(at: NSMaxRange(selection))
            let payload = NoteBlocksPayload(noteID: noteID, document: NoteTextMapping.document(from: piece, lastLineStyle: style))
            guard let data = try? JSONEncoder().encode(payload) else { return false }
            pboard.setData(data, forType: type)
            return true
        case .string:
            pboard.setString(piece.string.replacingOccurrences(of: "\u{FFFC}", with: ""), forType: .string)
            return true
        default:
            return false
        }
    }

    override func readSelection(from pboard: NSPasteboard, type: NSPasteboard.PasteboardType) -> Bool {
        guard isEditable, let incoming = pasted(from: pboard, type: type), incoming.length > 0 else { return false }
        let selection = selectedRange()
        guard shouldChangeText(in: selection, replacementString: incoming.string) else { return false }
        textStorage?.replaceCharacters(in: selection, with: incoming)
        didChangeText()
        setSelectedRange(NSRange(location: selection.location + incoming.length, length: 0))
        return true
    }

    /// The text a paste of `type` inserts, in the editor's canonical form;
    /// nil refuses the paste.
    private func pasted(from pboard: NSPasteboard, type: NSPasteboard.PasteboardType) -> NSAttributedString? {
        let base = blockStyle(at: selectedRange().location)
        switch type {
        case Self.blocksPasteboardType:
            guard let data = pboard.data(forType: type) else { return nil }
            return objects(blocksFromPasteboard(data))
        case .fileURL:
            let urls = pboard.readObjects(forClasses: [NSURL.self], options: [.urlReadingFileURLsOnly: true]) as? [URL] ?? []
            let blocks = urls.compactMap { importFile?($0) }
            return blocks.isEmpty ? nil : objects(NoteDocument(blocks: blocks))
        case .png, .tiff:
            guard let image = imageData(from: pboard, type: type), let id = importImage?(image.data, image.fileExtension) else {
                return nil
            }
            return objects(NoteDocument(blocks: [.image(attachmentID: id)]))
        case .rtfd, .rtf, .html, .string:
            return NotePasteSanitizer.sanitize(pboard, type: type, base: base, importImage: importImage,
                                               context: mappingContext)
        default:
            return nil
        }
    }

    /// Blocks copied from a note. Attachments from another note are copied
    /// into this one; ids that are not plain file names are dropped.
    private func blocksFromPasteboard(_ data: Data) -> NoteDocument {
        let payload = (try? JSONDecoder().decode(NoteBlocksPayload.self, from: data))
            ?? NoteBlocksPayload(noteID: nil, document: NoteDocument.decode(data))
        var document = payload.document
        document.blocks = document.blocks.compactMap { block in
            switch block {
            case .image(let id), .audio(let id):
                guard NoteAttachments.isValidID(id) else { return nil }
                guard let source = payload.noteID, source != noteID else { return block }
                guard let copied = copyAttachment?(id, source) else { return nil }
                if case .image = block { return .image(attachmentID: copied) }
                return .audio(attachmentID: copied)
            default:
                return block
            }
        }
        return document
    }

    private func objects(_ document: NoteDocument) -> NSAttributedString {
        NoteTextMapping.attributedString(from: document, context: mappingContext).text
    }

    private func imageData(from pboard: NSPasteboard, type: NSPasteboard.PasteboardType) -> (data: Data, fileExtension: String)? {
        if type == .png, let data = pboard.data(forType: .png) { return (data, "png") }
        if type == .tiff, let data = pboard.data(forType: .tiff),
           let png = NSBitmapImageRep(data: data)?.representation(using: .png, properties: [:]) {
            return (png, "png")
        }
        return nil
    }

    // MARK: Dictation phrases

    /// Where each pending phrase will land, in start order. Positions move
    /// with every edit (see `shiftPendingPhrases`); nothing is put in the
    /// text, so undo and saving never see a pending phrase.
    private(set) var pendingPhrases: [(id: UUID, location: Int)] = []
    /// Set while a finished phrase is inserted: its spot, and the phrases
    /// started after it (only those move after its text when waiting there).
    private var insertingPhraseAt: Int?
    private var phrasesStartedLater: Set<UUID> = []

    /// Holds the caret's place for a phrase; an accent-tinted marker shows it.
    func beginPhrase(_ id: UUID) {
        pendingPhrases.removeAll { $0.id == id }
        pendingPhrases.append((id, selectedRange().location))
        needsDisplay = true
    }

    /// Inserts the phrase's text where it began, joined to its neighbours
    /// (`PhraseJoiner`), as one undoable change. A phrase that was never
    /// begun lands at the caret. Returns whether text was inserted.
    @discardableResult
    func finishPhrase(_ id: UUID, text: String) -> Bool {
        guard let storage = textStorage else { return false }
        if hasMarkedText() { unmarkText() }
        let index = pendingPhrases.firstIndex { $0.id == id }
        let location = min(index.map { pendingPhrases[$0].location } ?? selectedRange().location, nsString.length)
        phrasesStartedLater = Set(index.map { pendingPhrases[($0 + 1)...].map(\.id) } ?? [])
        pendingPhrases.removeAll { $0.id == id }
        defer { needsDisplay = true }
        let line = nsString.lineRange(for: NSRange(location: location, length: 0))
        var lineEnd = NSMaxRange(line)
        if lineEnd > location, nsString.character(at: lineEnd - 1) == 0x0A { lineEnd -= 1 }
        let preceding = nsString.substring(with: NSRange(location: line.location, length: location - line.location))
        let following = nsString.substring(with: NSRange(location: location, length: lineEnd - location))
        let joined = PhraseJoiner.text(for: text, preceding: preceding, following: following)
        guard !joined.isEmpty else { return false }
        let selection = selectedRange()
        let insertAt = NSRange(location: location, length: 0)
        guard shouldChangeText(in: insertAt, replacementString: joined) else { return false }
        var attributes = NoteEditorStyle.typingAttributes(blockStyle(at: location))
        if location > 0, nsString.character(at: location - 1) != 0x0A {
            // Keep the run style of the text it follows (bold stays bold).
            let previous = storage.attributes(at: location - 1, effectiveRange: nil)
            for key in [NSAttributedString.Key.noteBold, .noteItalic, .underlineStyle, .strikethroughStyle] {
                attributes[key] = previous[key]
            }
        }
        insertingPhraseAt = location
        storage.replaceCharacters(in: insertAt, with: NSAttributedString(string: joined, attributes: attributes))
        insertingPhraseAt = nil
        didChangeText()
        // The caret stays with the text it was in.
        let length = (joined as NSString).length
        var caret = selection
        if caret.location >= location { caret.location += length }
        setSelectedRange(NSRange(location: min(caret.location, nsString.length), length: caret.length))
        return true
    }

    /// Takes over another view's waiting phrases (same text, same places):
    /// they finish into the note they began in after it is switched away.
    func adoptPendingPhrases(_ phrases: [(id: UUID, location: Int)]) {
        let length = nsString.length
        pendingPhrases = phrases.map { ($0.id, min($0.location, length)) }
    }

    /// Forgets a phrase that had no words.
    func dropPhrase(_ id: UUID) {
        pendingPhrases.removeAll { $0.id == id }
        needsDisplay = true
    }

    /// Forgets every pending phrase (dictation ended).
    func dropAllPhrases() {
        pendingPhrases.removeAll()
        needsDisplay = true
    }

    var pendingPhraseIDs: [UUID] { pendingPhrases.map(\.id) }

    /// Moves pending phrase positions for an edit that replaced
    /// `oldLength` characters at `location` with `newLength`. Text typed at a
    /// phrase's spot goes after the phrase (it was spoken first).
    private func shiftPendingPhrases(location: Int, oldLength: Int, newLength: Int) {
        guard !pendingPhrases.isEmpty else { return }
        let delta = newLength - oldLength
        for index in pendingPhrases.indices {
            let p = pendingPhrases[index].location
            if let at = insertingPhraseAt, p == at {
                // Started later: lands after this phrase. Started earlier:
                // stays before it.
                if phrasesStartedLater.contains(pendingPhrases[index].id) {
                    pendingPhrases[index].location = p + newLength
                }
            } else if p > location, p >= location + oldLength {
                pendingPhrases[index].location = p + delta
            } else if p > location {
                pendingPhrases[index].location = location
            }
        }
    }

    override func draw(_ dirtyRect: NSRect) {
        super.draw(dirtyRect)
        guard !pendingPhrases.isEmpty, let manager = textLayoutManager else { return }
        NoteEditorStyle.accent.setFill()
        for phrase in pendingPhrases {
            guard let start = manager.location(manager.documentRange.location, offsetBy: phrase.location),
                  let range = NSTextRange(location: start) as NSTextRange? else { continue }
            manager.enumerateTextSegments(in: range, type: .standard, options: [.rangeNotRequired]) { _, frame, _, _ in
                let height = min(frame.height, NoteEditorMetrics.pendingPhraseHeight)
                let rect = CGRect(x: frame.minX + textContainerOrigin.x + NoteEditorMetrics.pendingPhraseGap,
                                  y: frame.midY + textContainerOrigin.y - height / 2,
                                  width: NoteEditorMetrics.pendingPhraseMinWidth, height: height)
                NSBezierPath(roundedRect: rect, xRadius: height / 2, yRadius: height / 2).fill()
                return false
            }
        }
    }

    /// Selects and scrolls to the time chip of `segmentID` (a ⌘K transcript hit).
    func reveal(segmentID: UUID) {
        guard let storage = textStorage else { return }
        var found: NSRange?
        storage.enumerateAttribute(.attachment, in: NSRange(location: 0, length: storage.length)) { value, range, stop in
            if let attachment = value as? NoteBlockAttachment, attachment.block == .transcriptAnchor(segmentID: segmentID) {
                found = range
                stop.pointee = true
            }
        }
        guard let found else { return }
        setSelectedRange(NSRange(location: found.location, length: 0))
        scrollRangeToVisible(found)
    }

    // MARK: Meeting markers

    /// Inserts a "Mark moment" line after the caret's line (meeting notes).
    func insertMarker(time: TimeInterval) {
        insertObjectLine(.marker(time: time))
    }

    /// Inserts an image or audio line after the caret's line.
    func insertAttachment(_ block: NoteDocument.Block) {
        insertObjectLine(block)
    }

    private func insertObjectLine(_ block: NoteDocument.Block) {
        guard let storage = textStorage else { return }
        let line = nsString.lineRange(for: NSRange(location: selectedRange().location, length: 0))
        let piece = NSMutableAttributedString(
            attributedString: NoteTextMapping.attributedString(from: NoteDocument(blocks: [block]), context: mappingContext).text)
        let endsWithBreak = line.length > 0 && nsString.character(at: NSMaxRange(line) - 1) == 0x0A
        let insertAt = NSMaxRange(line)
        let breakAttributes = NoteEditorStyle.typingAttributes(.body)
        if endsWithBreak {
            piece.append(NSAttributedString(string: "\n", attributes: breakAttributes))
        } else {
            piece.insert(NSAttributedString(string: "\n", attributes: line.length > 0
                ? storage.attributes(at: line.location, effectiveRange: nil) : breakAttributes), at: 0)
        }
        guard shouldChangeText(in: NSRange(location: insertAt, length: 0), replacementString: piece.string) else { return }
        storage.replaceCharacters(in: NSRange(location: insertAt, length: 0), with: piece)
        didChangeText()
        setSelectedRange(NSRange(location: insertAt + piece.length, length: 0))
        onTextChange?()
    }

    // MARK: Restyling

    nonisolated func textStorage(_ textStorage: NSTextStorage, willProcessEditing editedMask: NSTextStorageEditActions,
                                 range editedRange: NSRange, changeInLength delta: Int) {
        // Text storage edits arrive on the main thread, from this view.
        nonisolated(unsafe) let textStorage = textStorage
        MainActor.assumeIsolated {
            if editedMask.contains(.editedCharacters), !isLoading {
                shiftPendingPhrases(location: editedRange.location, oldLength: editedRange.length - delta,
                                    newLength: editedRange.length)
            }
            guard !isRestyling else { return }
            isRestyling = true
            NoteTextMapping.restyle(textStorage, around: editedRange, clip: true)
            isRestyling = false
            pendingRestyle = pendingRestyle.map { NSUnionRange($0, editedRange) } ?? editedRange
        }
    }

    /// Lines edited since the last full restyle.
    private var pendingRestyle: NSRange?

    /// Restyles whole lines around the last edits (margins depend on the
    /// next line), as an attribute pass of its own; the selection stays.
    private func restyleEditedLines() {
        guard let range = pendingRestyle, let storage = textStorage, !isRestyling else { return }
        pendingRestyle = nil
        let selection = selectedRanges
        isRestyling = true
        let clamped = NSRange(location: min(range.location, storage.length),
                              length: max(0, min(NSMaxRange(range), storage.length) - min(range.location, storage.length)))
        storage.beginEditing()
        NoteTextMapping.restyle(storage, around: clamped)
        storage.endEditing()
        isRestyling = false
        selectedRanges = selection
    }

    override func didChangeText() {
        super.didChangeText()
        restyleEditedLines()
        if selectedRange().location >= nsString.length, nsString.length == 0 || nsString.character(at: nsString.length - 1) == 0x0A {
            emptyLastLineStyle = (typingAttributes[.noteBlock] as? NoteBlockStyle) ?? emptyLastLineStyle
        }
        invalidateListLayout()
        onTextChange?()
    }

    /// Numbers and boxes are drawn by the fragments; relayout after edits so
    /// a numbered list renumbers.
    private func invalidateListLayout() {
        guard let manager = textLayoutManager else { return }
        manager.invalidateLayout(for: manager.documentRange)
    }

    // MARK: NSTextLayoutManagerDelegate

    nonisolated func textLayoutManager(_ textLayoutManager: NSTextLayoutManager,
                                       textLayoutFragmentFor location: NSTextLocation,
                                       in textElement: NSTextElement) -> NSTextLayoutFragment {
        guard let paragraph = textElement as? NSTextParagraph, paragraph.attributedString.length > 0,
              let style = paragraph.attributedString.attribute(.noteBlock, at: 0, effectiveRange: nil) as? NoteBlockStyle,
              style.isList else {
            return NSTextLayoutFragment(textElement: textElement, range: textElement.elementRange)
        }
        var ordinal = 1
        if style.kind == .numbered, let range = textElement.elementRange {
            let offset = textLayoutManager.offset(from: textLayoutManager.documentRange.location, to: range.location)
            ordinal = MainActor.assumeIsolated {
                textStorage.map { Self.ordinal(in: $0, lineStart: offset, style: style) } ?? 1
            }
        }
        return NoteParagraphFragment(textElement: textElement, range: textElement.elementRange, style: style, ordinal: ordinal)
    }

    /// Counts the numbered lines of the same indent just before `lineStart`
    /// (deeper lines in between do not break the count).
    static func ordinal(in text: NSAttributedString, lineStart: Int, style: NoteBlockStyle) -> Int {
        let string = text.string as NSString
        var count = 1
        var location = lineStart
        while location > 0 {
            let previous = string.lineRange(for: NSRange(location: location - 1, length: 0))
            guard let other = text.attribute(.noteBlock, at: previous.location, effectiveRange: nil) as? NoteBlockStyle,
                  other.isList else { break }
            if other.indent < style.indent { break }
            if other.indent == style.indent {
                guard other.kind == .numbered else { break }
                count += 1
            }
            location = previous.location
        }
        return count
    }
}

/// Blocks on the pasteboard, with the note they came from (so a paste into
/// another note can copy their attachments).
struct NoteBlocksPayload: Codable {
    var noteID: UUID?
    var document: NoteDocument
}

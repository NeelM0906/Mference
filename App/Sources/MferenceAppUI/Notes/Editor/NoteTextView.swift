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
    /// Stores pasted or dropped image data as an attachment; returns its id.
    var importImage: ((Data, String) -> String?)?
    var mappingContext = NoteTextMapping.Context()

    /// The block style of an empty last line (it has no character to carry it).
    var emptyLastLineStyle: NoteBlockStyle = .title

    private var isRestyling = false

    /// The date line centered above the title; it scrolls with the text.
    var dateLine = "" {
        didSet {
            dateLabel.stringValue = dateLine
            dateLabel.isHidden = dateLine.isEmpty
            updateInsets()
        }
    }

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
        dateLabel.frame = CGRect(x: 0, y: NoteEditorMetrics.topInset, width: bounds.width,
                                 height: NoteEditorMetrics.dateLineHeight)
        let dateHeight = dateLine.isEmpty ? 0 : NoteEditorMetrics.dateLineHeight + NoteEditorMetrics.titleSpacingBefore
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
        textStorage?.setAttributedString(text)
        isRestyling = false
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
        guard let storage = textStorage else { return }
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
        setBlockStyle({ $0.kind == .check ? $0.with(done: !$0.done) : $0 }, lines: [line])
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
        default: return super.performKeyEquivalent(with: event)
        }
    }

    // MARK: Typing

    override func insertText(_ insertString: Any, replacementRange: NSRange) {
        typingAttributes[.link] = nil
        if let text = insertString as? String, text == " ", replacementRange.location == NSNotFound,
           applyMarkdownShortcut() {
            return
        }
        super.insertText(insertString, replacementRange: replacementRange)
    }

    /// `# `, `- `, `* `, `1. `, `[] ` at the start of a body line.
    private func applyMarkdownShortcut() -> Bool {
        let selection = selectedRange()
        guard selection.length == 0, let storage = textStorage else { return false }
        let line = nsString.lineRange(for: NSRange(location: selection.location, length: 0))
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
        let newLine = nsString.lineRange(for: NSRange(location: line.location, length: 0))
        setBlockStyle({ _ in NoteBlockStyle(kind) }, lines: [newLine])
        setSelectedRange(NSRange(location: line.location, length: 0))
        typingAttributes = typingAttributesForCaret()
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
        super.mouseDown(with: event)
    }

    // MARK: Paste and copy

    override var readablePasteboardTypes: [NSPasteboard.PasteboardType] {
        [Self.blocksPasteboardType] + super.readablePasteboardTypes
    }

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
            pboard.setData(NoteTextMapping.document(from: piece, lastLineStyle: style).encoded(), forType: type)
            return true
        case .string:
            pboard.setString(piece.string.replacingOccurrences(of: "\u{FFFC}", with: ""), forType: .string)
            return true
        default:
            return super.writeSelection(to: pboard, type: type)
        }
    }

    override func readSelection(from pboard: NSPasteboard, type: NSPasteboard.PasteboardType) -> Bool {
        let incoming: NSAttributedString?
        if type == Self.blocksPasteboardType, let data = pboard.data(forType: type) {
            incoming = NoteTextMapping.attributedString(from: NoteDocument.decode(data), context: mappingContext).text
        } else if let image = imageData(from: pboard, type: type), let id = importImage?(image.data, image.fileExtension) {
            incoming = NoteTextMapping.attributedString(
                from: NoteDocument(blocks: [.image(attachmentID: id)]), context: mappingContext).text
        } else if [.rtf, .rtfd, .html, .string].contains(type) {
            incoming = NotePasteSanitizer.sanitize(pboard, type: type, base: blockStyle(at: selectedRange().location))
        } else {
            incoming = nil
        }
        guard let incoming, incoming.length > 0 else { return super.readSelection(from: pboard, type: type) }
        let selection = selectedRange()
        guard shouldChangeText(in: selection, replacementString: incoming.string) else { return false }
        textStorage?.replaceCharacters(in: selection, with: incoming)
        didChangeText()
        setSelectedRange(NSRange(location: selection.location + incoming.length, length: 0))
        return true
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
    /// Set while a finished phrase is inserted, so later phrases waiting at
    /// the same spot move after it.
    private var insertingPhraseAt: Int?

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
        let location = min(pendingPhrases.first { $0.id == id }?.location ?? selectedRange().location, nsString.length)
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
                pendingPhrases[index].location = p + newLength
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
            if editedMask.contains(.editedCharacters) {
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

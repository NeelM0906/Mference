import AppKit
import MferenceAppCore
import SwiftUI

/// The note's text, as an AppKit text view in a scroll view (TextKit 2).
/// `NoteEditorController` owns what it shows and saves.
struct NoteEditorView: NSViewRepresentable {
    let controller: NoteEditorController
    /// The date line above the title ("19 November 2026 at 11:40").
    var dateLine: String

    func makeCoordinator() -> Coordinator { Coordinator(controller: controller) }

    func makeNSView(context: Context) -> NSScrollView {
        let scroll = NSScrollView()
        scroll.hasVerticalScroller = true
        scroll.autohidesScrollers = true
        scroll.drawsBackground = false
        scroll.borderType = .noBorder
        let textView = NoteTextView(usingTextLayoutManager: true)
        textView.configure()
        textView.minSize = .zero
        textView.maxSize = NSSize(width: CGFloat.greatestFiniteMagnitude, height: CGFloat.greatestFiniteMagnitude)
        textView.isVerticallyResizable = true
        textView.isHorizontallyResizable = false
        textView.autoresizingMask = [.width]
        textView.textContainer?.widthTracksTextView = true
        textView.dateLine = dateLine
        textView.delegate = context.coordinator
        scroll.documentView = textView
        controller.attach(textView)
        return scroll
    }

    func updateNSView(_ scroll: NSScrollView, context: Context) {
        guard let textView = scroll.documentView as? NoteTextView else { return }
        if textView.dateLine != dateLine { textView.dateLine = dateLine }
        if controller.textView !== textView { controller.attach(textView) }
    }

    final class Coordinator: NSObject, NSTextViewDelegate {
        let controller: NoteEditorController
        init(controller: NoteEditorController) { self.controller = controller }

        func textViewDidChangeSelection(_ notification: Notification) {
            controller.selectionDidChange()
        }

        func textView(_ textView: NSTextView, clickedOnLink link: Any, at charIndex: Int) -> Bool {
            // Links open in the default browser (spec 4.1).
            let url = (link as? URL) ?? (link as? String).flatMap(URL.init(string:))
            guard let url, ["http", "https", "mailto"].contains(url.scheme?.lowercased() ?? "") else { return true }
            NSWorkspace.shared.open(url)
            return true
        }
    }
}

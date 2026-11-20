import AppKit
import Carbon.HIToolbox
import MferenceAppCore
import MferenceDesign
import SwiftUI

/// The quick find panel (canvas board 1): a floating panel over any app,
/// opened with ⌥Space. Word matches show as you type; meaning matches merge
/// in when you pause. Return opens, ⌘Return shows in Finder, ⌘J asks Chat,
/// Escape closes.
public final class FindPanelController {
    public static let shared = FindPanelController()

    private var panel: FindFloatingPanel?
    private var hotKey: GlobalHotKey?
    private var find: FindService?
    private var model: FindViewModel?
    private var askAbout: (URL) -> Void = { _ in }
    private var openFind: () -> Void = {}

    /// Wires the panel to Find and registers ⌥Space. Call once at launch.
    public func install(find: FindService, askAbout: @escaping (URL) -> Void, openFind: @escaping () -> Void) {
        self.find = find
        self.askAbout = askAbout
        self.openFind = openFind
        guard hotKey == nil else { return }
        hotKey = GlobalHotKey(keyCode: UInt32(kVK_Space), modifiers: UInt32(optionKey)) { [weak self] in
            self?.toggle()
        }
    }

    public func toggle() {
        if let panel, panel.isVisible {
            panel.orderOut(nil)
        } else {
            show()
        }
    }

    public func show() {
        guard let find else { return }
        let panel = self.panel ?? makePanel(find: find)
        self.panel = panel
        let screen = NSScreen.screens.first { NSMouseInRect(NSEvent.mouseLocation, $0.frame, false) } ?? NSScreen.main
        if let frame = screen?.visibleFrame {
            let size = FindMetrics.panelSize
            panel.setFrame(NSRect(x: frame.midX - size.width / 2, y: frame.maxY - size.height - frame.height * 0.12,
                                  width: size.width, height: size.height), display: true)
        }
        panel.makeKeyAndOrderFront(nil)
    }

    public func close() {
        panel?.orderOut(nil)
    }

    private func makePanel(find: FindService) -> FindFloatingPanel {
        let model = FindViewModel(find: find)
        self.model = model
        let panel = FindFloatingPanel(contentRect: NSRect(origin: .zero, size: FindMetrics.panelSize))
        let view = FindPanelView(model: model,
                                 askAbout: { [weak self] url in
                                     self?.close()
                                     NSApp.activate()
                                     self?.askAbout(url)
                                 },
                                 openFind: { [weak self] in
                                     self?.close()
                                     NSApp.activate()
                                     self?.openFind()
                                 },
                                 close: { [weak self] in self?.close() })
        panel.contentView = NSHostingView(rootView: view)
        return panel
    }
}

/// A borderless panel that takes the keyboard without activating Mference,
/// and hides when it loses focus.
final class FindFloatingPanel: NSPanel {
    init(contentRect: NSRect) {
        super.init(contentRect: contentRect, styleMask: [.nonactivatingPanel, .titled, .fullSizeContentView],
                   backing: .buffered, defer: true)
        titleVisibility = .hidden
        titlebarAppearsTransparent = true
        standardWindowButton(.closeButton)?.isHidden = true
        standardWindowButton(.miniaturizeButton)?.isHidden = true
        standardWindowButton(.zoomButton)?.isHidden = true
        isMovableByWindowBackground = true
        level = .floating
        collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .transient]
        hidesOnDeactivate = false
        isReleasedWhenClosed = false
        backgroundColor = .clear
    }

    override var canBecomeKey: Bool { true }

    override func resignKey() {
        super.resignKey()
        orderOut(nil)
    }

    override func cancelOperation(_ sender: Any?) {
        orderOut(nil)
    }
}

/// A system-wide shortcut through Carbon's hot keys (no Accessibility
/// permission needed).
final class GlobalHotKey {
    private var reference: EventHotKeyRef?
    private var handler: EventHandlerRef?
    private let action: () -> Void

    init(keyCode: UInt32, modifiers: UInt32, action: @escaping () -> Void) {
        self.action = action
        var type = EventTypeSpec(eventClass: OSType(kEventClassKeyboard), eventKind: UInt32(kEventHotKeyPressed))
        let context = Unmanaged.passUnretained(self).toOpaque()
        InstallEventHandler(GetApplicationEventTarget(), { _, _, context in
            guard let context else { return noErr }
            let hotKey = Unmanaged<GlobalHotKey>.fromOpaque(context).takeUnretainedValue()
            DispatchQueue.main.async { hotKey.action() }
            return noErr
        }, 1, &type, context, &handler)
        let id = EventHotKeyID(signature: OSType(0x4D46_4E44), id: 1) // "MFND"
        RegisterEventHotKey(keyCode, modifiers, id, GetApplicationEventTarget(), 0, &reference)
    }

    isolated deinit {
        if let reference { UnregisterEventHotKey(reference) }
        if let handler { RemoveEventHandler(handler) }
    }
}

/// The panel's content.
struct FindPanelView: View {
    @Bindable var model: FindViewModel
    let askAbout: (URL) -> Void
    let openFind: () -> Void
    let close: () -> Void
    @FocusState private var focused: Bool

    var body: some View {
        VStack(spacing: 0) {
            searchBar
            Rectangle().fill(MFColor.chromeDivider).frame(height: MFLayout.hairline)
            if model.find.settings.enabled {
                tabs
                Rectangle().fill(MFColor.chromeDivider).frame(height: MFLayout.hairline)
                HStack(spacing: 0) {
                    list.frame(width: FindMetrics.panelListWidth)
                    Rectangle().fill(MFColor.chromeDivider).frame(width: MFLayout.hairline)
                    preview
                }
                .frame(maxHeight: .infinity)
            } else {
                setupHint
            }
            Rectangle().fill(MFColor.chromeDivider).frame(height: MFLayout.hairline)
            footer
        }
        .background(MFColor.surface)
        .clipShape(RoundedRectangle(cornerRadius: MFRadius.xl))
        .onAppear { focused = true }
        .onKeyPress(.escape) {
            close()
            return .handled
        }
    }

    private var searchBar: some View {
        HStack(spacing: MFSpacing.m) {
            Image(systemName: "magnifyingglass").font(MFFont.title).foregroundStyle(MFColor.secondary)
                .accessibilityHidden(true)
            TextField("Describe a file", text: $model.query)
                .textFieldStyle(.plain)
                .font(MFFont.title)
                .focused($focused)
                .onSubmit { if let url = model.selected?.url { FindFiles.open(url); close() } }
                .onKeyPress(.downArrow) { model.moveSelection(by: 1); return .handled }
                .onKeyPress(.upArrow) { model.moveSelection(by: -1); return .handled }
                .accessibilityLabel("Find files by what is in them")
            if model.isSearching { ProgressView().controlSize(.small) }
            MFChip("On this Mac", systemImage: "lock", tone: .success)
            MFKeycap("esc")
        }
        .padding(.horizontal, MFSpacing.l + MFSpacing.xxs)
        .padding(.vertical, MFSpacing.comfortable)
    }

    private var tabs: some View {
        HStack(spacing: MFSpacing.xs) {
            ForEach(FindKindFilter.allCases) { filter in
                MFPill(model.results.query.isEmpty ? filter.title : "\(filter.title) \(model.count(filter))",
                       isSelected: model.kind == filter) { model.kind = filter }
            }
            Spacer()
            Text(model.summary).font(MFFont.caption).foregroundStyle(MFColor.tertiary)
        }
        .padding(.horizontal, MFSpacing.comfortable)
        .padding(.vertical, MFSpacing.s)
    }

    @ViewBuilder
    private var list: some View {
        if model.query.trimmingCharacters(in: .whitespaces).isEmpty {
            VStack(alignment: .leading, spacing: MFSpacing.s) {
                FindStatusBanner(status: model.find.status, modelState: model.find.modelState,
                                 mediaModelState: model.find.mediaModelState)
                Text("Describe what is in the file, in your own words.")
                    .font(MFFont.callout).foregroundStyle(MFColor.secondary)
            }
            .padding(MFSpacing.l)
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        } else if model.results.strong.isEmpty && !model.isSearching {
            FindNothingFound(model: model)
        } else {
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(spacing: MFSpacing.xxs) {
                        ForEach(model.visible) { result in
                            FindResultRow(result: result, isSelected: result.url == model.selected?.url)
                                .id(result.url)
                                .onTapGesture { model.selection = result.url }
                                .onTapGesture(count: 2) { FindFiles.open(result.url); close() }
                        }
                    }
                    .padding(MFSpacing.s)
                }
                .onChange(of: model.selection) { _, url in
                    if let url { proxy.scrollTo(url) }
                }
            }
        }
    }

    @ViewBuilder
    private var preview: some View {
        if let selected = model.selected {
            FindPreview(result: selected, askAbout: askAbout)
                .padding(MFSpacing.l)
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
                .background(MFColor.canvas)
        } else {
            MFColor.canvas
        }
    }

    private var setupHint: some View {
        VStack(spacing: MFSpacing.m) {
            Text("Find is off").font(MFFont.headline).foregroundStyle(MFColor.ink)
            Text("Turn it on to search your files by what is in them, on this Mac.")
                .font(MFFont.callout).foregroundStyle(MFColor.secondary)
            Button("Set up Find", action: openFind).buttonStyle(.mfPrimary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var footer: some View {
        HStack(spacing: MFSpacing.comfortable) {
            hint("↑ ↓", "move")
            hint("↩", "open")
            hint("⌘↩", "show in Finder")
            hint("⌘J", "ask in Chat")
            Spacer()
            Text("\(model.find.status.totalFiles.formatted()) files · nothing leaves this Mac")
                .font(MFFont.caption).foregroundStyle(MFColor.tertiary)
        }
        .padding(.horizontal, MFSpacing.l)
        .padding(.vertical, MFSpacing.s)
    }

    private func hint(_ keys: String, _ label: String) -> some View {
        HStack(spacing: MFSpacing.xs) {
            MFKeycap(keys)
            Text(label).font(MFFont.caption).foregroundStyle(MFColor.secondary)
        }
    }
}

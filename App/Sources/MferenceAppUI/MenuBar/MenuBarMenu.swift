import MferenceAppCore
import Observation
import SwiftUI

/// The menu-bar extra's state, kept current from the app's one engine. It
/// reads and asks; it never builds an engine of its own.
@Observable
public final class MenuBarModel {
    public private(set) var status: EngineServiceStatus
    public private(set) var options: [ModelOption] = []

    @ObservationIgnored let engine: any EngineServicing
    @ObservationIgnored let physicalMemory: UInt64

    public init(engine: any EngineServicing,
                physicalMemory: UInt64 = ProcessInfo.processInfo.physicalMemory) {
        self.engine = engine
        self.physicalMemory = physicalMemory
        status = engine.status
    }

    /// Follows the engine until the task is cancelled or the engine shuts down.
    public func run() async {
        await refresh()
        for await next in engine.statusUpdates() {
            let catalogChanged = next.catalogRevision != status.catalogRevision
            status = next
            if catalogChanged || options.isEmpty {
                await refresh()
            } else {
                options = options.map { option in
                    var option = option
                    option.availability = Self.availability(of: option, in: next)
                    return option
                }
            }
        }
    }

    /// Re-reads the model list.
    public func refresh() async {
        options = await engine.models()
        status = engine.status
    }

    /// Releases the resident model (after running and queued requests).
    public func unload() {
        Task { try? await engine.unload() }
    }

    /// Loads `model` again after a failed load.
    public func retry(_ model: AppModel) {
        Task { try? await engine.load(model) }
    }

    public func summary(entries: [ModelEntry], idleUnloadMinutes: Int) -> MenuBarSummary {
        MenuBarSummary(status: status, options: options, entries: entries,
                       idleUnloadMinutes: idleUnloadMinutes, physicalMemory: physicalMemory)
    }

    /// Availability from a newer status, for installed models; the list
    /// itself is re-read only when the catalog changes.
    static func availability(of option: ModelOption, in status: EngineServiceStatus) -> ModelAvailability {
        if case .notInstalled = option.availability { return option.availability }
        if status.residentModel == option.model, status.phase == .ready { return .loaded }
        if status.targetModel == option.model {
            if status.phase == .loading { return .loading }
            if status.phase == .unloading { return .unloading }
        }
        return .ready
    }
}

/// What the menu's items do; the app points them at its window, chat and
/// quit sequence.
public struct MenuBarActions {
    public var openWindow: () -> Void
    public var newChat: () -> Void
    public var newPrivateChat: () -> Void
    public var selectModel: (AppModel) -> Void
    public var openSettings: () -> Void
    public var quit: () -> Void

    public init(openWindow: @escaping () -> Void, newChat: @escaping () -> Void,
                newPrivateChat: @escaping () -> Void,
                selectModel: @escaping (AppModel) -> Void, openSettings: @escaping () -> Void,
                quit: @escaping () -> Void) {
        self.openWindow = openWindow
        self.newChat = newChat
        self.newPrivateChat = newPrivateChat
        self.selectModel = selectModel
        self.openSettings = openSettings
        self.quit = quit
    }
}

/// The menu-bar extra's menu (menu style, so macOS draws the material,
/// highlight and dark mode). Engine first, then actions, then settings.
public struct MenuBarMenu: View {
    let model: MenuBarModel
    let library: (any ModelLibraryModel)?
    @Bindable var settings: AppSettings
    let actions: MenuBarActions

    public init(model: MenuBarModel, library: (any ModelLibraryModel)?, settings: AppSettings,
                actions: MenuBarActions) {
        self.model = model
        self.library = library
        self.settings = settings
        self.actions = actions
    }

    public var body: some View {
        let summary = model.summary(entries: library?.entries ?? [], idleUnloadMinutes: settings.idleUnloadMinutes)
        Section {
            Text(summary.headline)
            ForEach(summary.details, id: \.self) { Text($0) }
            if let retry = summary.retryModel {
                Button("Retry Loading \(retry.shortName)") { model.retry(retry) }
            }
        }
        if !summary.downloads.isEmpty {
            Section("Downloads") {
                ForEach(summary.downloads, id: \.self) { Text($0) }
            }
        }
        Section {
            Button("Open Mference", action: actions.openWindow)
                .keyboardShortcut("0")
            Button("New Chat", action: actions.newChat)
                .keyboardShortcut("n")
            Button("New Private Chat", action: actions.newPrivateChat)
                .keyboardShortcut("n", modifiers: [.command, .shift])
        }
        Section {
            Menu("Model") {
                if let reason = summary.switchDisabledReason {
                    Text(reason)
                }
                ForEach(ModelGroup.allCases, id: \.self) { group in
                    let rows = summary.modelRows.filter { $0.group == group }
                    if !rows.isEmpty {
                        Section(MenuBarMenu.title(group)) {
                            ForEach(rows) { row in
                                Toggle(isOn: Binding(get: { row.isResident },
                                                     set: { _ in actions.selectModel(row.model) })) {
                                    Text(row.title)
                                    Text(row.detail)
                                }
                                .disabled(!row.isEnabled)
                            }
                        }
                    }
                }
            }
            Button(summary.unloadDisabledReason.map { "Unload Model (\($0.lowercased()))" } ?? "Unload Model") {
                model.unload()
            }
            .disabled(summary.unloadDisabledReason != nil)
        }
        Section {
            Toggle("Offline Mode", isOn: $settings.offlineMode)
            Button("Settings…", action: actions.openSettings)
                .keyboardShortcut(",")
        }
        Button("Quit Mference", action: actions.quit)
            .keyboardShortcut("q")
    }

    static func title(_ group: ModelGroup) -> String {
        switch group {
        case .recommended: "Recommended"
        case .alsoOnThisMac: "Also on this Mac"
        case .builtIntoMacOS: "Built into macOS"
        }
    }
}

/// The icon in the menu bar; it changes with the engine's state.
public struct MenuBarLabel: View {
    let model: MenuBarModel

    public init(model: MenuBarModel) { self.model = model }

    public var body: some View {
        let summary = model.summary(entries: [], idleUnloadMinutes: 0)
        Image(systemName: summary.symbol)
            .accessibilityLabel("Mference, \(summary.headline)")
    }
}

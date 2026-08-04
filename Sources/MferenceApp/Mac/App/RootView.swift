import AppKit
import MferenceAppCore
import MferenceMacPresentation
import SwiftUI

struct RootView: View {
    let model: AppModel
    @State private var conversationChromeHeight: CGFloat = 0
    @AppStorage("Mference.inspectorVisible")
    private var isInspectorVisible = true

    var body: some View {
        NavigationSplitView {
            TemplateSidebarView(model: model)
                .navigationSplitViewColumnWidth(min: 220, ideal: 260, max: 340)
        } detail: {
            primaryContent
                .frame(minWidth: 480, maxWidth: .infinity, maxHeight: .infinity)
                .toolbar {
                    ToolbarItem(placement: .principal) {
                        ModelStatusMenu(model: model)
                    }
                    ToolbarItem(placement: .primaryAction) {
                        Button("New Chat", systemImage: "square.and.pencil") {
                            model.createChat()
                        }
                        .disabled(model.isRunning)
                        .help("New chat (⌘N)")
                    }
                    ToolbarItem(placement: .primaryAction) {
                        Button("Inspector", systemImage: "sidebar.trailing") {
                            isInspectorVisible.toggle()
                        }
                        .help(isInspectorVisible ? "Hide inspector" : "Show inspector")
                    }
                }
                .inspector(isPresented: $isInspectorVisible) {
                    InspectorView(model: model)
                        .inspectorColumnWidth(CGFloat(AppChromeLayout.inspectorWidth))
                }
        }
        .frame(minHeight: CGFloat(AppChromeLayout.minimumHeight))
        .containerBackground(for: .window) {
            LinearGradient(
                colors: [
                    Color(nsColor: .windowBackgroundColor),
                    Color(nsColor: .windowBackgroundColor).mix(
                        with: MferenceMacTheme.accentColor,
                        by: 0.04),
                ],
                startPoint: .top,
                endPoint: .bottom)
        }
        .tint(MferenceMacTheme.accentColor)
        .animation(.smooth(duration: 0.3), value: model.requiresModelInstallation)
        .animation(.smooth(duration: 0.25), value: model.error)
        .transaction { transaction in
            if model.isRunning {
                transaction.animation = nil
            }
        }
        .onReceive(
            NotificationCenter.default.publisher(
                for: NSApplication.willTerminateNotification)
        ) { _ in
            model.flushChatPersistence()
        }
    }

    private var primaryContent: some View {
        Group {
            if model.requiresModelInstallation {
                ModelInstallView(model: model)
            } else {
                conversationView
            }
        }
    }

    private var conversationView: some View {
        GeometryReader { geometry in
            ZStack(alignment: .bottom) {
                if model.hasOutputTranscript {
                    OutputPaneView(model: model)
                        .padding(.bottom, conversationChromeHeight)
                } else if conversationChromeHeight > 0 {
                    OutputPaneView(model: model)
                        .frame(
                            height: max(
                                0,
                                geometry.size.height - conversationChromeHeight))
                        .frame(maxHeight: .infinity, alignment: .top)
                }

                conversationChrome
                    .background {
                        GeometryReader { chromeGeometry in
                            Color.clear.preference(
                                key: ConversationChromeHeightKey.self,
                                value: chromeGeometry.size.height)
                        }
                    }
            }
            .onPreferenceChange(ConversationChromeHeightKey.self) { height in
                guard height > 0 else { return }
                var transaction = Transaction()
                transaction.disablesAnimations = true
                withTransaction(transaction) {
                    conversationChromeHeight = height
                }
            }
        }
    }

    private var conversationChrome: some View {
        VStack(spacing: 10) {
            ErrorBanner(model: model)
            PromptComposerView(model: model)
                .frame(maxWidth: 760)
        }
        .padding(.horizontal, 20)
        .padding(.bottom, 16)
    }
}

private struct ConversationChromeHeightKey: PreferenceKey {
    static let defaultValue: CGFloat = 0

    static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) {
        value = max(value, nextValue())
    }
}

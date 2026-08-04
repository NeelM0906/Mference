import MferenceAppCore
import MferenceMacPresentation
import SwiftUI

/// Toolbar model menu in the chat-template style: model name with a status
/// dot, opening a menu with the current state and load/unload actions.
struct ModelStatusMenu: View {
    let model: AppModel

    var body: some View {
        Menu {
            Text(model.presentation.label)
            if let detail = model.presentation.detail {
                Text(detail)
            }
            Divider()
            if model.canLoadModel {
                Button(model.loadState.isFailed ? "Retry Load" : "Load Model",
                       systemImage: "play.fill") {
                    model.loadModel()
                }
            }
            if model.canCancelLoad {
                Button("Cancel Load", systemImage: "xmark") {
                    model.cancelLoad()
                }
            }
            if model.canReloadModel {
                Button("Reload Model", systemImage: "arrow.clockwise") {
                    model.reloadModel()
                }
            }
            if model.canUnloadModel {
                Button("Unload Model", systemImage: "eject") {
                    model.unloadModel()
                }
            }
        } label: {
            HStack(spacing: 6) {
                Circle()
                    .fill(statusColor)
                    .frame(width: 7, height: 7)
                Text(model.installDescriptor.displayName)
                    .fontWeight(.semibold)
                    .lineLimit(1)
                Image(systemName: "chevron.down")
                    .font(.caption2.weight(.bold))
                    .foregroundStyle(.secondary)
            }
        }
        .menuStyle(.borderlessButton)
        .fixedSize()
        .help(model.presentation.label)
    }

    private var statusColor: Color {
        if model.loadState.isReady { return .green }
        if model.loadState.isFailed { return .red }
        if case .loading = model.loadState { return .orange }
        return .secondary.opacity(0.6)
    }
}

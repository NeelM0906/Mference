import MferenceAppCore
import MferenceMacPresentation
import SwiftUI

/// Toolbar model picker in the chat-template style: the active model's name
/// with a status dot, opening a menu that lists the whole shipped model
/// family. Installed models activate directly; missing ones show a download
/// action with size, quant, and parameter markers. Load/unload actions for
/// the active model follow below.
struct ModelStatusMenu: View {
    let model: AppModel

    var body: some View {
        Menu {
            Section("Models") {
                ForEach(model.modelCatalog) { entry in
                    modelRow(entry)
                }
            }

            Divider()
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

    @ViewBuilder
    private func modelRow(_ entry: ModelCatalogEntry) -> some View {
        let isActive = entry.descriptor.family == model.installDescriptor.family
        if entry.isInstalled {
            Button {
                model.selectModel(family: entry.descriptor.family)
            } label: {
                if isActive {
                    Label(entry.descriptor.displayName, systemImage: "checkmark")
                } else {
                    Text(entry.descriptor.displayName)
                }
            }
            .disabled(model.isRunning || model.isInstallingModel)
        } else {
            // displayName carries the parameter count and quant type
            // (e.g. "Qwen3.6 35B-A3B 4-bit"); append the download size.
            Button {
                model.selectModel(family: entry.descriptor.family)
            } label: {
                Label(
                    "\(entry.descriptor.displayName) · \(downloadSize(entry)) · Download",
                    systemImage: "arrow.down.circle")
            }
            .disabled(model.isRunning || model.isInstallingModel)
        }
    }

    private func downloadSize(_ entry: ModelCatalogEntry) -> String {
        String(format: "%.1f GB",
               Double(entry.descriptor.approximateDownloadBytes) / 1_000_000_000)
    }

    private var statusColor: Color {
        if model.loadState.isReady { return .green }
        if model.loadState.isFailed { return .red }
        if case .loading = model.loadState { return .orange }
        return .secondary.opacity(0.6)
    }
}

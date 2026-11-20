import AppKit
import MferenceAppCore
import MferenceDesign
import SwiftUI

/// Settings › Speech (Notes spec 4.8, board 8, `Settings.dc.html`): the
/// Model and Dictation cards on the left, Meetings and Recordings on the
/// right. Credits Fluid Speech Pro as its licence asks (CC BY 4.0).
public struct SpeechSettingsView: View {
    @Bindable var model: SpeechSettingsModel
    @State private var confirmsRemove = false
    @State private var confirmsDownload = false
    @State private var confirmsLabelsDownload = false

    public init(model: SpeechSettingsModel) { self.model = model }

    public var body: some View {
        HStack(alignment: .top, spacing: MFSpacing.l) {
            VStack(spacing: MFSpacing.l) {
                modelCard
                dictationCard
            }
            .frame(maxWidth: .infinity)
            VStack(spacing: MFSpacing.l) {
                meetingsCard
                recordingsCard
            }
            .frame(maxWidth: .infinity)
        }
        .task { await model.refresh() }
        .confirmationDialog("Remove Fluid Speech Pro?", isPresented: $confirmsRemove) {
            Button("Remove", role: .destructive) { Task { await model.removeSpeech() } }
            Button("Keep", role: .cancel) {}
        } message: {
            Text(Self.removeMessage(bytes: model.speechBytes))
        }
        .confirmationDialog("Download Fluid Speech Pro?", isPresented: $confirmsDownload) {
            Button("Download") { model.setup.download() }
            Button("Not now", role: .cancel) {}
        } message: {
            Text(SpeechSetupCard.approvalMessage(bytes: model.speechBytes))
        }
        .confirmationDialog("Download speaker labels?", isPresented: $confirmsLabelsDownload) {
            Button("Download") { model.downloadSpeakerLabels() }
            Button("Not now", role: .cancel) {}
        } message: {
            Text("Speaker labels download (\(VoiceFormat.size(model.speakerLabelsBytes))) from Hugging Face to this Mac and run here. Nothing else is sent.")
        }
    }

    static func removeMessage(bytes: Int64) -> String {
        "This frees \(VoiceFormat.size(bytes)). Your notes and transcripts stay. Dictation, meetings and audio files need the download again."
    }

    // MARK: Model

    private var modelCard: some View {
        MFCard {
            cardHeader("Model")
            MFListRow("Fluid Speech Pro", subtitle: Self.speechSubtitle(model)) {
                speechTrailing
            }
            MFListRow("Language", subtitle: "25 European languages, detected per recording") {
                VoiceMenuField(Self.languageName(model.preferences.language), accessibilityLabel: "Language") {
                    Button("Automatic") { model.preferences.language = .automatic }
                    Divider()
                    ForEach(VoiceFormat.languages, id: \.code) { language in
                        Button(language.name) { model.preferences.language = .fixed(language.code) }
                    }
                }
            }
            MFListRow("Speaker labels", subtitle: Self.speakerLabelsSubtitle(model)) {
                speakerLabelsTrailing
            }
            Text(Self.credit)
                .font(MFFont.caption)
                .foregroundStyle(MFColor.tertiary)
                .lineSpacing(MFLineSpacing.tight)
                .fixedSize(horizontal: false, vertical: true)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.horizontal, MFSpacing.l)
                .padding(.vertical, MFSpacing.m)
        }
    }

    /// The credits the licences ask for (spec 3, App/Resources/Speech/SOURCES.md):
    /// Fluid Speech Pro and the speaker-label models (both CC BY 4.0), LS-EEND
    /// (MIT) and the FluidAudio SDK (Apache 2.0).
    static let credit = "Fluid Speech Pro by FluidVoice, built on NVIDIA Parakeet TDT 0.6B v3 and Moondream Parakeet Ultra. Licensed CC BY 4.0. "
        + "Speaker labels use the pyannote Community-1 models by pyannote, WeSpeaker, BUT Speech@FIT and Fluid Inference, "
        + "modified for Core ML and licensed CC BY 4.0, and LS-EEND (MIT). Speech runs on FluidAudio (Apache 2.0)."

    /// The size line, then a failure or the Offline line when there is one.
    static func speechSubtitle(_ model: SpeechSettingsModel) -> String {
        var lines = [speechLine(model.speechState, bytes: model.speechBytes)]
        if let failure = model.speechFailure, failure != lines[0] {
            lines.append(failure)
        } else if model.setup.offline, !model.speechState.isReady {
            lines.append(SpeechError.offline.message)
        }
        return lines.joined(separator: "\n")
    }

    static func speakerLabelsSubtitle(_ model: SpeechSettingsModel) -> String {
        let size = model.speakerLabelsBytes > 0 ? "Extra \(VoiceFormat.size(model.speakerLabelsBytes)) download"
                                                : "Extra download"
        var lines = ["\(size), used for meetings and files"]
        if let failure = model.speakerLabelsFailure {
            lines.append(failure)
        } else if model.setup.offline, !model.speakerLabelsState.isReady {
            lines.append(SpeechError.offline.message)
        }
        return lines.joined(separator: "\n")
    }

    static func speechLine(_ state: SpeechModelState, bytes: Int64) -> String {
        let base = (bytes > 0 ? "\(VoiceFormat.size(bytes)) · " : "") + "Neural Engine · CC BY 4.0"
        switch state {
        case .downloading(let progress): return "\(base) · downloading \(SpeechSetupCard.percent(progress))"
        case .paused(let progress): return "\(base) · paused at \(SpeechSetupCard.percent(progress))"
        case .preparing: return "\(base) · preparing…"
        case .failed(let error): return error.message
        case .notInstalled, .ready: return base
        }
    }

    static func languageName(_ language: SpeechLanguage) -> String {
        switch language {
        case .automatic: "Automatic"
        case .fixed(let code): VoiceFormat.languageName(code)
        }
    }

    @ViewBuilder
    private var speechTrailing: some View {
        switch model.speechState {
        case .ready:
            MFBadge(.installed)
            Button("Remove") { confirmsRemove = true }
                .buttonStyle(.mfSecondary)
                .disabled(model.isRemoving)
                .accessibilityLabel("Remove Fluid Speech Pro")
        case .notInstalled, .failed:
            MFBadge(.custom("Not installed", .neutral))
            Button(model.speechFailure == nil ? "Download" : "Try again") { confirmsDownload = true }
                .buttonStyle(.mfPrimary)
                .disabled(model.setup.offline)
                .accessibilityLabel("Download Fluid Speech Pro")
        case .downloading(let progress):
            MFProgressBadge("Downloading", fraction: progress)
            Button("Pause") { model.setup.pause() }
                .buttonStyle(.mfSecondary)
                .accessibilityLabel("Pause the Fluid Speech Pro download")
            Button("Cancel") { model.setup.cancel() }
                .buttonStyle(.mfQuiet)
                .accessibilityLabel("Cancel the Fluid Speech Pro download")
        case .paused:
            Button("Resume") { model.setup.download() }
                .buttonStyle(.mfPrimary)
                .disabled(model.setup.offline)
                .accessibilityLabel("Resume the Fluid Speech Pro download")
            Button("Cancel") { model.setup.cancel() }
                .buttonStyle(.mfQuiet)
                .accessibilityLabel("Cancel the Fluid Speech Pro download")
        case .preparing:
            MFBadge(.custom("Preparing…", .neutral))
        }
    }

    @ViewBuilder
    private var speakerLabelsTrailing: some View {
        switch model.speakerLabelsState {
        case .ready:
            MFBadge(.installed)
        case .notInstalled, .failed:
            Button(model.speakerLabelsFailure == nil ? "Download" : "Try again") { confirmsLabelsDownload = true }
                .buttonStyle(.mfSecondary)
                .disabled(model.setup.offline || !model.speechState.isReady)
                .accessibilityLabel("Download speaker labels")
        case .downloading(let progress):
            MFProgressBadge("Downloading", fraction: progress)
            Button("Pause") { model.pauseSpeakerLabels() }
                .buttonStyle(.mfSecondary)
                .accessibilityLabel("Pause the speaker labels download")
            Button("Cancel") { model.cancelSpeakerLabels() }
                .buttonStyle(.mfQuiet)
                .accessibilityLabel("Cancel the speaker labels download")
        case .paused:
            Button("Resume") { model.downloadSpeakerLabels() }
                .buttonStyle(.mfSecondary)
                .disabled(model.setup.offline)
                .accessibilityLabel("Resume the speaker labels download")
            Button("Cancel") { model.cancelSpeakerLabels() }
                .buttonStyle(.mfQuiet)
                .accessibilityLabel("Cancel the speaker labels download")
        case .preparing:
            MFBadge(.custom("Preparing…", .neutral))
        }
    }

    // MARK: Dictation

    private var dictationCard: some View {
        MFCard {
            cardHeader("Dictation")
            MFListRow("Start and stop", subtitle: "In any note") {
                MFKeycap("⇧⌘D")
            }
            MFListRow("Clean up filler words", subtitle: "Drops um, uh and repeated starts") {
                Toggle("Clean up filler words", isOn: Binding(get: { model.preferences.cleanUpFillers },
                                                            set: { model.preferences.cleanUpFillers = $0 }))
                    .toggleStyle(.mf)
            }
            MFListRow("Dictate in other apps", subtitle: "Hold a key and talk into any text field") {
                MFBadge(.later)
            }
        }
    }

    // MARK: Meetings

    private var meetingsCard: some View {
        MFCard {
            cardHeader("Meetings")
            MFListRow("Microphone",
                      subtitle: model.microphoneNeedsSystemSettings ? "Microphone access is off in System Settings." : nil) {
                if model.microphoneNeedsSystemSettings {
                    Button("Open System Settings") { model.openSystemSettings(for: .microphone) }
                        .buttonStyle(.mfSecondary)
                }
                VoiceMenuField(model.microphoneName, accessibilityLabel: "Microphone") {
                    Button("System default") { model.preferences.microphoneID = nil }
                    if !model.devices.isEmpty { Divider() }
                    ForEach(model.devices) { device in
                        Button(device.name) { model.preferences.microphoneID = device.id }
                    }
                }
            }
            MFListRow("Call audio", subtitle: model.callAudioLine) {
                if model.callAudioNeedsSystemSettings {
                    Button("Open System Settings") { model.openSystemSettings(for: .systemAudio) }
                        .buttonStyle(.mfSecondary)
                }
                VoiceMenuField(model.callAudioName, accessibilityLabel: "Call audio") {
                    Button("All apps") { model.preferences.callAudio = .allApps }
                    if !model.apps.isEmpty { Divider() }
                    ForEach(model.apps) { app in
                        Button(app.name) { model.preferences.callAudio = .app(bundleID: app.bundleID, name: app.name) }
                    }
                    Divider()
                    Button("Off: microphone only") { model.preferences.callAudio = .off }
                }
            }
            MFListRow("Summaries", subtitle: "Written when you stop, with the model below") {
                VoiceMenuField(SpeechSettingsModel.summaryName(model.preferences.summaryModel),
                               accessibilityLabel: "Summary model") {
                    ForEach(SpeechPreferences.SummaryModel.allCases, id: \.self) { choice in
                        Button(SpeechSettingsModel.summaryName(choice)) { model.preferences.summaryModel = choice }
                    }
                }
            }
            MFListRow("Ask before a calendar meeting",
                      subtitle: "Offer to record when a call app opens during an event") {
                MFBadge(.later)
            }
        }
    }

    // MARK: Recordings

    private var recordingsCard: some View {
        MFCard {
            cardHeader("Recordings")
            MFListRow("Keep audio", subtitle: "The transcript always stays") {
                VoiceMenuField(SpeechSettingsModel.keepAudioName(model.preferences.keepAudio),
                               accessibilityLabel: "Keep audio") {
                    ForEach(SpeechPreferences.KeepAudio.allCases, id: \.self) { choice in
                        Button(SpeechSettingsModel.keepAudioName(choice)) { model.preferences.keepAudio = choice }
                    }
                }
            }
            MFListRow("Space used", subtitle: model.spaceUsedLine) {
                Button("Show in Finder") { Self.showInFinder(model.recordingsRoot) }
                    .buttonStyle(.mfSecondary)
                    .disabled(model.recordingsBytes == 0)
            }
        }
    }

    static func showInFinder(_ folder: URL) {
        if FileManager.default.fileExists(atPath: folder.path) {
            NSWorkspace.shared.activateFileViewerSelecting([folder])
        }
    }

    private func cardHeader(_ title: String) -> some View {
        MFSectionHeader(title)
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal, MFSpacing.l)
            .padding(.top, MFSpacing.l)
            .padding(.bottom, MFSpacing.s)
    }
}

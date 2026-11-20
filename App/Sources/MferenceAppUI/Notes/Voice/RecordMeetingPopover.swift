import MferenceAppCore
import MferenceDesign
import SwiftUI

/// "Record a meeting" (board 3, `MeetingStart.dc.html` `.pop`): the
/// microphone and call-audio pickers, three switches, the consent reminder,
/// Cancel and Start recording. Choices write back to Settings › Speech.
public struct RecordMeetingPopover: View {
    @Bindable var model: RecordMeetingModel
    let onClose: () -> Void

    public init(model: RecordMeetingModel, onClose: @escaping () -> Void) {
        self.model = model
        self.onClose = onClose
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Text("Record a meeting")
                .font(MFFont.sectionTitle)
                .foregroundStyle(MFColor.ink)
                .accessibilityAddTraits(.isHeader)
            Text("Opens a new note with a live transcript beside it. Everything is transcribed on this Mac.")
                .font(MFFont.preview)
                .foregroundStyle(MFColor.tertiary)
                .lineSpacing(MFLineSpacing.tight)
                .fixedSize(horizontal: false, vertical: true)
                .padding(.top, MFSpacing.xs)
                .padding(.bottom, MFSpacing.comfortable)
            row("Microphone", "What you say in the room") {
                VoiceMenuField(model.microphoneName, accessibilityLabel: "Microphone") {
                    Button("System default") { model.preferences.microphoneID = nil }
                    if !model.devices.isEmpty { Divider() }
                    ForEach(model.devices) { device in
                        Button(device.name) { model.preferences.microphoneID = device.id }
                    }
                }
            }
            row("Call audio", "The other side of Zoom, Meet, Teams or FaceTime") {
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
            row("Speaker labels", model.speakerLabelsAvailable ? "You, then Speaker 2, 3… Rename them after."
                                                                : "Download speaker labels in Settings › Speech. Until then: You and Others.") {
                toggle("Speaker labels", isOn: Binding(get: { model.preferences.speakerLabels && model.speakerLabelsAvailable },
                                                        set: { model.preferences.speakerLabels = $0 }))
                    .disabled(!model.speakerLabelsAvailable)
            }
            row("Keep the recording", "Play any line back from the transcript") {
                toggle("Keep the recording", isOn: Binding(get: { model.preferences.keepAudio == .keep },
                                                            set: { model.preferences.keepAudio = $0 ? .keep : .deleteAfterTranscribing }))
            }
            row("Summary when you stop", model.preferences.summaryModel == .off
                    ? "Summaries are off in Settings › Speech." : "Summary, decisions and action items") {
                toggle("Summary when you stop", isOn: Binding(get: { model.preferences.writesMeetingSummary },
                                                               set: { model.preferences.summaryWhenYouStop = $0 }))
                    .disabled(model.preferences.summaryModel == .off)
            }
            consent
                .padding(.top, MFSpacing.snug)
            if let message = model.errorMessage {
                Text(message)
                    .font(MFFont.preview)
                    .foregroundStyle(MFColor.danger)
                    .fixedSize(horizontal: false, vertical: true)
                    .padding(.top, MFSpacing.s)
            }
            HStack(spacing: MFSpacing.s) {
                Spacer()
                Button("Cancel", action: onClose)
                    .buttonStyle(MFButtonStyle(.secondary, size: .medium))
                    .keyboardShortcut(.cancelAction)
                Button {
                    Task { if await model.startRecording() { onClose() } }
                } label: {
                    HStack(spacing: MFSpacing.s) {
                        Circle().fill(MFColor.danger).frame(width: RecordMetrics.dot, height: RecordMetrics.dot)
                            .accessibilityHidden(true)
                        Text("Start recording")
                    }
                }
                .buttonStyle(MFButtonStyle(.primary, size: .medium))
                .keyboardShortcut(.defaultAction)
                .disabled(model.isStarting)
            }
            .padding(.top, MFSpacing.comfortable)
        }
        .padding(RecordMetrics.padding)
        .frame(width: RecordMetrics.width)
        .task { await model.load() }
    }

    private var consent: some View {
        HStack(alignment: .top, spacing: MFSpacing.compact) {
            Image(systemName: "exclamationmark.circle")
                .font(MFFont.callout)
                .accessibilityHidden(true)
            Text("Let people know you are recording. Some places require everyone's consent.")
                .fixedSize(horizontal: false, vertical: true)
        }
        .font(MFFont.preview)
        .foregroundStyle(MFColor.preview)
        .padding(.horizontal, MFSpacing.m)
        .padding(.vertical, MFSpacing.compact)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(RoundedRectangle(cornerRadius: MFRadius.m).fill(MFColor.approvalFill))
        .overlay(RoundedRectangle(cornerRadius: MFRadius.m).strokeBorder(MFColor.approvalBorder, lineWidth: MFLayout.hairline))
        .accessibilityElement(children: .combine)
    }

    private func toggle(_ label: String, isOn: Binding<Bool>) -> some View {
        Toggle(label, isOn: isOn).toggleStyle(.mf)
    }

    private func row<Trailing: View>(_ title: String, _ subtitle: String,
                                     @ViewBuilder trailing: () -> Trailing) -> some View {
        VStack(spacing: 0) {
            Rectangle().fill(MFColor.hairlineFaint).frame(height: MFLayout.hairline)
            HStack(spacing: MFSpacing.compact) {
                VStack(alignment: .leading, spacing: MFSpacing.xxs) {
                    Text(title)
                        .font(MFFont.rowTitle)
                        .foregroundStyle(MFColor.ink)
                    Text(subtitle)
                        .font(MFFont.caption)
                        .foregroundStyle(MFColor.tertiary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .accessibilityElement(children: .combine)
                trailing()
            }
            .padding(.vertical, MFSpacing.compact)
        }
    }
}

/// Popover sizes from `MeetingStart.dc.html` (`.pop`).
enum RecordMetrics {
    static let width: CGFloat = 400
    static let padding: CGFloat = 18
    static let dot: CGFloat = 9
}

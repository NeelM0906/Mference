import MferenceAppCore
import MferenceDesign
import SwiftUI

/// The setup card (board 7, `Setup.dc.html`): what the download turns on,
/// the model row with Download (then progress, pause, resume, cancel, and
/// "Preparing…"), the two permissions and when they are asked, and the
/// privacy line. Notes shows it in the editor area while a voice action
/// waits for the model.
public struct SpeechSetupCard: View {
    @Bindable var model: SpeechSetupModel
    @State private var confirmsDownload = false

    public init(model: SpeechSetupModel) { self.model = model }

    public var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Image(systemName: "waveform")
                .font(.system(size: SetupMetrics.iconGlyph, weight: .medium))
                .foregroundStyle(MFColor.accentStrong)
                .frame(width: SetupMetrics.icon, height: SetupMetrics.icon)
                .background(RoundedRectangle(cornerRadius: SetupMetrics.iconRadius).fill(MFColor.accentFill))
                .accessibilityHidden(true)
            Text("Speech runs on this Mac")
                .font(MFFont.title)
                .foregroundStyle(MFColor.ink)
                .accessibilityAddTraits(.isHeader)
                .padding(.top, MFSpacing.comfortable)
            Text("One download turns on dictation, meeting transcripts and audio files. The language is picked for you, from 25 European languages including English.")
                .font(MFFont.body)
                .foregroundStyle(MFColor.secondary)
                .lineSpacing(MFLineSpacing.relaxed)
                .fixedSize(horizontal: false, vertical: true)
                .padding(.top, MFSpacing.snug)
            modelRow
                .padding(.top, SetupMetrics.blockGap)
            permissions
                .padding(.top, SetupMetrics.blockGap)
            Label("Audio and text never leave this Mac.", systemImage: "lock")
                .font(MFFont.caption)
                .foregroundStyle(MFColor.success)
                .padding(.top, MFSpacing.l)
        }
        .padding(.horizontal, SetupMetrics.padding.width)
        .padding(.vertical, SetupMetrics.padding.height)
        .frame(width: SetupMetrics.width)
        .background(RoundedRectangle(cornerRadius: MFRadius.xl).fill(MFColor.surface))
        .overlay(RoundedRectangle(cornerRadius: MFRadius.xl).strokeBorder(MFColor.hairlineStrong, lineWidth: MFLayout.hairline))
        .shadow(color: MFColor.ink.opacity(SetupMetrics.shadowOpacity), radius: SetupMetrics.shadowRadius,
                y: SetupMetrics.shadowY)
        .confirmationDialog("Download Fluid Speech Pro?", isPresented: $confirmsDownload) {
            Button("Download") { model.download() }
            Button("Not now", role: .cancel) {}
        } message: {
            Text(Self.approvalMessage(bytes: model.downloadBytes))
        }
    }

    /// The download approval, worded as for Quick and Deep.
    static func approvalMessage(bytes: Int64) -> String {
        "Fluid Speech Pro downloads (\(VoiceFormat.size(bytes))) from Hugging Face to this Mac and runs here. Nothing else is sent."
    }

    /// The model row's second line.
    static func modelLine(bytes: Int64) -> String {
        "\(VoiceFormat.size(bytes)) · runs on the Neural Engine, so chat and the model you use there keep running"
    }

    // MARK: Model row

    private var modelRow: some View {
        HStack(alignment: .center, spacing: MFSpacing.m) {
            VStack(alignment: .leading, spacing: SetupMetrics.lineGap) {
                Text("Fluid Speech Pro")
                    .font(MFFont.bodyEmphasis.weight(.semibold))
                    .foregroundStyle(MFColor.ink)
                Text(Self.modelLine(bytes: model.downloadBytes))
                    .font(MFFont.caption)
                    .foregroundStyle(MFColor.tertiary)
                    .lineSpacing(MFLineSpacing.tight)
                    .fixedSize(horizontal: false, vertical: true)
                progressLine
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .accessibilityElement(children: .combine)
            actions
        }
        .padding(MFSpacing.comfortable)
        .overlay(RoundedRectangle(cornerRadius: MFRadius.l).strokeBorder(MFColor.hairline, lineWidth: MFLayout.hairline))
    }

    @ViewBuilder
    private var progressLine: some View {
        switch model.display {
        case .download where model.offline:
            Text("Turn off Offline mode in Settings to download.")
                .font(MFFont.caption)
                .foregroundStyle(MFColor.preview)
                .padding(.top, MFSpacing.xs)
        case .downloading(let fraction):
            VStack(alignment: .leading, spacing: MFSpacing.xs) {
                MFProgressBar(fraction, label: "Downloading Fluid Speech Pro")
                Text("Downloading · \(Self.percent(fraction))")
                    .font(MFFont.caption)
                    .foregroundStyle(MFColor.secondary)
                    .monospacedDigit()
            }
            .padding(.top, MFSpacing.s)
        case .paused(let fraction):
            VStack(alignment: .leading, spacing: MFSpacing.xs) {
                MFProgressBar(fraction, label: "Fluid Speech Pro download, paused")
                Text(model.offline ? "Paused at \(Self.percent(fraction)). Turn off Offline mode in Settings to download."
                                   : "Paused at \(Self.percent(fraction))")
                    .font(MFFont.caption)
                    .foregroundStyle(MFColor.secondary)
            }
            .padding(.top, MFSpacing.s)
        case .preparing:
            Text("Preparing… The first start takes a few seconds.")
                .font(MFFont.caption)
                .foregroundStyle(MFColor.secondary)
                .padding(.top, MFSpacing.xs)
        case .failed(let message):
            Text(message)
                .font(MFFont.caption)
                .foregroundStyle(MFColor.danger)
                .padding(.top, MFSpacing.xs)
        case .download, .ready:
            EmptyView()
        }
    }

    @ViewBuilder
    private var actions: some View {
        switch model.display {
        case .download:
            Button("Download") { confirmsDownload = true }
                .buttonStyle(MFButtonStyle(.primary, size: .large))
                .disabled(model.offline)
        case .downloading:
            Button("Pause") { model.pause() }
                .buttonStyle(.mfSecondary)
            Button("Cancel") { model.cancel() }
                .buttonStyle(.mfQuiet)
                .accessibilityLabel("Cancel the download")
        case .paused:
            Button("Resume") { model.download() }
                .buttonStyle(.mfPrimary)
                .disabled(model.offline)
            Button("Cancel") { model.cancel() }
                .buttonStyle(.mfQuiet)
                .accessibilityLabel("Cancel the download")
        case .preparing:
            ProgressView()
                .controlSize(.small)
                .accessibilityLabel("Preparing Fluid Speech Pro")
        case .failed:
            Button("Try again") { model.download() }
                .buttonStyle(.mfPrimary)
                .disabled(model.offline)
        case .ready:
            MFBadge(.ready)
        }
    }

    static func percent(_ fraction: Double) -> String {
        "\(MFProgress.percent(fraction) ?? 0)%"
    }

    // MARK: Permissions

    private var permissions: some View {
        VStack(alignment: .leading, spacing: MFSpacing.snug) {
            MFSectionHeader("What Mference will ask for, and when")
            permissionRow(systemImage: "mic", title: "Microphone", detail: AppPermission.microphone.reason,
                          when: "First dictation")
            permissionRow(systemImage: "speaker.wave.2", title: "Call audio",
                          detail: "To hear the other side of a call. macOS calls it system audio recording.",
                          when: "First meeting")
        }
    }

    private func permissionRow(systemImage: String, title: String, detail: String, when: String) -> some View {
        VStack(spacing: 0) {
            Rectangle().fill(MFColor.hairlineFaint).frame(height: MFLayout.hairline)
            HStack(alignment: .top, spacing: MFSpacing.m) {
                Image(systemName: systemImage)
                    .font(MFFont.callout)
                    .foregroundStyle(MFColor.secondary)
                    .frame(width: SetupMetrics.permissionIcon)
                    .padding(.top, MFSpacing.xxs)
                    .accessibilityHidden(true)
                VStack(alignment: .leading, spacing: MFSpacing.xxs) {
                    Text(title)
                        .font(MFFont.rowTitle)
                        .foregroundStyle(MFColor.ink)
                    Text(detail)
                        .font(MFFont.preview)
                        .foregroundStyle(MFColor.tertiary)
                        .lineSpacing(MFLineSpacing.tight)
                        .fixedSize(horizontal: false, vertical: true)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                Text(when)
                    .font(MFFont.caption)
                    .foregroundStyle(MFColor.tertiary)
                    .padding(.top, MFSpacing.xxs)
            }
            .padding(.vertical, SetupMetrics.permissionPadding)
            .accessibilityElement(children: .combine)
        }
    }
}

/// Setup card sizes from `Setup.dc.html` (`.card`, `.ic`, `.model`, `.pr`).
enum SetupMetrics {
    static let width: CGFloat = 520
    static let padding = CGSize(width: 28, height: 26)
    static let icon: CGFloat = 44
    static let iconGlyph: CGFloat = 20
    static let iconRadius: CGFloat = 12
    static let blockGap: CGFloat = 18
    static let lineGap: CGFloat = 3
    static let permissionIcon: CGFloat = 16
    static let permissionPadding: CGFloat = 9
    static let shadowOpacity: Double = 0.08
    static let shadowRadius: CGFloat = 20
    static let shadowY: CGFloat = 12
}

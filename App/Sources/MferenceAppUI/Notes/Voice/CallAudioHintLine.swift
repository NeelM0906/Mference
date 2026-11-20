import MferenceAppCore
import MferenceDesign
import SwiftUI

/// The quiet line under the recording bar while call audio is on but the
/// tap has heard only silence (`MeetingSession.callAudioHint`): the hint,
/// Open System Settings (through the permission broker) and a close button.
/// Same voice as the bar's "Call audio off" note: caption, no alert.
struct CallAudioHintLine: View {
    let message: String
    let onOpenSettings: () -> Void
    let onDismiss: () -> Void

    var body: some View {
        HStack(spacing: MFSpacing.s) {
            Image(systemName: "speaker.slash")
                .font(MFFont.caption)
                .foregroundStyle(MFColor.tertiary)
                .accessibilityHidden(true)
            Text(message)
                .font(MFFont.caption)
                .foregroundStyle(MFColor.secondary)
                .lineLimit(1)
                .truncationMode(.tail)
                .help(message)
            Button("Open System Settings", action: onOpenSettings)
                .buttonStyle(.plain)
                .font(MFFont.caption)
                .foregroundStyle(MFColor.accent)
                .accessibilityHint("Opens Call audio in System Settings")
            Spacer(minLength: MFSpacing.s)
            Button(action: onDismiss) {
                Image(systemName: "xmark")
                    .font(MFFont.caption)
                    .foregroundStyle(MFColor.tertiary)
                    .frame(width: LiveMetrics.hintClose, height: LiveMetrics.hintClose)
                    .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .accessibilityLabel("Hide the call audio hint")
            .help("Hide")
        }
        .padding(.leading, LiveMetrics.barPadding)
        .padding(.trailing, MFSpacing.s)
        .background(MFColor.surface)
        .overlay(alignment: .bottom) {
            Rectangle().fill(MFColor.chromeDivider).frame(height: MFLayout.hairline)
        }
        .accessibilityElement(children: .contain)
    }
}

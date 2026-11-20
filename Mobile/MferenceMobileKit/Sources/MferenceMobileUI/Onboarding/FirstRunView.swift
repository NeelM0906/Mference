import MferenceDesign
import SwiftUI

/// First run: welcome, the Quick model, and pairing a Mac (optional).
/// Permissions are asked the first time a feature needs them, never here.
struct FirstRunView: View {
    @Bindable var model: MobileAppModel
    @State private var step = 1
    @State private var progress = 0.0

    var body: some View {
        ZStack {
            LinearGradient(colors: [Color.mf(light: 0xE7E3FF, dark: 0x1D1A33),
                                    Color.mf(light: 0xF3F4F9, dark: 0x16171B),
                                    Color.mf(light: 0xFFE5D4, dark: 0x2A1E18)],
                           startPoint: .topLeading, endPoint: .bottomTrailing)
                .overlay(MFColor.canvas.opacity(0.55))
                .ignoresSafeArea()
            ScrollView {
                VStack(alignment: .leading, spacing: MFSpacing.l + 2) {
                    HStack {
                        AppMark(side: 52)
                        Spacer()
                        Text("Step \(step) of 3").font(MFPhoneFont.caption).foregroundStyle(MFColor.tertiary)
                    }
                    switch step {
                    case 1: welcome
                    case 2: models
                    default: pairing
                    }
                }
                .frame(maxWidth: 520, alignment: .leading)
                .padding(.horizontal, MFSpacing.xl - 4)
                .padding(.top, MFSpacing.m)
                .frame(maxWidth: .infinity)
            }
            .safeAreaInset(edge: .bottom) { footer }
        }
        .animation(.snappy(duration: 0.25), value: step)
    }

    private var welcome: some View {
        VStack(alignment: .leading, spacing: MFSpacing.m) {
            Text("Your own AI,\non this iPhone.")
                .font(MFFont.geist(30, .semibold, relativeTo: .largeTitle))
                .foregroundStyle(MFColor.ink)
            Text("Chat, web answers, your calendar and scanning, run by a model on your phone. Bigger jobs can go to your Mac.")
                .font(MFPhoneFont.callout)
                .foregroundStyle(MFColor.secondary)
            PhoneCard {
                PhoneRow(title: "No account", subtitle: "Nothing to sign up for, and no usage data sent anywhere.")
                PhoneRow(title: "Permissions as you need them", subtitle: "Calendar or camera access is asked the first time, with a reason.")
            }
            .padding(.top, MFSpacing.s)
        }
    }

    private var models: some View {
        VStack(alignment: .leading, spacing: MFSpacing.l) {
            Text("Your own AI,\non this iPhone.")
                .font(MFFont.geist(30, .semibold, relativeTo: .largeTitle))
                .foregroundStyle(MFColor.ink)
            Text("We checked this iPhone: [model], [8 GB] memory, [96 GB] free. Quick fits and runs here.")
                .font(MFPhoneFont.callout)
                .foregroundStyle(MFColor.secondary)
            VStack(alignment: .leading, spacing: MFSpacing.compact) {
                HStack(spacing: MFSpacing.s) {
                    Image(systemName: "checkmark").font(.system(size: 15, weight: .bold)).foregroundStyle(MFColor.ink)
                    Text("Quick").font(MFPhoneFont.headline).foregroundStyle(MFColor.ink)
                    PhoneChip(title: "Qwen3.5 4B")
                }
                Text("Chat, web answers, your calendar and scanning, all on this iPhone.")
                    .font(MFPhoneFont.preview).foregroundStyle(MFColor.secondary)
                ProgressView(value: progress)
                    .tint(MFColor.accent)
                HStack {
                    Text(progress >= 1 ? "Ready · [2.9 GB]" : "Downloading · \(Int(progress * 100))% of [2.9 GB]")
                    Spacer()
                    Text("Wi-Fi only")
                }
                .font(MFPhoneFont.caption)
                .foregroundStyle(MFColor.tertiary)
            }
            .padding(MFSpacing.l)
            .background(RoundedRectangle(cornerRadius: MFRadius.l).fill(MFColor.surface))
            .overlay(RoundedRectangle(cornerRadius: MFRadius.l).strokeBorder(MFColor.ink, lineWidth: MFLayout.strokeEmphasis))
            PhoneCard {
                PhoneRow(title: "Apple's on-device model", subtitle: "Titles, short summaries and scan fields when Apple Intelligence is on.") {
                    PhoneChip(title: "Built in")
                }
            }
            Text("Bigger models like Deep stay on your Mac. Next: pair this iPhone with your Mac, or skip it.")
                .font(MFPhoneFont.caption)
                .foregroundStyle(MFColor.tertiary)
        }
        .task {
            // A stand-in for the real download in this sample build.
            while progress < 1 {
                try? await Task.sleep(for: .milliseconds(60))
                progress = min(1, progress + 0.02)
            }
        }
    }

    private var pairing: some View {
        VStack(alignment: .leading, spacing: MFSpacing.l) {
            Text("Pair your Mac")
                .font(MFFont.geist(30, .semibold, relativeTo: .largeTitle))
                .foregroundStyle(MFColor.ink)
            Text("Your Mac can take the bigger jobs and run web searches through its SearXNG. It works on the same Wi-Fi only, encrypted between your devices.")
                .font(MFPhoneFont.callout)
                .foregroundStyle(MFColor.secondary)
            PhoneCard {
                HStack(spacing: MFSpacing.m) {
                    Image(systemName: "laptopcomputer")
                        .font(.system(size: 20))
                        .foregroundStyle(MFColor.ink)
                        .frame(width: 44, height: 44)
                        .background(RoundedRectangle(cornerRadius: 12).fill(MFColor.fill))
                    VStack(alignment: .leading, spacing: 2) {
                        Text("MacBook Pro").font(MFPhoneFont.rowTitle).foregroundStyle(MFColor.ink)
                        Text("Found on this Wi-Fi · Deep 27B ready").font(MFPhoneFont.preview).foregroundStyle(MFColor.secondary)
                    }
                    Spacer()
                    if model.pairedMac != nil {
                        PhoneChip(title: "Paired", systemImage: "checkmark", tone: .success)
                    }
                }
                .padding(MFSpacing.l)
            }
            Text("Check that your Mac shows the same code: [4 8 1 9]")
                .font(MFPhoneFont.caption)
                .foregroundStyle(MFColor.tertiary)
        }
    }

    private var footer: some View {
        VStack(spacing: MFSpacing.s) {
            switch step {
            case 3:
                if model.pairedMac == nil {
                    Button("Pair") { model.pairedMac = .sample }
                        .buttonStyle(PhoneButtonStyle(kind: .primary, size: .large, fullWidth: true))
                    Button("Skip for now") { finish() }
                        .buttonStyle(PhoneButtonStyle(kind: .secondary, fullWidth: true))
                } else {
                    Button("Start") { finish() }
                        .buttonStyle(PhoneButtonStyle(kind: .primary, size: .large, fullWidth: true))
                }
            default:
                Button("Continue") { step += 1 }
                    .buttonStyle(PhoneButtonStyle(kind: .primary, size: .large, fullWidth: true))
                if step > 1 {
                    Button("Back") { step -= 1 }
                        .buttonStyle(PhoneButtonStyle(kind: .quiet, fullWidth: true))
                }
            }
        }
        .frame(maxWidth: 520)
        .padding(.horizontal, MFSpacing.xl - 4)
        .padding(.bottom, MFSpacing.s)
    }

    private func finish() {
        model.selection = .chat
        model.isFirstRunPresented = false
    }
}

/// The silver "M" mark.
struct AppMark: View {
    var side: CGFloat = 44

    var body: some View {
        Text("M")
            .font(MFFont.geist(side * 0.46, .semibold, relativeTo: .title))
            .foregroundStyle(MFColor.secondary)
            .frame(width: side, height: side)
            .background(RoundedRectangle(cornerRadius: side * 0.27)
                .fill(LinearGradient(colors: [MFColor.surface, MFColor.hairlineStrong],
                                     startPoint: .topLeading, endPoint: .bottomTrailing)))
            .overlay(RoundedRectangle(cornerRadius: side * 0.27).strokeBorder(MFColor.hairlineStrong))
            .shadow(color: .black.opacity(0.08), radius: 8, y: 4)
            .accessibilityLabel("Mference")
    }
}

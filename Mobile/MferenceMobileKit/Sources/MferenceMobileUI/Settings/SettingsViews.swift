import MferenceDesign
import SwiftUI

/// The groups in Settings. On iPhone they stack in one scroll; on Duo open
/// they are the left pane's list.
enum SettingsGroup: String, CaseIterable, Identifiable {
    case models = "Models", yourMac = "Your Mac", privacy = "Privacy", memory = "Memory",
         connections = "Connections", about = "About"
    var id: String { rawValue }
}

/// Settings on iPhone and Duo closed.
struct SettingsScreen: View {
    @Bindable var model: MobileAppModel

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: MFSpacing.m) {
                    Text("This iPhone: [model], [8 GB] memory. Chat, scanning and your calendar run here.")
                        .font(MFPhoneFont.preview)
                        .foregroundStyle(MFColor.secondary)
                    ForEach(SettingsGroup.allCases) { group in
                        PhoneSectionHeader(group.rawValue).padding(.top, MFSpacing.s)
                        SettingsGroupContent(group: group, model: model)
                    }
                }
                .padding(.horizontal, MFPhoneMetrics.screenMargin)
                .padding(.bottom, MFSpacing.xxl)
            }
            .background(MFColor.canvas)
            .navigationTitle("Settings")
            .navigationBarTitleDisplayMode(.large)
        }
    }
}

/// One group's card.
struct SettingsGroupContent: View {
    let group: SettingsGroup
    @Bindable var model: MobileAppModel

    var body: some View {
        switch group {
        case .models:
            PhoneCard {
                PhoneRow(title: "Quick · Qwen3.5 4B", subtitle: "Default · [2.9 GB] on this iPhone · Apache 2.0") {
                    PhoneChip(title: "Loaded", tone: .success)
                }
                PhoneRow(title: "Apple's on-device model", subtitle: "Titles, summaries, scan fields") {
                    Toggle("Apple's on-device model", isOn: $model.appleModelEnabled).labelsHidden().tint(MFColor.ink)
                }
                PhoneRow(title: "Unload when idle") {
                    Picker("Unload when idle", selection: $model.unloadAfterMinutes) {
                        ForEach([5, 10, 30, 60], id: \.self) { Text("\($0) min").tag($0) }
                    }
                    .tint(MFColor.secondary)
                }
            }
        case .yourMac:
            PhoneCard {
                if let mac = model.pairedMac {
                    HStack(spacing: MFSpacing.m) {
                        Image(systemName: "laptopcomputer")
                            .font(.system(size: 18))
                            .foregroundStyle(MFColor.ink)
                            .frame(width: 36, height: 36)
                            .background(RoundedRectangle(cornerRadius: MFRadius.m).fill(MFColor.fill))
                        PhoneRow(title: mac.name, subtitle: "Paired · \(mac.models.first?.shortName ?? "Deep") and \(max(0, mac.models.count - 1)) more") {
                            PhoneChip(title: mac.isOnSameWiFi ? "Same Wi-Fi" : "Not nearby", tone: mac.isOnSameWiFi ? .success : .neutral)
                        }
                        .padding(.leading, -MFSpacing.l)
                    }
                    .padding(.leading, MFSpacing.l)
                    PhoneRow(title: "Big jobs", subtitle: "Ask each time before sending to the Mac")
                    PhoneRow(title: "Forget this Mac") {
                        Button("Forget") { model.pairedMac = nil }
                            .buttonStyle(PhoneButtonStyle(kind: .secondary, size: .compact))
                    }
                } else {
                    PhoneRow(title: "No Mac paired", subtitle: "Open Mference on your Mac, then Settings › iPhone. Same Wi-Fi only.") {
                        Button("Pair") { model.pairedMac = .sample }
                            .buttonStyle(PhoneButtonStyle(kind: .primary, size: .compact))
                    }
                }
            }
        case .privacy:
            PhoneCard {
                PhoneRow(title: "Web search via paired Mac",
                         subtitle: model.pairedMac == nil
                            ? "Needs a paired Mac. Searches run through SearXNG on it."
                            : "Searches run through SearXNG on your Mac. Only the queries leave it.") {
                    Toggle("Web search via paired Mac", isOn: $model.webSearchViaMac).labelsHidden().tint(MFColor.ink)
                        .disabled(model.pairedMac == nil)
                }
                PhoneRow(title: "Ask before any send or delete", subtitle: "Always on.") {
                    PhoneChip(title: "Always")
                }
                PhoneRow(title: "No account", subtitle: "No usage reporting. There is nothing to opt out of.")
            }
        case .memory:
            PhoneCard {
                PhoneRow(title: "Prefers short, friendly replies at work", subtitle: "Learned from chat")
                PhoneRow(title: "Dentist is Northside Dental", subtitle: "From chat, Sep 14")
                PhoneRow(title: "\(max(0, model.memoryCount - 2)) more", subtitle: "All on this iPhone. Edit or delete anything.")
            }
        case .connections:
            PhoneCard {
                PhoneRow(title: "Calendar", subtitle: "Apple Calendar on this iPhone · asks the first time") {
                    PhoneChip(title: "Sample data")
                }
                PhoneRow(title: "Reminders", subtitle: "To-dos sync both ways · asks the first time") {
                    PhoneChip(title: "Sample data")
                }
            }
        case .about:
            PhoneCard {
                PhoneRow(title: "Mference for iPhone", subtitle: "Preview build · sample data, no model loaded")
                PhoneRow(title: "Show first run again") {
                    Button("Show") { model.isFirstRunPresented = true }
                        .buttonStyle(PhoneButtonStyle(kind: .secondary, size: .compact))
                }
            }
        }
    }
}

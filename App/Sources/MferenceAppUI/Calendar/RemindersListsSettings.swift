import EventKit
import MferenceAppCore
import MferenceDesign
import SwiftUI

/// Settings > Connections, under the Reminders switch: which Reminders
/// lists show as to-dos, and which list a to-do added here goes to.
struct RemindersListsSettings: View {
    @Environment(CalendarWeekModel.self) private var model: CalendarWeekModel?
    /// Bumped after a change: the choices live in UserDefaults, which
    /// SwiftUI does not observe.
    @State private var revision = 0

    var body: some View {
        if let model {
            let lists = model.reminderLists
            let shown = model.shownReminderListIDs
            VStack(alignment: .leading, spacing: 0) {
                if lists.isEmpty {
                    MFListRow("Reminders lists",
                              subtitle: "Shown once Mference can see Reminders. Every list syncs until you choose.") {
                        EmptyView()
                    }
                } else {
                    MFListRow("Show reminders from",
                              subtitle: "Open reminders in these lists show as to-dos. Changes go both ways; a delete in one app deletes in both.") {
                        EmptyView()
                    }
                    ForEach(lists) { list in
                        Toggle(isOn: Binding(
                            get: { shown?.contains(list.id) ?? true },
                            set: { model.setShown($0, list: list.id); revision += 1 })) {
                            HStack(spacing: MFSpacing.s) {
                                Circle().fill(color(list.color)).frame(width: 9, height: 9)
                                Text(list.isDefault ? "\(list.title) (default)" : list.title)
                                    .font(MFFont.body)
                                    .foregroundStyle(MFColor.ink)
                            }
                        }
                        .toggleStyle(.checkbox)
                        .padding(.leading, MFSpacing.l)
                        .padding(.vertical, MFSpacing.xxs)
                    }
                    MFListRow("New to-dos go to") {
                        Picker("New to-dos go to", selection: Binding(
                            get: { model.targetReminderListID ?? "" },
                            set: { model.setTargetReminderList($0.isEmpty ? nil : $0); revision += 1 })) {
                            Text("Default list").tag("")
                            ForEach(lists) { list in Text(list.title).tag(list.id) }
                        }
                        .labelsHidden()
                        .fixedSize()
                    }
                }
            }
            .id(revision)
            .task {
                await model.loadReminderLists()
                await model.syncRemindersQuietly()
            }
            // Access just granted (the switch asks): show the lists now.
            .onReceive(NotificationCenter.default.publisher(for: .EKEventStoreChanged)) { _ in
                Task { await model.loadReminderLists() }
            }
        }
    }

    private func color(_ hex: UInt32?) -> Color {
        guard let hex else { return MFColor.tertiary }
        return Color(red: Double((hex >> 16) & 0xFF) / 255, green: Double((hex >> 8) & 0xFF) / 255,
                     blue: Double(hex & 0xFF) / 255)
    }
}

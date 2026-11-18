import MferenceAppCore
import MferenceDesign
import SwiftUI

/// The to-do panel (mockup 04, right): items with source links, the
/// Reminders sync state and the one-click sync that asks first.
struct TodoList: View {
    var model: CalendarWeekModel
    @State private var newTitle = ""
    @FocusState private var addFieldFocused: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: MFSpacing.m) {
            HStack(alignment: .firstTextBaseline, spacing: MFSpacing.s) {
                Text("To-dos")
                    .font(MFFont.headline)
                    .foregroundStyle(MFColor.ink)
                    .accessibilityAddTraits(.isHeader)
                Text("filled from mail and meetings")
                    .font(MFFont.callout)
                    .foregroundStyle(MFColor.tertiary)
            }
            syncCard
            FitOrScroll {
                VStack(spacing: 0) {
                    ForEach(model.sortedTodos) { todo in
                        TodoRow(todo: todo) { model.setDone($0, for: todo.id) }
                        Rectangle().fill(MFColor.hairlineFaint).frame(height: 1)
                    }
                    if model.todos.isEmpty {
                        Text("No to-dos yet. Mference adds the ones it finds in your mail; you can add your own below.")
                            .font(MFFont.callout)
                            .foregroundStyle(MFColor.secondary)
                            .padding(.vertical, MFSpacing.m)
                    }
                }
            }
            Spacer(minLength: 0)
            AppleSectionHeader(text: "Add a to-do")
            HStack(spacing: MFSpacing.s) {
                AppleTextField(title: "Add a to-do", text: $newTitle)
                    .font(MFFont.body)
                    .focused($addFieldFocused)
                    .onSubmit(add)
                Button(action: add) {
                    Image(systemName: "arrow.up")
                        .font(MFFont.calloutEmphasis)
                        .foregroundStyle(MFColor.onInk)
                        .padding(MFSpacing.s)
                        .background(Circle().fill(MFColor.ink))
                }
                .buttonStyle(.plain)
                .accessibilityLabel("Add to-do")
                .disabled(newTitle.trimmingCharacters(in: .whitespaces).isEmpty)
            }
            .padding(.leading, MFSpacing.l)
            .padding(.trailing, MFSpacing.xs)
            .padding(.vertical, MFSpacing.xs)
            .background(Capsule().fill(MFColor.surface))
            .overlay(Capsule().strokeBorder(MFColor.hairlineStrong))
        }
        .padding(MFSpacing.l + MFSpacing.xs)
    }

    @ViewBuilder
    private var syncCard: some View {
        HStack(alignment: .center, spacing: MFSpacing.s) {
            Text(model.remindersSyncEnabled
                 ? "Apple Reminders sync is on. Each sync asks first."
                 : "Apple Reminders sync is off. Each sync asks first.")
                .font(MFFont.callout)
                .foregroundStyle(MFColor.secondary)
                .fixedSize(horizontal: false, vertical: true)
            Spacer(minLength: MFSpacing.xs)
            if model.remindersSyncEnabled {
                Button(model.isSyncingReminders ? "Syncing…" : "Sync now") { Task { await model.syncReminders() } }
                    .buttonStyle(.plain)
                    .font(MFFont.calloutEmphasis)
                    .foregroundStyle(MFColor.accent)
                    .disabled(model.isSyncingReminders || model.todos.isEmpty)
            } else {
                Text("Settings")
                    .font(MFFont.calloutEmphasis)
                    .foregroundStyle(MFColor.accent)
                    .accessibilityHint("Turn on Reminders sync in Settings > Connections")
            }
        }
        .padding(MFSpacing.m + MFSpacing.xxs)
        .background(RoundedRectangle(cornerRadius: MFRadius.m).fill(MFColor.surface))
        .overlay(RoundedRectangle(cornerRadius: MFRadius.m).strokeBorder(MFColor.hairline))
    }

    private func add() {
        let title = newTitle.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !title.isEmpty else { return }
        model.addTodo(TodoItem(title: title))
        newTitle = ""
    }
}

struct TodoRow: View {
    var todo: TodoItem
    var setDone: (Bool) -> Void

    var body: some View {
        HStack(alignment: .top, spacing: MFSpacing.m) {
            Button { setDone(!todo.isDone) } label: {
                Image(systemName: todo.isDone ? "checkmark" : "square")
                    .font(MFFont.bodyEmphasis)
                    .foregroundStyle(todo.isDone ? MFColor.ink : MFColor.tertiary)
                    .frame(width: MFSpacing.l + MFSpacing.xs, height: MFSpacing.l + MFSpacing.xs)
            }
            .buttonStyle(.plain)
            .accessibilityLabel(todo.isDone ? "Mark \(todo.title) not done" : "Mark \(todo.title) done")
            VStack(alignment: .leading, spacing: MFSpacing.xs + MFSpacing.xxs) {
                Text(todo.title)
                    .font(MFFont.body)
                    .foregroundStyle(todo.isDone ? MFColor.tertiary : MFColor.ink)
                    .strikethrough(todo.isDone)
                if !todo.isDone, todo.source.kind != .manual || todo.due != nil {
                    AppleChip(text: sourceLine, tone: todo.source.kind == .meeting ? .accent : .neutral)
                }
            }
            Spacer(minLength: 0)
        }
        .padding(.vertical, MFSpacing.m)
        .accessibilityElement(children: .contain)
    }

    private var sourceLine: String {
        var parts: [String] = []
        if todo.source.kind != .manual { parts.append(todo.source.kindLabel) }
        if let due = todo.due {
            parts.append("due \(due.formatted(.dateTime.weekday(.abbreviated)))")
        } else if todo.source.kind != .manual {
            parts.append(todo.source.label)
        }
        return parts.joined(separator: " · ")
    }
}

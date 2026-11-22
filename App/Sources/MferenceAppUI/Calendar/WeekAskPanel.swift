import MferenceAppCore
import MferenceDesign
import SwiftUI

/// Ask about your week: the suggestion chips, the answer card and the box.
/// Under the to-dos in Day and Week; a bar under the grid in Month.
struct WeekAskPanel: View {
    @Bindable var ask: WeekAskModel
    var calendar: CalendarWeekModel
    /// Month's slim bar: the card floats over the grid above it.
    var compact = false
    /// Opens a Private chat with these answers; nil hides the button.
    var continueInChat: (() -> Void)?
    @FocusState private var focused: Bool

    static let suggestions = ["When am I free for 2 hours?", "What's my busiest day?", "What's due this week?",
                              "Anything overlapping?"]

    private var isIdle: Bool { ask.answers.isEmpty && ask.pending == nil }

    var body: some View {
        VStack(alignment: .leading, spacing: MFSpacing.s) {
            if !compact { header }
            if isIdle && !compact {
                FlowLayout(spacing: MFSpacing.xs) {
                    ForEach(Self.suggestions, id: \.self) { suggestion in
                        Button(suggestion) { Task { await ask.ask(suggestion) } }
                            .buttonStyle(ChipButtonStyle())
                    }
                }
                Text("Answers use only this week's events and to-dos. Nothing is saved.")
                    .font(MFFont.caption)
                    .foregroundStyle(MFColor.tertiary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            if !isIdle { card }
            if let notice = ask.notice { noticeBar(notice) }
            composer
        }
        .onChange(of: calendar.askWeek) { ask.scopeChanged() }
    }

    private var header: some View {
        HStack(spacing: MFSpacing.xs) {
            AppleSectionHeader(text: "Ask about your week")
            scopePill
        }
    }

    private var scopePill: some View {
        Text(ask.scopeLabel)
            .font(MFFont.caption)
            .foregroundStyle(MFColor.secondary)
            .padding(.horizontal, MFSpacing.s)
            .padding(.vertical, MFSpacing.xxs)
            .background(Capsule().fill(MFColor.fill))
            .accessibilityLabel("About the week of \(ask.scopeLabel)")
    }

    // MARK: Card

    private var card: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Text(threadLabel)
                    .font(MFFont.caption)
                    .foregroundStyle(MFColor.tertiary)
                Spacer(minLength: 0)
                Button { ask.clear() } label: { Image(systemName: "xmark") }
                    .buttonStyle(.appleQuiet)
                    .accessibilityLabel("Clear these answers")
            }
            .padding(.leading, MFSpacing.m)
            .padding(.trailing, MFSpacing.xs)
            .padding(.top, MFSpacing.xs)
            // Bounded already (three answers, three rows each until "Show
            // all"), so the card sizes to its content and the to-do list
            // above gives way instead.
            VStack(alignment: .leading, spacing: MFSpacing.m) {
                ForEach(earlier) { answer in
                    VStack(alignment: .leading, spacing: MFSpacing.xxs) {
                        Text(answer.question).font(MFFont.caption).foregroundStyle(MFColor.tertiary)
                        Text(answer.sentence).font(MFFont.callout).foregroundStyle(MFColor.secondary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
                if let pending = ask.pending {
                    pendingView(pending)
                } else if let latest = ask.answers.last {
                    WeekAnswerView(answer: latest, ask: ask, isStale: ask.isStale)
                    if let continueInChat {
                        HStack {
                            Spacer(minLength: 0)
                            Button("Continue in Chat", action: continueInChat)
                                .buttonStyle(.appleSecondary)
                                .accessibilityHint("Opens a new Private chat with these questions and answers")
                        }
                    }
                }
            }
            .padding(.horizontal, MFSpacing.m)
            .padding(.bottom, MFSpacing.m)
            HStack(alignment: .firstTextBaseline, spacing: MFSpacing.xs) {
                Image(systemName: "lock")
                Text("Not saved · clears when you leave this week")
                    .fixedSize(horizontal: false, vertical: true)
            }
            .font(MFFont.caption)
            .foregroundStyle(MFColor.tertiary)
            .padding(.horizontal, MFSpacing.m)
            .padding(.vertical, MFSpacing.s)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(MFColor.fillSubtle)
        }
        .background(RoundedRectangle(cornerRadius: MFRadius.m).fill(MFColor.surface))
        .clipShape(RoundedRectangle(cornerRadius: MFRadius.m))
        .overlay(RoundedRectangle(cornerRadius: MFRadius.m).strokeBorder(MFColor.hairline))
    }

    /// The earlier answers, one line each (the latest is shown in full).
    private var earlier: [WeekAnswer] {
        ask.pending == nil ? Array(ask.answers.dropLast()) : ask.answers
    }

    private var threadLabel: String {
        let count = ask.answers.count + (ask.pending == nil ? 0 : 1)
        return count == 1 ? "This week · 1 question"
            : "This week · \(count) of \(WeekAskModel.maximumQuestions) questions"
    }

    private func pendingView(_ pending: WeekAskPending) -> some View {
        VStack(alignment: .leading, spacing: MFSpacing.xs) {
            Text(pending.question)
                .font(MFFont.bodyEmphasis)
                .foregroundStyle(MFColor.ink)
            HStack(spacing: MFSpacing.s) {
                ProgressView().controlSize(.small)
                Text(pending.step.label + "…")
                    .font(MFFont.callout)
                    .foregroundStyle(MFColor.secondary)
            }
            .accessibilityElement(children: .combine)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    // MARK: Notice and box

    private func noticeBar(_ notice: WeekAskNotice) -> some View {
        HStack(spacing: MFSpacing.s) {
            Text(notice.text)
                .font(MFFont.callout)
                .foregroundStyle(MFColor.onInk)
                .fixedSize(horizontal: false, vertical: true)
            Spacer(minLength: 0)
            if notice.canUndo {
                Button("Undo") { Task { await ask.undoBlock() } }
                    .buttonStyle(.plain)
                    .font(MFFont.calloutEmphasis)
                    .foregroundStyle(MFColor.onInk)
            }
            Button("OK") { ask.dismissNotice() }
                .buttonStyle(.plain)
                .font(MFFont.callout)
                .foregroundStyle(MFColor.onInk)
        }
        .padding(MFSpacing.m)
        .background(RoundedRectangle(cornerRadius: MFRadius.s).fill(MFColor.ink))
        .accessibilityElement(children: .contain)
        .accessibilityAddTraits(.updatesFrequently)
    }

    private var composer: some View {
        HStack(spacing: MFSpacing.s) {
            if compact { scopePill }
            AppleTextField(title: placeholder, text: $ask.draft)
                .font(MFFont.body)
                .focused($focused)
                .onSubmit { Task { await ask.ask() } }
                .accessibilityLabel("Ask about your week")
            if ask.pending != nil {
                Button { ask.stop() } label: {
                    Image(systemName: "stop.fill")
                        .font(MFFont.caption)
                        .foregroundStyle(MFColor.onInk)
                        .padding(MFSpacing.s)
                        .background(Circle().fill(MFColor.ink))
                }
                .buttonStyle(.plain)
                .accessibilityLabel("Stop")
            } else {
                Button { Task { await ask.ask() } } label: {
                    Image(systemName: "arrow.up")
                        .font(MFFont.calloutEmphasis)
                        .foregroundStyle(MFColor.onInk)
                        .padding(MFSpacing.s)
                        .background(Circle().fill(MFColor.ink))
                }
                .buttonStyle(.plain)
                .accessibilityLabel("Ask")
                .disabled(!ask.canAsk)
            }
        }
        .padding(.leading, compact ? MFSpacing.xs : MFSpacing.l)
        .padding(.trailing, MFSpacing.xs)
        .padding(.vertical, MFSpacing.xs)
        .background(Capsule().fill(MFColor.surface))
        .overlay(Capsule().strokeBorder(focused ? MFColor.accent : MFColor.hairlineStrong))
    }

    private var placeholder: String {
        if ask.isFull { return "Continue in Chat to ask more" }
        if compact { return "Ask about this week" }
        return ask.answers.isEmpty ? "When am I free for 2 hours?" : "Ask a follow-up"
    }
}

/// One answer in full: the question, the sentence, the rows from the facts,
/// and where the sentence came from.
struct WeekAnswerView: View {
    var answer: WeekAnswer
    var ask: WeekAskModel
    var isStale: Bool
    @State private var showsAll = false
    @State private var showsHow = false

    private var text: WeekText { ask.text }

    var body: some View {
        VStack(alignment: .leading, spacing: MFSpacing.s) {
            Text(answer.question)
                .font(MFFont.calloutEmphasis)
                .foregroundStyle(MFColor.ink)
            Text(answer.sentence)
                .font(MFFont.callout)
                .foregroundStyle(MFColor.ink)
                .fixedSize(horizontal: false, vertical: true)
                .textSelection(.enabled)
            if isStale {
                Label("Your calendar changed since this answer. Ask again.", systemImage: "exclamationmark.circle")
                    .font(MFFont.caption)
                    .foregroundStyle(MFColor.secondary)
            }
            VStack(alignment: .leading, spacing: MFSpacing.xs) {
                ForEach(shownFacts) { fact in
                    WeekFactRow(fact: fact, ask: ask, text: text, maximumMinutes: maximumMinutes)
                }
            }
            .opacity(isStale ? 0.5 : 1)
            if answer.facts.facts.count > Self.collapsedCount {
                Button(showsAll ? "Show fewer" : "Show all \(answer.facts.facts.count)") { showsAll.toggle() }
                    .buttonStyle(.plain)
                    .font(MFFont.callout)
                    .foregroundStyle(MFColor.accent)
            }
            HStack(alignment: .firstTextBaseline, spacing: MFSpacing.xs) {
                Text(caption)
                    .font(MFFont.caption)
                    .foregroundStyle(MFColor.tertiary)
                    .fixedSize(horizontal: false, vertical: true)
                Spacer(minLength: 0)
                Button { showsHow.toggle() } label: {
                    Image(systemName: showsHow ? "info.circle.fill" : "info.circle")
                }
                .buttonStyle(.plain)
                .font(MFFont.caption)
                .foregroundStyle(MFColor.accent)
                .accessibilityLabel(showsHow ? "Hide how this was answered" : "How was this answered?")
            }
            if showsHow {
                VStack(alignment: .leading, spacing: MFSpacing.xxs) {
                    ForEach(howLines, id: \.self) { line in
                        Text("• " + line).fixedSize(horizontal: false, vertical: true)
                    }
                }
                .font(MFFont.caption)
                .foregroundStyle(MFColor.secondary)
                .padding(MFSpacing.s)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(RoundedRectangle(cornerRadius: MFRadius.s).fill(MFColor.fillSubtle))
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    static let collapsedCount = 3

    private var shownFacts: [WeekFact] {
        showsAll ? answer.facts.facts : Array(answer.facts.facts.prefix(Self.collapsedCount))
    }

    private var maximumMinutes: Int {
        answer.facts.facts.compactMap { fact -> Int? in
            if case .day(let load) = fact.kind { load.bookedMinutes } else { nil }
        }.max() ?? 0
    }

    private var caption: String {
        switch answer.source {
        case .model(let label): "\(label) · checked against your calendar"
        default: answer.source.caption
        }
    }

    private var howLines: [String] {
        var lines = ["Question read as: \(WeekAnswerView.kindName(answer.query))."]
        lines.append("Days, times and counts were worked out by Mference from your calendar, not by a model.")
        switch answer.source {
        case .model(let label):
            lines.append("\(label) wrote the sentence. Every day, time and number in it was checked against those facts.")
        case .fixed(.failedCheck(let label)):
            lines.append("\(label)'s sentence named something your calendar doesn't hold, so Mference wrote this one.")
        case .fixed(.noModel):
            lines.append("No model could run, so Mference wrote the sentence.")
        case .fixed(.noModelNeeded):
            lines.append("No model was run for this.")
        }
        return lines
    }

    static func kindName(_ query: WeekQuery) -> String {
        switch query {
        case .freeTime(let request): "free time, at least \(request.minimumMinutes) minutes"
            + (request.part.map { ", \($0.rawValue)s" } ?? "")
        case .findEvent(let name): "find the event \u{201C}\(name)\u{201D}"
        case .compareDays: "which day is busiest"
        case .daySummary: "what's on one day"
        case .todosDue: "to-dos due"
        case .overlaps: "events that overlap"
        case .outsideWeek: "another week"
        case .other: "not about this week's calendar"
        }
    }
}

/// One row under an answer: an open block (with Block), a day's load, a
/// to-do or an event.
struct WeekFactRow: View {
    var fact: WeekFact
    var ask: WeekAskModel
    var text: WeekText
    var maximumMinutes: Int

    var body: some View {
        switch fact.kind {
        case .openTime(let interval):
            HStack(spacing: MFSpacing.s) {
                Text(text.shortWeekday(interval.start)).font(MFFont.caption.weight(.semibold))
                    .frame(width: 30, alignment: .leading)
                Text(text.range(interval.start, interval.end)).font(MFFont.caption).lineLimit(1)
                Text(text.duration(minutes: Int(interval.duration / 60))).font(MFFont.caption)
                    .foregroundStyle(MFColor.success).lineLimit(1)
                Spacer(minLength: 0)
                if ask.blocked[interval] != nil {
                    Text("Added").font(MFFont.caption.weight(.semibold)).foregroundStyle(MFColor.success)
                } else if !ask.isStale {
                    let block = ask.blockInterval(for: interval)
                    Button("Block") { Task { await ask.block(interval) } }
                        .buttonStyle(.plain)
                        .font(MFFont.caption.weight(.semibold))
                        .foregroundStyle(MFColor.accent)
                        .accessibilityLabel("Block \(text.duration(minutes: Int(block.duration / 60)))")
                        .accessibilityHint("Adds a Focus block event to Calendar, \(text.range(block.start, block.end))")
                }
            }
            .foregroundStyle(MFColor.ink)
            .padding(.horizontal, MFSpacing.s)
            .padding(.vertical, MFSpacing.xs)
            .background(RoundedRectangle(cornerRadius: MFRadius.s).fill(MFColor.successFill))
        case .day(let load):
            HStack(spacing: MFSpacing.s) {
                Text(text.shortWeekday(load.day))
                    .font(MFFont.callout.weight(load.bookedMinutes == maximumMinutes && !load.isPast ? .semibold : .regular))
                    .foregroundStyle(load.isPast ? MFColor.tertiary : MFColor.ink)
                    .frame(width: 36, alignment: .leading)
                GeometryReader { proxy in
                    Capsule().fill(MFColor.fill)
                        .overlay(alignment: .leading) {
                            Capsule().fill(load.isPast ? MFColor.hairlineStrong : MFColor.accent)
                                .frame(width: proxy.size.width * CGFloat(load.bookedMinutes) / CGFloat(max(maximumMinutes, 1)))
                        }
                }
                .frame(height: 6)
                Text("\(text.duration(minutes: load.bookedMinutes)) · \(load.eventCount)")
                    .font(MFFont.caption)
                    .foregroundStyle(MFColor.secondary)
                    .frame(width: 92, alignment: .trailing)
            }
            .accessibilityElement(children: .combine)
        case .todo(let todo):
            row(todo.title, todo.due.map { text.dayWordCapitalized($0) } ?? "No date")
        case .event(let event):
            row(event.title, event.isAllDay ? "\(text.shortWeekday(event.start)) · all day"
                : "\(text.shortWeekday(event.start)) · \(text.range(event.start, event.end))")
        case .overlap(let first, let second, let shared):
            row("\(first.title) and \(second.title)",
                "\(text.shortWeekday(shared.start)) · \(text.range(shared.start, shared.end))")
        }
    }

    private func row(_ title: String, _ detail: String) -> some View {
        HStack(spacing: MFSpacing.s) {
            Text(title).font(MFFont.callout).foregroundStyle(MFColor.ink).lineLimit(2)
            Spacer(minLength: MFSpacing.s)
            Text(detail).font(MFFont.caption).foregroundStyle(MFColor.secondary)
        }
        .padding(.horizontal, MFSpacing.s)
        .padding(.vertical, MFSpacing.xs)
        .overlay(RoundedRectangle(cornerRadius: MFRadius.s).strokeBorder(MFColor.hairline))
        .accessibilityElement(children: .combine)
    }
}

/// A suggestion chip.
struct ChipButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(MFFont.callout)
            .foregroundStyle(MFColor.ink)
            .padding(.horizontal, MFSpacing.m)
            .padding(.vertical, MFSpacing.xs)
            .background(Capsule().fill(configuration.isPressed ? MFColor.fill : MFColor.surface))
            .overlay(Capsule().strokeBorder(MFColor.hairlineStrong))
            .contentShape(Capsule())
    }
}

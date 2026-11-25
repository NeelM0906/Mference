import AppKit
import MferenceAppCore
import MferenceDesign
import SwiftUI

/// Named sizes for Ask about your week.
enum WeekAskMetrics {
    /// The answer card's height before it scrolls inside, so "Show all"
    /// never pushes the to-dos or the box off screen.
    static let cardMaxHeight: CGFloat = 340
    static let dayColumnWidth: CGFloat = 30
}

/// Ask about your week: the suggestion chips, the answer card and the box.
/// Under the to-dos in Day and Week only.
struct WeekAskPanel: View {
    @Bindable var ask: WeekAskModel
    var calendar: CalendarWeekModel
    /// Opens a Private chat with these answers; nil hides the button.
    var continueInChat: (() -> Void)?
    @FocusState private var focused: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: MFSpacing.s) {
            header
            if ask.isIdle {
                FlowLayout(spacing: MFSpacing.xs) {
                    ForEach(ask.chips, id: \.self) { chip in
                        Button(chip.title(ask.text)) { Task { await ask.ask(chip) } }
                            .buttonStyle(ChipButtonStyle())
                    }
                }
                Text("Answers use only this week's events and to-dos. Nothing is saved.")
                    .font(MFFont.caption)
                    .foregroundStyle(MFColor.tertiary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            if !ask.isIdle { WeekAskCard(ask: ask, continueInChat: continueInChat) }
            if let notice = ask.notice { noticeBar(notice) }
            composer
        }
        .onChange(of: calendar.askWeek) { ask.scopeChanged() }
        // The week may have changed while no panel was shown (Year view).
        .onAppear { ask.scopeChanged() }
        // Free-time answers follow the clock while Calendar is on screen:
        // passed openings drop off at the next quarter hour or the next
        // opening's start. No background timer.
        .task(id: ask.refreshKey) {
            while !Task.isCancelled, let next = ask.nextRefresh {
                let wait = max(next.timeIntervalSince(calendar.currentTime), 1)
                try? await Task.sleep(for: .seconds(wait))
                guard !Task.isCancelled else { return }
                ask.refreshFreeTime()
            }
        }
        .onReceive(NotificationCenter.default.publisher(for: NSApplication.didBecomeActiveNotification)) { _ in
            ask.refreshFreeTime()
        }
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
        .padding(.leading, MFSpacing.l)
        .padding(.trailing, MFSpacing.xs)
        .padding(.vertical, MFSpacing.xs)
        .background(Capsule().fill(MFColor.surface))
        .overlay(Capsule().strokeBorder(focused ? MFColor.accent : MFColor.hairlineStrong))
    }

    private var placeholder: String {
        if ask.isFull { return "Continue in Chat to ask more" }
        guard ask.answers.isEmpty else { return "Ask a follow-up" }
        return calendar.mode == .day ? "Ask about this day or week" : "When am I free for 2 hours?"
    }
}

/// The thread: earlier answers in one line each, the latest in full (or
/// the question being answered), Continue in Chat. Its height is capped;
/// past the cap it scrolls inside.
struct WeekAskCard: View {
    var ask: WeekAskModel
    var continueInChat: (() -> Void)?

    var body: some View {
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
            // Sized to its content up to the cap, then it scrolls inside.
            ViewThatFits(in: .vertical) {
                thread
                MFScroll { thread }
            }
            .frame(maxHeight: WeekAskMetrics.cardMaxHeight)
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

    private var thread: some View {
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
            ForEach(answer.warnings, id: \.self) { warning in
                Label(warning, systemImage: "exclamationmark.triangle")
                    .font(MFFont.caption)
                    .foregroundStyle(MFColor.danger)
                    .fixedSize(horizontal: false, vertical: true)
            }
            if isStale {
                Label("Your calendar changed since this answer. Ask again.", systemImage: "exclamationmark.circle")
                    .font(MFFont.caption)
                    .foregroundStyle(MFColor.secondary)
            }
            VStack(alignment: .leading, spacing: MFSpacing.xs) {
                ForEach(shownFacts) { fact in
                    WeekFactRow(fact: fact, ask: ask, text: text)
                }
            }
            .opacity(isStale ? 0.5 : 1)
            if answer.rows.count > Self.collapsedCount {
                Button(showsAll ? "Show fewer" : "Show all \(answer.rows.count)") { showsAll.toggle() }
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
        showsAll ? answer.rows : Array(answer.rows.prefix(Self.collapsedCount))
    }

    private var caption: String { answer.source.caption }

    private var howLines: [String] {
        switch answer.source {
        case .calendar(nil):
            return ["Mference worked this out from your calendar in the app. No model was used."]
        case .calendar(let label?):
            return ["\(label) read your question and the week's schedule, and asked Mference for open time.",
                    "Mference worked the open times out from your calendar, so they are exact."]
        case .model(let label):
            var lines = ["The question was sent with your week's schedule: its events and to-dos, with open and "
                + "booked time already worked out by Mference.", "\(label) wrote the answer from that schedule."]
            if !answer.warnings.isEmpty {
                lines.append("A note above means the answer names a time your calendar doesn't hold.")
            }
            return lines
        }
    }
}

/// One row under an answer: an open block (with Block), an event or a
/// to-do.
struct WeekFactRow: View {
    var fact: WeekFact
    var ask: WeekAskModel
    var text: WeekText

    var body: some View {
        switch fact.kind {
        case .openTime(let interval):
            HStack(spacing: MFSpacing.s) {
                Text(text.shortWeekday(interval.start)).font(MFFont.caption.weight(.semibold))
                    .frame(width: WeekAskMetrics.dayColumnWidth, alignment: .leading)
                Text(text.range(interval.start, interval.end)).font(MFFont.caption).lineLimit(1)
                Text(text.duration(minutes: Int(interval.duration / 60))).font(MFFont.caption)
                    .foregroundStyle(MFColor.success).lineLimit(1)
                Spacer(minLength: 0)
                if ask.blocked[interval] != nil {
                    Text("Added").font(MFFont.caption.weight(.semibold)).foregroundStyle(MFColor.success)
                } else if ask.canBlock, !ask.hasStarted(interval) {
                    let block = ask.blockInterval(for: interval)
                    Button("Block") { Task { await ask.block(interval) } }
                        .buttonStyle(.plain)
                        .font(MFFont.caption.weight(.semibold))
                        .foregroundStyle(MFColor.accent)
                        .accessibilityLabel("Block \(text.duration(minutes: Int(block.duration / 60)))")
                        .accessibilityHint("Adds a Focus block event to Calendar, \(text.range(block.start, block.end))")
                        .disabled(ask.blocking.contains(interval))
                }
            }
            .foregroundStyle(MFColor.ink)
            .padding(.horizontal, MFSpacing.s)
            .padding(.vertical, MFSpacing.xs)
            .background(RoundedRectangle(cornerRadius: MFRadius.s).fill(MFColor.successFill))
        case .todo(let todo):
            row(todo.title, todo.due.map { text.dayWordCapitalized($0) } ?? "No date")
        case .event(let event):
            row(event.title, event.isAllDay ? "\(text.shortWeekday(event.start)) · all day"
                : "\(text.shortWeekday(event.start)) · \(text.range(event.start, event.end))")
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

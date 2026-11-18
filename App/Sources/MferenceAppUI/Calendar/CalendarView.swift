import MferenceAppCore
import MferenceDesign
import SwiftUI

/// Calendar and to-dos (mockup 04): the work week from Apple Calendar,
/// proposals drawn dashed with one-click Add, and the to-do panel.
public struct CalendarView: View {
    @Environment(CalendarWeekModel.self) private var environmentModel: CalendarWeekModel?
    private let injected: CalendarWeekModel?
    private let loadsOnAppear: Bool

    /// The shell's slot: the model comes from the environment.
    public init() {
        injected = nil
        loadsOnAppear = true
    }

    /// Snapshots and previews; `loadsOnAppear` false keeps fixture state.
    public init(model: CalendarWeekModel, loadsOnAppear: Bool = true) {
        injected = model
        self.loadsOnAppear = loadsOnAppear
    }

    public var body: some View {
        if let model = injected ?? environmentModel {
            CalendarContent(model: model)
                .task { if loadsOnAppear { await model.load() } }
        } else {
            AppleEmptyState(systemImage: "calendar", title: "Calendar",
                            message: "Calendar is starting. Try again in a moment.")
        }
    }
}

struct CalendarContent: View {
    @Bindable var model: CalendarWeekModel

    var body: some View {
        HStack(spacing: 0) {
            VStack(alignment: .leading, spacing: MFSpacing.m) {
                header
                switch model.access {
                case .granted:
                    WeekGrid(model: model)
                    if let proposal = model.proposalsThisWeek.first {
                        ProposalCard(proposal: proposal, model: model)
                    }
                case .notAsked:
                    AppleEmptyState(systemImage: "calendar", title: "See your week here",
                                    message: AppPermission.calendar.reason,
                                    actionTitle: "Show my calendar") { Task { await model.requestAccess() } }
                case .blocked:
                    AppleEmptyState(systemImage: "lock", title: "Calendar is turned off for Mference",
                                    message: "Turn on Mference under System Settings > Privacy & Security > Calendars.")
                }
                if let message = model.errorMessage {
                    Text(message)
                        .font(MFFont.callout)
                        .foregroundStyle(MFColor.danger)
                }
            }
            .padding(.horizontal, MFSpacing.xl)
            .padding(.vertical, MFSpacing.l)
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
            Rectangle().fill(MFColor.hairline).frame(width: 1)
            TodoList(model: model)
                .frame(width: 315)
                .background(MFColor.canvas)
        }
        .background(MFColor.surface)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var header: some View {
        HStack(spacing: MFSpacing.m) {
            Text(model.weekTitle)
                .font(MFFont.title)
                .foregroundStyle(MFColor.ink)
                .accessibilityAddTraits(.isHeader)
            Button("Today") { Task { await model.showToday() } }
                .buttonStyle(.appleSecondary)
                .keyboardShortcut("t", modifiers: .command)
            HStack(spacing: MFSpacing.xs) {
                Button { Task { await model.showWeek(offset: -1) } } label: { Image(systemName: "chevron.left") }
                    .accessibilityLabel("Previous week")
                    .keyboardShortcut(.leftArrow, modifiers: .command)
                Button { Task { await model.showWeek(offset: 1) } } label: { Image(systemName: "chevron.right") }
                    .accessibilityLabel("Next week")
                    .keyboardShortcut(.rightArrow, modifiers: .command)
            }
            .buttonStyle(.appleQuiet)
            Spacer()
            HStack(spacing: 0) {
                Text("Week")
                    .font(MFFont.bodyEmphasis)
                    .foregroundStyle(MFColor.onInk)
                    .padding(.horizontal, MFSpacing.l)
                    .padding(.vertical, MFSpacing.s)
                    .background(Capsule().fill(MFColor.ink))
            }
            .padding(MFSpacing.xs)
            .background(Capsule().fill(MFColor.fill))
            .accessibilityLabel("Week view")
        }
    }
}

/// The hour grid with the week's blocks.
struct WeekGrid: View {
    var model: CalendarWeekModel
    /// Hours share the height the screen has; never shorter than this.
    private let minimumHourHeight: CGFloat = 36
    private let gutter: CGFloat = 70

    var body: some View {
        let layout = model.layout
        let hours = Array(layout.firstHour..<layout.lastHour)
        VStack(spacing: 0) {
            HStack(spacing: 0) {
                Color.clear.frame(width: gutter, height: 1)
                ForEach(Array(model.days.enumerated()), id: \.offset) { _, day in
                    dayLabel(day).frame(maxWidth: .infinity)
                }
            }
            .padding(.bottom, MFSpacing.s)
            let outside = model.outsideGrid
            if !outside.isEmpty {
                HStack(alignment: .top, spacing: 0) {
                    Text("All day")
                        .font(MFFont.caption)
                        .foregroundStyle(MFColor.tertiary)
                        .frame(width: gutter, alignment: .leading)
                    ForEach(model.days.indices, id: \.self) { day in
                        VStack(alignment: .leading, spacing: MFSpacing.xxs) {
                            ForEach(outside[day] ?? []) { event in
                                Text(event.isAllDay ? event.title : "\(EventBlockView.time(event.start)) \(event.title)")
                                    .font(MFFont.caption)
                                    .foregroundStyle(MFColor.ink)
                                    .lineLimit(1)
                                    .padding(.horizontal, MFSpacing.s)
                                    .padding(.vertical, MFSpacing.xxs)
                                    .frame(maxWidth: .infinity, alignment: .leading)
                                    .background(RoundedRectangle(cornerRadius: MFRadius.xs).fill(MFColor.fill))
                                    .accessibilityLabel(event.isAllDay ? "\(event.title), all day"
                                                        : "\(event.title), \(EventBlockView.range(event.start, event.end))")
                            }
                        }
                        .padding(.horizontal, MFSpacing.xs)
                        .frame(maxWidth: .infinity, alignment: .topLeading)
                    }
                }
                .padding(.bottom, MFSpacing.s)
            }
            Rectangle().fill(MFColor.hairline).frame(height: 1)
            GeometryReader { outer in
                let hourHeight = max(minimumHourHeight, outer.size.height / CGFloat(max(hours.count, 1)))
                ZStack(alignment: .topLeading) {
                    VStack(spacing: 0) {
                        ForEach(hours, id: \.self) { hour in
                            HStack(alignment: .top, spacing: 0) {
                                Text(hourLabel(hour))
                                    .font(MFFont.callout)
                                    .foregroundStyle(MFColor.tertiary)
                                    .frame(width: gutter, alignment: .leading)
                                    .padding(.top, MFSpacing.xs)
                                Spacer()
                            }
                            .frame(height: hourHeight, alignment: .top)
                            .overlay(alignment: .bottom) { Rectangle().fill(MFColor.hairlineFaint).frame(height: 1) }
                        }
                    }
                    let columnWidth = (outer.size.width - gutter) / CGFloat(max(model.days.count, 1))
                    ForEach(model.blocks) { block in
                        let laneWidth = (columnWidth - MFSpacing.s) / CGFloat(block.lanes)
                        EventBlockView(block: block)
                            .frame(width: laneWidth - MFSpacing.xxs,
                                   height: CGFloat(block.height) * hourHeight - MFSpacing.xs)
                            .offset(x: gutter + CGFloat(block.day) * columnWidth + MFSpacing.xs
                                        + CGFloat(block.lane) * laneWidth,
                                    y: CGFloat(block.top) * hourHeight + MFSpacing.xxs)
                    }
                }
                .frame(width: outer.size.width, height: outer.size.height, alignment: .topLeading)
                .clipped()
            }
        }
    }

    private func dayLabel(_ day: Date) -> some View {
        let calendar = model.layout.calendar
        let weekday = day.formatted(Date.FormatStyle(timeZone: calendar.timeZone).weekday(.abbreviated))
        let number = calendar.component(.day, from: day)
        let today = model.isToday(day)
        return HStack(spacing: MFSpacing.xs) {
            Text(weekday).font(MFFont.body)
            Text("\(number)").font(MFFont.bodyEmphasis)
        }
        .foregroundStyle(today ? MFColor.onInk : MFColor.secondary)
        .padding(.horizontal, today ? MFSpacing.m : 0)
        .padding(.vertical, today ? MFSpacing.xxs : 0)
        .background(Capsule().fill(today ? MFColor.ink : .clear))
        .accessibilityElement(children: .combine)
        .accessibilityAddTraits(today ? .isSelected : [])
    }

    private func hourLabel(_ hour: Int) -> String {
        switch hour {
        case 0: "12 AM"
        case 12: "12 PM"
        case 13...: "\(hour - 12) PM"
        default: "\(hour) AM"
        }
    }
}

/// One block: solid event, dashed proposal, struck-through moving event.
struct EventBlockView: View {
    var block: WeekBlock

    var body: some View {
        VStack(alignment: .leading, spacing: MFSpacing.xxs) {
            Text(block.title)
                .font(MFFont.calloutEmphasis)
                .strikethrough(isMoving)
            if let detail {
                Text(detail)
                    .font(MFFont.callout)
                    .strikethrough(isMoving)
            }
        }
        .foregroundStyle(foreground)
        .padding(.horizontal, MFSpacing.s + MFSpacing.xxs)
        .padding(.vertical, MFSpacing.xs + MFSpacing.xxs)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .background(RoundedRectangle(cornerRadius: MFRadius.s).fill(fill))
        .overlay(RoundedRectangle(cornerRadius: MFRadius.s)
            .strokeBorder(isProposal ? MFColor.accent : .clear,
                          style: StrokeStyle(lineWidth: 1.5, dash: isProposal ? [5, 4] : [])))
        .accessibilityElement(children: .combine)
        .accessibilityLabel(accessibilityText)
    }

    private var isProposal: Bool { if case .proposal = block.kind { true } else { false } }
    private var isMoving: Bool { if case .moving = block.kind { true } else { false } }

    private var detail: String? {
        switch block.kind {
        case .proposal(let proposal):
            "\(Self.time(proposal.start)) · proposed from \(proposal.source.kindLabel)"
        case .moving(let event):
            "\(Self.time(event.start)), moving"
        case .event(let event):
            block.height >= 1 ? Self.range(event.start, event.end) : nil
        }
    }

    private var accessibilityText: String {
        switch block.kind {
        case .proposal(let proposal): "Proposed: \(proposal.title), \(Self.range(proposal.start, proposal.end))"
        case .moving(let event): "\(event.title), moving"
        case .event(let event): "\(event.title), \(Self.range(event.start, event.end))"
        }
    }

    private var fill: Color {
        switch block.kind {
        case .proposal: MFColor.accentFillFaint
        case .moving: MFColor.fill
        case .event(let event): Self.palette(for: event).fill
        }
    }

    private var foreground: Color {
        switch block.kind {
        case .proposal: MFColor.accentStrong
        case .moving: MFColor.tertiary
        case .event(let event): Self.palette(for: event).text
        }
    }

    /// Calendars get the mockup's tints, by calendar name so they stay put.
    static func palette(for event: CalendarEvent) -> (fill: Color, text: Color) {
        let tints: [(Color, Color)] = [(MFColor.accentFill, MFColor.accentStrong),
                                       (MFColor.successFill, MFColor.success),
                                       (MFColor.previewFill, MFColor.preview),
                                       (MFColor.fill, MFColor.ink)]
        let index = event.calendarName.unicodeScalars.reduce(0) { ($0 &* 31 &+ Int($1.value)) & 0xFFFF } % tints.count
        return tints[index]
    }

    static func time(_ date: Date) -> String {
        date.formatted(date: .omitted, time: .shortened)
    }

    static func range(_ start: Date, _ end: Date) -> String {
        "\(time(start)) to \(time(end))"
    }
}

/// "Move the design review to Wed 11:00?" with Dismiss and Add.
struct ProposalCard: View {
    var proposal: ProposedEvent
    var model: CalendarWeekModel

    var body: some View {
        AppleActionCard {
            HStack(spacing: MFSpacing.m) {
                let note = Text(proposal.note.map { " \($0)" } ?? "")
                    .font(MFFont.body).foregroundStyle(MFColor.secondary)
                Text("\(Text(question).font(MFFont.bodyEmphasis).foregroundStyle(MFColor.ink))\(note)")
                    .fixedSize(horizontal: false, vertical: true)
                Spacer(minLength: MFSpacing.m)
                Button("Dismiss") { model.dismiss(proposal) }
                    .buttonStyle(.appleSecondary)
                Button("Add") { Task { await model.add(proposal) } }
                    .buttonStyle(.applePrimary)
                    .disabled(model.addingProposals.contains(proposal.id))
                    .keyboardShortcut(.return, modifiers: .command)
            }
        }
    }

    private var question: String {
        let when = proposal.start.formatted(.dateTime.weekday(.abbreviated).hour().minute())
        return proposal.replacesEventID == nil
            ? "Add \(proposal.title) on \(when)?"
            : "Move \(proposal.title.lowercased()) to \(when)?"
    }
}

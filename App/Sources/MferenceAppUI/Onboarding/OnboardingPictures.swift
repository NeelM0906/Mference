import MferenceAppCore
import MferenceDesign
import SwiftUI

/// The timing of the first run's moving pictures, as pure functions of the
/// seconds since a step appeared, so a still render (snapshots, Reduce
/// Motion) shows each picture's resting frame.
nonisolated enum OnboardingMotion {
    /// 0 before `delay`, 1 after `delay + duration`, eased in between.
    static func settle(_ t: Double, delay: Double, duration: Double) -> Double {
        guard duration > 0 else { return t >= delay ? 1 : 0 }
        let x = min(max((t - delay) / duration, 0), 1)
        return x * x * (3 - 2 * x)
    }

    /// Welcome: tile `index` of four flies in, one after another.
    static func tile(_ index: Int, at t: Double) -> Double {
        settle(t, delay: 0.2 + 0.18 * Double(index), duration: 0.7)
    }

    /// Models: the memory bar grows to its sizes.
    static func memoryBar(at t: Double) -> Double {
        settle(t, delay: 0.1, duration: 0.9)
    }

    /// Permissions: how many of `count` cards are lit; they light one by one,
    /// hold, then start again.
    static func litPermissions(_ count: Int, at t: Double) -> Int {
        let step = 0.9
        let period = step * Double(count) + 1.8
        let phase = t.truncatingRemainder(dividingBy: period)
        return min(count, Int(phase / step) + 1)
    }

    /// Mail: letter `index` of `count` moves from Inbox to its pile; the
    /// sort repeats.
    static func letter(_ index: Int, of count: Int, at t: Double) -> Double {
        let period = 0.8 + 0.75 * Double(count) + 1.6
        let phase = t.truncatingRemainder(dividingBy: period)
        return settle(phase, delay: 0.8 + 0.75 * Double(index), duration: 0.55)
    }

    /// A moment at which every picture is at rest: tiles settled, the bar
    /// grown, every permission lit, every letter sorted.
    static func restingTime(lettersCount: Int, permissionsCount: Int) -> Double {
        let lettersDone = 0.8 + 0.75 * Double(lettersCount - 1) + 0.6
        let permissionsDone = 0.9 * Double(permissionsCount - 1) + 0.05
        return max(lettersDone, permissionsDone, 2)
    }
}

/// One moving picture at the top of each first-run step.
struct OnboardingPicture: View {
    let step: OnboardingView.Step
    let library: any ModelLibraryModel
    @Environment(\.mfRendersStatically) private var rendersStatically
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var start = Date.now

    private var isStill: Bool { rendersStatically || reduceMotion }

    var body: some View {
        TimelineView(.animation(minimumInterval: 1.0 / 30, paused: isStill)) { context in
            let t = isStill
                ? OnboardingMotion.restingTime(lettersCount: OnboardingPictureMetrics.letters.count,
                                               permissionsCount: OnboardingPictureMetrics.permissions.count)
                : context.date.timeIntervalSince(start)
            content(at: t)
        }
        .frame(maxWidth: .infinity)
        .frame(height: OnboardingPictureMetrics.height)
        .background(RoundedRectangle(cornerRadius: MFRadius.l).fill(MFColor.accentFillFaint))
        .overlay(RoundedRectangle(cornerRadius: MFRadius.l)
            .strokeBorder(MFColor.hairline, lineWidth: MFLayout.hairline))
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(Self.description(step))
        .onChange(of: step) { start = .now }
    }

    @ViewBuilder
    private func content(at t: Double) -> some View {
        switch step {
        case .welcome: WelcomePicture(t: t)
        case .models: ModelsPicture(library: library, t: t)
        case .permissions: PermissionsPicture(t: t)
        case .mail: MailPicture(t: t)
        }
    }

    static func description(_ step: OnboardingView.Step) -> String {
        switch step {
        case .welcome: "Chat, Mail, Calendar and Scan, all on this Mac."
        case .models: "This Mac's memory, with room for the models, and Quick's download."
        case .permissions: "Calendar, Reminders, Full Disk Access and Camera, each asked only when first needed."
        case .mail: "Mail sorted from the Inbox into Important and To-dos."
        }
    }
}

enum OnboardingPictureMetrics {
    static let height: CGFloat = 196
    static let tile = CGSize(width: 92, height: 72)
    static let screen = CGSize(width: 452, height: 112)
    static let screenStand = CGSize(width: 96, height: 10)
    static let tileTravel: CGFloat = 70
    static let barHeight: CGFloat = 26
    static let barWidth: CGFloat = 560
    static let gridColumns = 24
    static let gridRows = 3
    static let gridCell: CGFloat = 12
    static let gridGap: CGFloat = 3
    static let permissionCard = CGSize(width: 128, height: 104)
    static let pile = CGSize(width: 150, height: 146)
    /// Below each pile's name.
    static let letterTop: CGFloat = 32
    static let letter = CGSize(width: 104, height: 22)
    static let tiles: [(String, String)] = [
        ("Chat", "bubble.left.and.bubble.right"),
        ("Mail", "envelope"),
        ("Calendar", "calendar"),
        ("Scan", "doc.viewfinder"),
    ]
    static let permissions: [(String, String, String)] = [
        ("Calendar", "calendar", "When you open Calendar"),
        ("Reminders", "checklist", "When you add a to-do"),
        ("Full Disk Access", "internaldrive", "When you connect Mail"),
        ("Camera", "camera", "When you scan a receipt"),
    ]
    /// Each letter's pile: 1 Important, 2 To-dos.
    static let letters: [(String, Int)] = [
        ("Lease renewal", 1),
        ("Pay the invoice", 2),
        ("Flight moved", 1),
        ("Send the slides", 2),
    ]
}

/// Welcome: Chat, Mail, Calendar and Scan settle into a Mac.
private struct WelcomePicture: View {
    let t: Double

    var body: some View {
        VStack(spacing: MFSpacing.s) {
            VStack(spacing: 0) {
                HStack(spacing: MFSpacing.m) {
                    ForEach(Array(OnboardingPictureMetrics.tiles.enumerated()), id: \.offset) { index, tile in
                        let p = OnboardingMotion.tile(index, at: t)
                        VStack(spacing: MFSpacing.xs) {
                            Image(systemName: tile.1)
                                .font(MFFont.title)
                                .foregroundStyle(MFColor.accent)
                            Text(tile.0)
                                .font(MFFont.caption)
                                .foregroundStyle(MFColor.ink)
                        }
                        .frame(width: OnboardingPictureMetrics.tile.width, height: OnboardingPictureMetrics.tile.height)
                        .background(RoundedRectangle(cornerRadius: MFRadius.m).fill(MFColor.surface))
                        .overlay(RoundedRectangle(cornerRadius: MFRadius.m)
                            .strokeBorder(MFColor.hairline, lineWidth: MFLayout.hairline))
                        .opacity(p)
                        .scaleEffect(0.8 + 0.2 * p)
                        .offset(y: -OnboardingPictureMetrics.tileTravel * (1 - p))
                    }
                }
                .frame(width: OnboardingPictureMetrics.screen.width, height: OnboardingPictureMetrics.screen.height)
                .background(RoundedRectangle(cornerRadius: MFRadius.m).fill(MFColor.fill))
                .overlay(RoundedRectangle(cornerRadius: MFRadius.m)
                    .strokeBorder(MFColor.hairlineStrong, lineWidth: MFLayout.strokeEmphasis))
                RoundedRectangle(cornerRadius: MFRadius.xs)
                    .fill(MFColor.hairlineStrong)
                    .frame(width: OnboardingPictureMetrics.screenStand.width,
                           height: OnboardingPictureMetrics.screenStand.height)
            }
            HStack(spacing: MFSpacing.xs) {
                Image(systemName: "lock")
                    .foregroundStyle(MFColor.success)
                Text("No account · no usage data sent")
                    .foregroundStyle(MFColor.secondary)
            }
            .font(MFFont.caption)
            .opacity(OnboardingMotion.tile(4, at: t))
        }
    }
}

/// Models: this Mac's memory with Quick (and Deep, where offered) sized to
/// their estimated memory, and Quick's download filling a grid.
private struct ModelsPicture: View {
    let library: any ModelLibraryModel
    let t: Double

    private var quick: ModelEntry? { library.entries.first { $0.id == AppModel.quick.modelID } }
    private var deep: ModelEntry? {
        library.entries.first { $0.id == AppModel.deep.modelID && $0.isOffered(on: library.machine) }
    }

    /// Estimated resident bytes, from the size on disk (`MemoryFit`).
    private func memory(_ model: AppModel, _ entry: ModelEntry?) -> UInt64? {
        guard let size = entry?.sizeBytes, size > 0 else { return nil }
        return MemoryFit(physicalMemory: library.machine.physicalMemoryBytes)
            .estimatedResidentBytes(model, sizeBytes: UInt64(size))
    }

    var body: some View {
        let physical = Double(max(library.machine.physicalMemoryBytes, 1))
        let grow = OnboardingMotion.memoryBar(at: t)
        let quickMemory = memory(.quick, quick)
        let deepMemory = memory(.deep, deep)
        let width = OnboardingPictureMetrics.barWidth
        VStack(alignment: .leading, spacing: MFSpacing.m) {
            HStack {
                Text("This Mac: \(ModelFormat.memory(library.machine.physicalMemoryBytes)) memory")
                    .font(MFFont.calloutSemibold)
                    .foregroundStyle(MFColor.ink)
                Spacer()
            }
            ZStack(alignment: .leading) {
                RoundedRectangle(cornerRadius: MFRadius.s).fill(MFColor.fill)
                HStack(spacing: MFSpacing.xxs) {
                    if let quickMemory {
                        segment("Quick 4B · \(MenuBarSummary.gigabytes(quickMemory))", fill: MFColor.accent,
                                text: MFColor.onInk,
                                width: width * min(Double(quickMemory) / physical, 1) * grow)
                    }
                    if let deepMemory {
                        segment("Deep 27B · \(MenuBarSummary.gigabytes(deepMemory))", fill: MFColor.accentFill,
                                text: MFColor.accentStrong,
                                width: width * min(Double(deepMemory) / physical, 1) * grow)
                    }
                }
            }
            .frame(width: width, height: OnboardingPictureMetrics.barHeight)
            DownloadGrid(fraction: Self.fraction(quick?.state))
            Text(Self.caption(quick))
                .font(MFFont.caption)
                .foregroundStyle(MFColor.secondary)
        }
        .frame(width: width)
    }

    private func segment(_ label: String, fill: Color, text: Color, width: CGFloat) -> some View {
        Text(label)
            .font(MFFont.chip)
            .foregroundStyle(text)
            .lineLimit(1)
            .padding(.horizontal, MFSpacing.s)
            .frame(width: max(width, 0), alignment: .leading)
            .frame(maxHeight: .infinity)
            .background(RoundedRectangle(cornerRadius: MFRadius.s).fill(fill))
            .clipped()
    }

    /// How full Quick's grid is: 1 once installed, the download's share while
    /// it downloads or is paused, 0 before.
    static func fraction(_ state: ModelEntryState?) -> Double {
        guard let state else { return 0 }
        if state.isInstalled { return 1 }
        return state.transfer?.fraction ?? 0
    }

    static func caption(_ quick: ModelEntry?) -> String {
        guard let quick else { return "Quick isn't in this build yet" }
        let size = quick.sizeBytes.map(ModelFormat.bytes) ?? ""
        switch quick.state {
        case .loaded, .ready, .loading, .loadFailed: return "Quick 4B is installed"
        case .downloading(let progress):
            return "Installing Quick 4B · \(ModelFormat.transfer(progress))"
        case .paused(let progress): return "Quick 4B paused · \(ModelFormat.transfer(progress))"
        case .downloadFailed: return "Quick 4B didn't download"
        case .notInstalled: return size.isEmpty ? "Quick 4B installs now" : "Quick 4B installs now · \(size)"
        }
    }
}

/// A grid of blocks that fills as a download progresses.
private struct DownloadGrid: View {
    let fraction: Double

    var body: some View {
        let columns = OnboardingPictureMetrics.gridColumns
        let total = columns * OnboardingPictureMetrics.gridRows
        let filled = Int((Double(total) * fraction).rounded(.down))
        VStack(spacing: OnboardingPictureMetrics.gridGap) {
            ForEach(0..<OnboardingPictureMetrics.gridRows, id: \.self) { row in
                HStack(spacing: OnboardingPictureMetrics.gridGap) {
                    ForEach(0..<columns, id: \.self) { column in
                        let index = column * OnboardingPictureMetrics.gridRows + row
                        RoundedRectangle(cornerRadius: MFRadius.xs / 2)
                            .fill(index < filled ? MFColor.accent : MFColor.fill)
                            .frame(width: OnboardingPictureMetrics.gridCell, height: OnboardingPictureMetrics.gridCell)
                    }
                }
            }
        }
        .animation(.easeOut(duration: 0.4), value: filled)
    }
}

/// Permissions: four cards light one by one, each with the moment it is asked.
private struct PermissionsPicture: View {
    let t: Double

    var body: some View {
        let cards = OnboardingPictureMetrics.permissions
        let lit = OnboardingMotion.litPermissions(cards.count, at: t)
        VStack(spacing: MFSpacing.m) {
            HStack(spacing: MFSpacing.m) {
                ForEach(Array(cards.enumerated()), id: \.offset) { index, card in
                    let on = index < lit
                    VStack(spacing: MFSpacing.xs) {
                        Image(systemName: card.1)
                            .font(MFFont.title)
                            .foregroundStyle(on ? MFColor.accent : MFColor.tertiary)
                        Text(card.0)
                            .font(MFFont.calloutSemibold)
                            .foregroundStyle(MFColor.ink)
                            .lineLimit(1)
                            .minimumScaleFactor(0.8)
                        Text(card.2)
                            .font(MFFont.caption)
                            .foregroundStyle(MFColor.secondary)
                            .multilineTextAlignment(.center)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    .padding(MFSpacing.s)
                    .frame(width: OnboardingPictureMetrics.permissionCard.width,
                           height: OnboardingPictureMetrics.permissionCard.height)
                    .background(RoundedRectangle(cornerRadius: MFRadius.m).fill(MFColor.surface))
                    .overlay(RoundedRectangle(cornerRadius: MFRadius.m)
                        .strokeBorder(on ? MFColor.accent : MFColor.hairline,
                                      lineWidth: on ? MFLayout.strokeEmphasis : MFLayout.hairline))
                    .opacity(on ? 1 : 0.55)
                    .animation(.easeOut(duration: 0.3), value: on)
                }
            }
            Text("Nothing is asked now")
                .font(MFFont.caption)
                .foregroundStyle(MFColor.secondary)
        }
    }
}

/// Mail: letters move from the Inbox into Important and To-dos.
private struct MailPicture: View {
    let t: Double

    var body: some View {
        let letters = OnboardingPictureMetrics.letters
        let pile = OnboardingPictureMetrics.pile
        let gap = MFSpacing.xl
        let pileX: [CGFloat] = [0, pile.width + gap, 2 * (pile.width + gap)]
        ZStack(alignment: .topLeading) {
            HStack(spacing: gap) {
                ForEach(["Inbox", "Important", "To-dos"], id: \.self) { name in
                    VStack {
                        Text(name)
                            .font(MFFont.sectionHeader)
                            .tracking(MFFont.sectionHeaderTracking)
                            .foregroundStyle(MFColor.secondary)
                            .padding(.top, MFSpacing.s)
                        Spacer()
                    }
                    .frame(width: pile.width, height: pile.height)
                    .background(RoundedRectangle(cornerRadius: MFRadius.m).fill(MFColor.surface))
                    .overlay(RoundedRectangle(cornerRadius: MFRadius.m)
                        .strokeBorder(MFColor.hairline, lineWidth: MFLayout.hairline))
                }
            }
            ForEach(Array(letters.enumerated()), id: \.offset) { index, letter in
                let p = OnboardingMotion.letter(index, of: letters.count, at: t)
                let sortedBefore = letters.prefix(index).filter { $0.1 == letter.1 }.count
                let fromX = (pile.width - OnboardingPictureMetrics.letter.width) / 2
                let fromY = OnboardingPictureMetrics.letterTop + CGFloat(index) * (OnboardingPictureMetrics.letter.height + MFSpacing.xs)
                let toX = pileX[letter.1] + fromX
                let toY = OnboardingPictureMetrics.letterTop + CGFloat(sortedBefore) * (OnboardingPictureMetrics.letter.height + MFSpacing.xs)
                HStack(spacing: MFSpacing.xs) {
                    Image(systemName: "envelope.fill")
                        .foregroundStyle(letter.1 == 1 ? MFColor.accent : MFColor.preview)
                    Text(letter.0)
                        .lineLimit(1)
                        .foregroundStyle(MFColor.ink)
                }
                .font(MFFont.chip)
                .padding(.horizontal, MFSpacing.s)
                .frame(width: OnboardingPictureMetrics.letter.width, height: OnboardingPictureMetrics.letter.height,
                       alignment: .leading)
                .background(RoundedRectangle(cornerRadius: MFRadius.xs).fill(MFColor.fill))
                .offset(x: fromX + (toX - fromX) * p, y: fromY + (toY - fromY) * p)
            }
        }
        .frame(width: 3 * pile.width + 2 * gap, height: pile.height, alignment: .topLeading)
    }
}

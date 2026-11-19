import CoreGraphics
import SwiftUI
import Testing
@testable import MferenceMobileUI

/// The shell's layout decision at the measured window sizes (decision D4).
@Suite("Shell layout")
struct ShellLayoutTests {
    @Test func iPhonesAndTheClosedDuoUseTabs() {
        #expect(ShellLayout(windowSize: CGSize(width: 402, height: 874)) == .tabs)  // iPhone 18 Pro
        #expect(ShellLayout(windowSize: CGSize(width: 375, height: 812)) == .tabs)  // the narrowest held
        #expect(ShellLayout(windowSize: CGSize(width: 440, height: 956)) == .tabs)  // the widest iPhone
        #expect(ShellLayout(windowSize: CGSize(width: 466, height: 678)) == .tabs)  // Duo closed
    }

    @Test func theOpenDuoSplitsInTwoPanes() {
        #expect(ShellLayout(windowSize: CGSize(width: 951, height: 669)) == .twoPane)
        #expect(!ShellLayout(windowSize: CGSize(width: 951, height: 669)).sheetsBelowFold)
        #expect(!ShellLayout(windowSize: CGSize(width: 402, height: 874)).sheetsBelowFold)
    }

    @Test func theTurnedDuoIsOneColumnAtMost560Wide() throws {
        let layout = ShellLayout(windowSize: CGSize(width: 669, height: 951))
        guard case .turned(let turned) = layout else {
            Issue.record("669 × 951 should be the turned layout, got \(layout)")
            return
        }
        #expect(turned.columnWidth == 560)
        #expect(turned.foldY == 475.5)
        #expect(layout.sheetsBelowFold)
        // Still the compact (one-column) mode, never two panes.
        #expect(FoldGeometry(width: 669).mode == .compact)
    }

    @Test func theColumnKeepsTheScreenMargin() {
        // At the turned threshold the 560 cap still applies.
        let atThreshold = TurnedFold(windowSize: CGSize(width: 600, height: 900))
        #expect(atThreshold.columnWidth == 560)
        #expect(ShellLayout(windowSize: CGSize(width: 600, height: 900)) == .turned(atThreshold))
        // Narrower than 560 + 2 × 16, the margin wins.
        #expect(TurnedFold(windowSize: CGSize(width: 580, height: 900)).columnWidth == 548)
    }

    @Test func theTurnedThresholdIs600Wide() {
        #expect(ShellLayout(windowSize: CGSize(width: 599, height: 900)) == .tabs)
        #expect(ShellLayout(windowSize: CGSize(width: 600, height: 900)) != .tabs)
        // A square window is not portrait, so not turned.
        #expect(ShellLayout(windowSize: CGSize(width: 650, height: 650)) == .tabs)
    }

    @Test func sizesNeverGoNegative() {
        let tiny = TurnedFold(windowSize: CGSize(width: 20, height: 20))
        #expect(tiny.columnWidth == 0)
    }

    @Test func onlyTheTurnedLayoutHasAColumn() {
        #expect(ShellLayout(windowSize: CGSize(width: 669, height: 951)).columnWidth == 560)
        #expect(ShellLayout(windowSize: CGSize(width: 466, height: 678)).columnWidth == nil)
        #expect(ShellLayout(windowSize: CGSize(width: 951, height: 669)).columnWidth == nil)
    }

    @Test func aWideButShortWindowIsNotTurned() {
        // Below the two-pane width but landscape: tabs, not the turned column.
        #expect(ShellLayout(windowSize: CGSize(width: 680, height: 500)) == .tabs)
    }
}

@Suite("Sheet detents")
struct PhoneSheetDetentsTests {
    @Test func withoutALimitTheDetentsPassThrough() {
        let detents: Set<PresentationDetent> = [.medium, .large]
        #expect(PhoneSheetDetents.detents(detents, belowFold: false) == detents)
    }

    @Test func belowTheFoldEverySheetOpensAtMedium() {
        #expect(PhoneSheetDetents.detents([.medium, .large], belowFold: true) == [.medium])
        #expect(PhoneSheetDetents.detents([.large], belowFold: true) == [.medium])
    }
}

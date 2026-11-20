import Foundation
import SwiftData
import Testing
@testable import MferenceAppCore

/// The meeting owners as `AppServices` holds them: one of each, sharing the
/// store, the summary hooked to the end of a meeting, and launch recovery
/// run once. Model-free (`AppServices.preview`).
@MainActor
@Suite struct MeetingsWiringTests {
    @Test func appServicesHoldsTheMeetingOwners() async throws {
        let speech = FakeSpeechService()
        let services = AppServices.preview(speech: speech)
        let noteID = try await services.meetings.start(MeetingStartOptions(summaryWhenStopped: true))
        #expect(services.speech.status.liveSession?.noteID == noteID)
        await services.meetings.stop()
        await services.meetings.waitUntilFinished()
        // The meeting's note is in the app's store, and its summary was asked for.
        let notes = try services.modelContainer.mainContext.fetch(FetchDescriptor<Note>())
        #expect(notes.map(\.id) == [noteID])
        #expect(services.summaries.status(for: noteID) != .none)
        await services.summaries.waitUntilIdle()
    }

    @Test func launchRecoveryRunsOnce() async throws {
        let speech = FakeSpeechService()
        let services = AppServices.preview(speech: speech)
        let first = services.startLaunchRecovery()
        let second = services.startLaunchRecovery()
        #expect(first == second)
        await first.value
        #expect(speech.calls.filter { $0 == .recoverInterruptedRecordings }.count == 1)
        #expect(services.launchRecovery == first)
    }
}

import Foundation
@testable import MferenceAppCore
import Testing

@MainActor
@Suite struct SpeechPreferencesTests {
    @Test func defaultsFollowTheSpec() {
        let preferences = SpeechPreferences(store: InMemorySettingsStore())
        #expect(preferences.language == .automatic)
        #expect(preferences.cleanUpFillers, "Clean up filler words is on by default (4.8)")
        #expect(preferences.microphoneID == nil)
        #expect(preferences.callAudio == .allApps, "All apps is the safe default (spec 12)")
        #expect(preferences.speakerLabels, "D3")
        #expect(preferences.keepAudio == .keep, "D4")
        #expect(preferences.summaryModel == .quick && preferences.summaryWhenYouStop, "D5")
        #expect(preferences.writesMeetingSummary)
    }

    @Test func choicesPersistAcrossLaunches() {
        let store = InMemorySettingsStore()
        let first = SpeechPreferences(store: store)
        first.language = .fixed("de")
        first.cleanUpFillers = false
        first.microphoneID = "usb-1"
        first.callAudio = .app(bundleID: "us.zoom.xos", name: "Zoom")
        first.speakerLabels = false
        first.keepAudio = .deleteAfterTranscribing
        first.summaryModel = .off
        first.summaryWhenYouStop = false
        first.fileSpeakerLabels = false
        first.fileSummary = false

        let second = SpeechPreferences(store: store)
        #expect(second.language == .fixed("de"))
        #expect(!second.cleanUpFillers)
        #expect(second.microphoneID == "usb-1")
        #expect(second.callAudio == .app(bundleID: "us.zoom.xos", name: "Zoom"))
        #expect(!second.speakerLabels)
        #expect(second.keepAudio == .deleteAfterTranscribing)
        #expect(second.summaryModel == .off && !second.summaryWhenYouStop)
        #expect(!second.fileSpeakerLabels && !second.fileSummary)
    }

    @Test func storedFormsRoundTripAndRejectJunk() {
        for source in [CallAudioSource.off, .allApps, .app(bundleID: "com.microsoft.teams2", name: "Microsoft Teams")] {
            #expect(SpeechPreferences.decodeCallAudio(SpeechPreferences.encode(source)) == source)
        }
        #expect(SpeechPreferences.decodeCallAudio("app\u{1F}\u{1F}Zoom") == nil)
        #expect(SpeechPreferences.decodeCallAudio("everything") == nil)
        for code in SpeechLanguage.supportedCodes {
            #expect(SpeechPreferences.decodeLanguage(SpeechPreferences.encode(.fixed(code))) == .fixed(code))
        }
        #expect(SpeechPreferences.decodeLanguage("auto") == .automatic)
        #expect(SpeechPreferences.decodeLanguage("ja") == nil, "only the 25 languages")
        let store = InMemorySettingsStore([SpeechPreferences.Key.language: "xx",
                                           SpeechPreferences.Key.callAudio: "nonsense"])
        let preferences = SpeechPreferences(store: store)
        #expect(preferences.language == .automatic && preferences.callAudio == .allApps)
    }

    @Test func optionsCarryTheChoices() {
        let preferences = SpeechPreferences(store: InMemorySettingsStore())
        preferences.language = .fixed("fr")
        preferences.cleanUpFillers = false
        preferences.microphoneID = "usb-1"
        preferences.keepAudio = .deleteAfterTranscribing
        preferences.callAudio = .off
        let dictation = preferences.dictationOptions
        #expect(dictation.language == .fixed("fr") && !dictation.cleanUpFillers && dictation.microphoneID == "usb-1")
        #expect(dictation.endOfSpeechSilence == .milliseconds(500))
        let note = UUID()
        let meeting = preferences.meetingOptions(noteID: note)
        #expect(meeting.noteID == note && meeting.callAudio == .off && !meeting.keepAudio)
        #expect(meeting.speakerLabels && meeting.language == .fixed("fr") && meeting.microphoneID == "usb-1")
        preferences.summaryModel = .off
        #expect(!preferences.writesMeetingSummary, "Settings' Off wins over the popover switch")
    }
}

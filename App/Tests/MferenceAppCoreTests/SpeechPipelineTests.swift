@preconcurrency import AVFoundation
import Foundation
@testable import MferenceAppCore
import Testing

/// The speech pipeline's pure stages (Notes spec 6.3, 6.4): segmentation,
/// filler clean-up, echo de-duplication and speaker alignment. Fixture PCM
/// and scripted probabilities only: no model, microphone, tap or TCC.
@Suite struct SpeechSegmenterTests {
    /// 100 ms frames so durations read easily.
    static let frame = 1_600
    static func frames(_ pattern: String) -> [(samples: [Float], probability: Float)] {
        pattern.map { ([Float](repeating: $0 == "S" ? 0.3 : 0, count: frame), $0 == "S" ? 0.95 : 0.1) }
    }

    static func run(_ pattern: String, configuration: Segmenter.Configuration = .init()) -> [SegmenterEvent] {
        var segmenter = Segmenter(track: .microphone, configuration: configuration)
        var events: [SegmenterEvent] = []
        for (samples, probability) in frames(pattern) {
            events += segmenter.process(frame: samples, probability: probability)
        }
        return events + segmenter.flush()
    }

    static func phrases(_ events: [SegmenterEvent]) -> [Phrase] {
        events.compactMap { if case .finished(let phrase) = $0 { phrase } else { nil } }
    }

    @Test func aPhraseEndsAfterHalfASecondOfSilence() {
        // 0.3 s silence, 1 s speech, 0.4 s pause (kept), 0.5 s speech, 0.6 s silence.
        let events = Self.run("___SSSSSSSSSS____SSSSS______")
        let phrases = Self.phrases(events)
        #expect(phrases.count == 1, "a 400 ms pause does not end the phrase")
        let phrase = phrases[0]
        #expect(abs(phrase.start - 0.2) < 1e-9, "starts 100 ms (padding) before the first voiced frame")
        // Last voiced frame ends at 2.2 s; 100 ms of padding kept.
        #expect(abs(phrase.end - 2.3) < 1e-9)
        #expect(phrase.samples.count == SpeechAudio.samples(phrase.end - phrase.start))
        if case .started(let id, let start) = events.first {
            #expect(id == phrase.id && abs(start - 0.2) < 1e-9)
        } else {
            Issue.record("first event is .started")
        }
    }

    @Test func twoPhrasesWhenTheSilenceIsLongEnough() {
        let phrases = Self.phrases(Self.run("SSSSS_____SSSS_____"))
        #expect(phrases.count == 2)
        #expect(phrases[0].end <= phrases[1].start)
        #expect(Set(phrases.map(\.id)).count == 2)
    }

    @Test func longSpeechIsCutAtFifteenSecondsAndGoesOn() {
        let phrases = Self.phrases(Self.run(String(repeating: "S", count: 200) + "______"))
        #expect(phrases.count == 2)
        #expect(abs(phrases[0].end - phrases[0].start - 15) < 1e-9, "the model's 15 s window")
        #expect(abs(phrases[1].start - phrases[0].end) < 1e-9, "the next phrase starts at the cut")
        #expect(phrases.allSatisfy { $0.end - $0.start <= 15 + 1e-9 })
    }

    @Test func aClickIsDropped() {
        let events = Self.run("___S______")
        #expect(Self.phrases(events).isEmpty)
        #expect(events.contains { if case .dropped = $0 { true } else { false } })
    }

    @Test func flushEndsAnOpenPhrase() {
        let phrases = Self.phrases(Self.run("__SSSSS"))
        #expect(phrases.count == 1)
        #expect(abs(phrases[0].end - 0.7) < 1e-9)
    }

    @Test func jumpMovesTheClockForANewFile() {
        var segmenter = Segmenter(track: .call)
        _ = segmenter.jump(to: 120)
        var events: [SegmenterEvent] = []
        for (samples, probability) in Self.frames("SSSS______") {
            events += segmenter.process(frame: samples, probability: probability)
        }
        #expect(Self.phrases(events).first.map { $0.start >= 120 } == true)
        #expect(Self.phrases(events).first?.track == .call)
    }

    @Test func trackSegmenterFramesChunksOfAnySize() async throws {
        let voice = ScriptedVoiceActivity(frameLength: 1_600)
        let segmenter = TrackSegmenter(track: .microphone, detector: try await voice.makeStream(),
                                       frameLength: voice.frameLength, configuration: .init())
        let pcm = FakeAudioSource.tone(seconds: 0.5, amplitude: 0) + FakeAudioSource.tone(seconds: 1)
            + FakeAudioSource.tone(seconds: 1, amplitude: 0)
        var events: [SegmenterEvent] = []
        var index = 0
        for size in [700, 3_000, 1_234, 9_999, 100_000] where index < pcm.count {
            let end = min(pcm.count, index + size)
            events += try await segmenter.feed(Array(pcm[index..<end]))
            index = end
        }
        events += await segmenter.flush()
        let phrases = SpeechSegmenterTests.phrases(events)
        #expect(phrases.count == 1)
        #expect(abs(phrases[0].start - 0.4) < 0.01)
        #expect(abs(phrases[0].end - 1.6) < 0.01)
    }
}

@Suite struct FillerFilterTests {
    @Test(arguments: [
        ("en", "Um, I think we should, uh, ship it.", "I think we should, ship it."),
        ("en", "So erm we wait hmm.", "So we wait."),
        ("fr", "Euh, je pense que, euh, c'est bon.", "Je pense que, c'est bon."),
        ("de", "Ähm, wir sollten äh morgen anfangen.", "Wir sollten morgen anfangen."),
        ("es", "Eh, creo que ehm vamos bien.", "Creo que vamos bien."),
        ("it", "Ehm, domani eh partiamo.", "Domani partiamo."),
        ("pt", "Hum, acho que hã está certo.", "Acho que está certo."),
        ("nl", "Ehm, we gaan uh morgen.", "We gaan morgen."),
    ])
    func removesEachLanguagesFillers(language: String, input: String, expected: String) {
        #expect(FillerFilter.clean(input, languageCode: language) == expected)
    }

    @Test func neverInsideWords() {
        let text = "The umbrella and hummus are uhmazing; Ähre, ähnlich, euhemerism, Heuer, hump."
        #expect(FillerFilter.clean(text, languageCode: "en") == text)
        #expect(FillerFilter.clean(text, languageCode: "de") == text)
        #expect(FillerFilter.clean(text, languageCode: nil) == text)
    }

    @Test func fillersOfOneLanguageAreWordsInAnother() {
        // "em" is Portuguese for "in"; "er" is German for "he".
        #expect(FillerFilter.clean("Moro em Lisboa.", languageCode: "pt") == "Moro em Lisboa.")
        #expect(FillerFilter.clean("Er kommt morgen.", languageCode: "de") == "Er kommt morgen.")
        #expect(FillerFilter.clean("Moro em Lisboa.", languageCode: nil) == "Moro em Lisboa.")
    }

    @Test func repeatedWords() {
        #expect(FillerFilter.clean("I I think the the plan works.", languageCode: "en") == "I think the plan works.")
        #expect(FillerFilter.clean("We we should go.", languageCode: "en") == "We should go.")
        // Grammatical doubles and numbers stay; so do repeats split by punctuation.
        #expect(FillerFilter.clean("I know that that is true.", languageCode: "en") == "I know that that is true.")
        #expect(FillerFilter.clean("Nous nous sommes vus.", languageCode: "fr") == "Nous nous sommes vus.")
        #expect(FillerFilter.clean("Call 20 20 now.", languageCode: "en") == "Call 20 20 now.")
        #expect(FillerFilter.clean("No, no, no.", languageCode: "en") == "No, no, no.")
    }

    @Test func punctuationStaysTidy() {
        #expect(FillerFilter.clean("We ship today um.", languageCode: "en") == "We ship today.")
        #expect(FillerFilter.clean("Is it done, uh?", languageCode: "en") == "Is it done?")
        #expect(FillerFilter.clean("Done. Um, next one.", languageCode: "en") == "Done. Next one.")
        #expect(FillerFilter.clean("Um.", languageCode: "en") == "")
        #expect(FillerFilter.clean("", languageCode: "en") == "")
    }

    @Test func unknownLanguageUsesTheSafeList() {
        #expect(FillerFilter.clean("Um, äh, euh, okay.", languageCode: nil) == "Okay.")
        #expect(FillerFilter.clean("Um, okay.", languageCode: "fi") == "Okay.")
    }
}

@Suite struct EchoDeduplicatorTests {
    static func segment(_ track: TrackKind, _ start: TimeInterval, _ end: TimeInterval, _ text: String) -> TranscriptSegment {
        TranscriptSegment(track: track, start: start, end: end,
                          speakerID: track == .microphone ? SpeakerInfo.youID : "s1", text: text, isFinal: true)
    }

    @Test func aMicSegmentRepeatingTheCallIsAnEcho() {
        let call = [Self.segment(.call, 10, 14, "Let's move the launch to Friday, okay?")]
        let mic = Self.segment(.microphone, 10.2, 14.3, "let's move the launch to friday")
        #expect(EchoDeduplicator.isEcho(mic, of: call))
    }

    @Test func diacriticsAndPunctuationDoNotMatter() {
        let call = [Self.segment(.call, 0, 3, "Café à côté, s'il vous plaît.")]
        #expect(EchoDeduplicator.isEcho(Self.segment(.microphone, 0.5, 3, "cafe a cote s il vous plait"), of: call))
    }

    @Test func differentWordsOrNoOverlapAreKept() {
        let call = [Self.segment(.call, 10, 14, "Let's move the launch to Friday.")]
        #expect(!EchoDeduplicator.isEcho(Self.segment(.microphone, 10, 14, "No, Monday works better for me."), of: call))
        #expect(!EchoDeduplicator.isEcho(Self.segment(.microphone, 20, 24, "Let's move the launch to Friday."), of: call))
        // 3 of 5 words is below 0.8.
        #expect(!EchoDeduplicator.isEcho(Self.segment(.microphone, 10, 14, "move the launch tomorrow please"), of: call))
    }

    @Test func callTextAcrossTwoSegmentsCounts() {
        let call = [Self.segment(.call, 0, 2, "We agreed on the plan."), Self.segment(.call, 2, 4, "Ship it Friday.")]
        #expect(EchoDeduplicator.isEcho(Self.segment(.microphone, 1, 4, "the plan ship it friday"), of: call))
    }
}

@Suite struct SpeakerAlignmentTests {
    static func call(_ start: TimeInterval, _ end: TimeInterval, _ speaker: String) -> TranscriptSegment {
        TranscriptSegment(track: .call, start: start, end: end, speakerID: speaker, text: "x", isFinal: true)
    }

    @Test func aSegmentTakesTheSpeakerItOverlapsMost() {
        let turns = [SpeakerTurn(speaker: "a", start: 0, end: 4), SpeakerTurn(speaker: "b", start: 4, end: 10)]
        #expect(SpeakerAlignment.speaker(from: 3, to: 6, in: turns) == "b")
        #expect(SpeakerAlignment.speaker(from: 0, to: 3, in: turns) == "a")
        #expect(SpeakerAlignment.speaker(from: 11, to: 12, in: turns) == nil)
    }

    @Test func renamesCarryOverByBestOverlap() {
        // Live: L1 talks 0-10, L2 talks 10-20. The final pass calls them F9 and F3.
        let live = [Self.call(0, 4, "L1"), Self.call(5, 9, "L1"), Self.call(10, 15, "L2"), Self.call(16, 20, "L2")]
        let final = [SpeakerTurn(speaker: "F9", start: 0, end: 9.5), SpeakerTurn(speaker: "F3", start: 9.5, end: 20)]
        let mapping = SpeakerAlignment.mapping(liveSegments: live, finalTurns: final)
        #expect(mapping == ["L1": "F9", "L2": "F3"])
        let relabelled = SpeakerAlignment.relabel(live, finalTurns: final, mapping: mapping)
        #expect(relabelled.map(\.speakerID) == ["F9", "F9", "F3", "F3"])
        #expect(relabelled.map(\.id) == live.map(\.id), "segment ids never change")
    }

    @Test func liveSpeakersTheFinalPassMergesMapToOne() {
        let live = [Self.call(0, 4, "L1"), Self.call(5, 9, "L2")]
        let mapping = SpeakerAlignment.mapping(liveSegments: live, finalTurns: [SpeakerTurn(speaker: "F1", start: 0, end: 10)])
        #expect(mapping == ["L1": "F1", "L2": "F1"])
    }

    @Test func microphoneSegmentsStayYou() {
        let you = TranscriptSegment(track: .microphone, start: 0, end: 3, speakerID: SpeakerInfo.youID, text: "x", isFinal: true)
        let relabelled = SpeakerAlignment.relabel([you], finalTurns: [SpeakerTurn(speaker: "F1", start: 0, end: 10)], mapping: [:])
        #expect(relabelled[0].speakerID == SpeakerInfo.youID)
    }

    @Test func speakersAreNumberedInOrderOfFirstWords() {
        let you = TranscriptSegment(track: .microphone, start: 0, end: 1, speakerID: SpeakerInfo.youID, text: "x", isFinal: true)
        let speakers = SpeakerAlignment.speakers(for: [Self.call(5, 6, "b"), you, Self.call(2, 3, "a")])
        #expect(speakers.map(\.id) == [SpeakerInfo.youID, "a", "b"])
        #expect(speakers.map(\.name) == ["You", "Speaker 2", "Speaker 3"])
        #expect(Set(speakers.map(\.colorIndex)).count == 3)
    }
}

@Suite struct SpeechAudioTests {
    @Test func levelsRunFromSilenceToFullScale() {
        #expect(SpeechAudio.level([]) == 0)
        #expect(SpeechAudio.level([Float](repeating: 0, count: 100)) == 0)
        #expect(SpeechAudio.level([Float](repeating: 1, count: 100)) == 1)
        let quiet = SpeechAudio.level(FakeAudioSource.tone(seconds: 0.1, amplitude: 0.01))
        let loud = SpeechAudio.level(FakeAudioSource.tone(seconds: 0.1, amplitude: 0.5))
        #expect(quiet > 0 && quiet < loud && loud < 1)
    }

    @Test func resamplesStereo48kToMono16k() throws {
        let format = try #require(AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 48_000, channels: 2,
                                                interleaved: false))
        let resampler = try AudioResampler(inputFormat: format)
        var total = 0
        for _ in 0..<10 {
            let buffer = try #require(AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 4_800))
            buffer.frameLength = 4_800
            for channel in 0..<2 {
                for index in 0..<4_800 {
                    buffer.floatChannelData![channel][index] = 0.5 * sin(2 * .pi * 440 * Float(index) / 48_000)
                }
            }
            total += try resampler.convert(buffer).count
        }
        // One second in; the converter keeps its filter's latency (about
        // 14 ms) in flight, which comes out with the next buffer.
        #expect(total <= 16_000 && total > 16_000 - 512)
    }

    @Test func aPinnedLanguageLabelsEveryPhrase() {
        #expect(PhraseLanguage.code(for: "hi", setting: .fixed("de")) == "de")
        #expect(PhraseLanguage.code(for: "ok", setting: .automatic) == nil, "too short to guess")
        let guess = PhraseLanguage.guess("Guten Morgen, wir besprechen heute den Entwurf und verteilen die Aufgaben.")
        #expect(guess.map { SpeechLanguage.supportedCodes.contains($0) } ?? true)
    }
}

import AppKit
import AVFoundation
import CoreText
import Foundation
import ImageIO
@testable import MferenceAppCore
import Testing
import UniformTypeIdentifiers

/// Find end to end with the real search model: a folder of everyday files
/// (PDF, Word, RTF, Excel, PowerPoint, a screenshot, notes) indexed by the
/// real `FindService`, then queried. A model run: only with
/// `MFERENCE_FIND_MODEL` set to the EmbeddingGemma 2 text GGUF, under the
/// machine lock (AGENTS.md). `MFERENCE_FIND_HELPER` (the built MferenceMac
/// executable) reads files in the helper process as the app does;
/// `MFERENCE_FIND_REPORT` writes the timings and results there.
/// `MFERENCE_FIND_MEDIA_MODEL` (the mmproj GGUF) adds the photo, video and
/// audio run (`MFERENCE_FIND_MEDIA_REPORT` for its report).
@MainActor
@Suite(.serialized) struct FindLiveModelTests {
    nonisolated static let environment = ProcessInfo.processInfo.environment
    nonisolated static let modelPath = environment["MFERENCE_FIND_MODEL"]
    nonisolated static let mediaModelPath = environment["MFERENCE_FIND_MEDIA_MODEL"]

    struct Expectation {
        var query: String
        /// Any of these file names in the top three passes.
        var expected: Set<String>
    }

    static let expectations = [
        Expectation(query: "lease clause about pets", expected: ["Lease 2418 Mission St.pdf"]),
        Expectation(query: "screenshot with the wifi password", expected: ["Screenshot 2026-08-19 at 21.03.png"]),
        Expectation(query: "how much money did we make last quarter",
                    expected: ["Board update Q3.pptx", "Q3 close.xlsx", "Investor letter draft.docx"]),
        Expectation(query: "when does the kitchen renovation have to be finished",
                    expected: ["Renovation contract.docx"]),
        Expectation(query: "dentist appointment", expected: ["Smile Dental confirmation.rtf"]),
        Expectation(query: "what should I pack for the camping trip", expected: ["Camping packing list.txt"]),
        Expectation(query: "renters insurance covers dog bites", expected: ["Renters insurance policy.pdf"]),
    ]

    @Test(.enabled(if: modelPath != nil, "set MFERENCE_FIND_MODEL"))
    func everydayFilesEndToEnd() async throws {
        let folder = try FindTempFolder()
        let indexFolder = try FindTempFolder()
        try Self.makeCorpus(in: folder.url)
        var report: [String] = ["# Find live-model run", ""]

        let settings = FindSettings(store: InMemorySettingsStore())
        settings.learnOnlyOnPower = false
        let helper = Self.environment["MFERENCE_FIND_HELPER"].map { URL(filePath: $0) }
        // The app's readers: a pool of helper processes.
        let extractor: any FindExtracting = helper.map { executable in
            FindReaderPool((0..<FindReaderPool.size()).map { _ in HelperFindExtractor(executable: executable) })
        } ?? InProcessFindExtractor()
        let model = URL(filePath: Self.modelPath!)
        let find = FindService(settings: settings, dependencies: .init(
            indexDirectory: indexFolder.url, model: PresentFindModel(url: model),
            makeEmbedder: { path, _ in FindEmbedderLive(modelPath: path) }, extractor: extractor, watchFolders: true))
        report.append("- reader: \(helper == nil ? "in process" : "helper process (\(helper!.lastPathComponent))")")

        // Turn on and wait for words, then meaning.
        let started = Date()
        await find.turnOn(roots: [folder.url])
        var wordsReady: Double?
        var firstDocument: Double?
        for _ in 0..<1_200 {
            if firstDocument == nil,
               await find.searchWords(FindQuery("cleaning booked")).strong.first?.name == "Smile Dental confirmation.rtf" {
                firstDocument = Date().timeIntervalSince(started)
            }
            if wordsReady == nil, find.status.totalFiles > 0, find.status.readFiles + find.status.skippedFiles
                == find.status.totalFiles { wordsReady = Date().timeIntervalSince(started) }
            if find.status.totalFiles > 0, find.status.learnedFiles + find.status.skippedFiles == find.status.totalFiles,
               find.status.phase == .upToDate { break }
            try await Task.sleep(for: .milliseconds(100))
        }
        let learned = Date().timeIntervalSince(started)
        let status = find.status
        report.append("- files: \(status.totalFiles), read \(status.readFiles), learned \(status.learnedFiles), skipped \(status.skippedFiles)")
        report.append(String(format: "- a document searchable by words after %.2f s; every file's words after %.2f s; meaning learned after %.2f s",
                             firstDocument ?? -1, wordsReady ?? -1, learned))
        report.append("- index on disk: \(status.indexBytes) bytes")
        let skipped = await find.skippedFiles()
        for file in skipped { report.append("  - skipped \(file.url.lastPathComponent): \(file.reason)") }
        #expect(status.learnedFiles + status.skippedFiles == status.totalFiles, "meaning was not learned for every file")
        #expect(skipped.isEmpty)

        // Queries: words while typing, then the full search (cold, then warm).
        report.append("")
        report.append("| query | top 3 | reasons of #1 | words ms | full ms | pass |")
        report.append("|---|---|---|---|---|---|")
        var passed = 0
        for expectation in Self.expectations {
            let words = await find.searchWords(FindQuery(expectation.query))
            let clock = ContinuousClock()
            var full = FindResults.empty
            let duration = await clock.measure { full = await find.search(FindQuery(expectation.query)) }
            let top = full.strong.prefix(3).map(\.name)
            let ok = !expectation.expected.isDisjoint(with: top)
            if ok { passed += 1 }
            let reasons = full.strong.first.map { $0.reasons.map(\.title).joined(separator: ", ") } ?? "—"
            report.append("| \(expectation.query) | \(top.joined(separator: "; ")) | \(reasons) | "
                + String(format: "%.1f | %.1f | %@ |", words.seconds * 1000,
                         Double(duration.components.attoseconds) / 1e15 + Double(duration.components.seconds) * 1000,
                         ok ? "yes" : "NO"))
            #expect(ok, "“\(expectation.query)” → \(top)")
        }
        report.append("")
        report.append("\(passed) of \(Self.expectations.count) queries found the expected file in the top three.")

        // A question with no answer in these files.
        let none = await find.search(FindQuery("volcanic eruptions on mars"))
        report.append("- no-answer query “volcanic eruptions on mars”: \(none.strong.count) strong results"
            + (none.strong.isEmpty ? "" : " (\(none.strong.prefix(3).map(\.name).joined(separator: ", ")))"))

        // The folder watcher: a new file is found without a rescan.
        let added = Date()
        try Data("The parking permit for the street must be renewed before March 3 at the city office.".utf8)
            .write(to: folder.url.appending(path: "Notes/parking.txt"))
        var foundAfter: Double?
        for _ in 0..<150 {
            let results = await find.search(FindQuery("parking permit renewal"))
            if results.strong.first?.name == "parking.txt" {
                foundAfter = Date().timeIntervalSince(added)
                break
            }
            try await Task.sleep(for: .milliseconds(200))
        }
        report.append(String(format: "- new file found by the watcher after %.1f s", foundAfter ?? -1))
        #expect(foundAfter != nil, "the folder watcher did not pick up a new file")

        // Chat's tool over the same index.
        let toolbox = try #require(find.makeFilesToolbox())
        let outcome = await toolbox.run(ToolCall(id: "1", name: "search_files",
                                                 arguments: .object(["query": .string("pet deposit in my lease")]),
                                                 argumentsJSON: "{}"))
        report.append("- chat search_files “pet deposit in my lease”: sources \(outcome.citations.map { "[\($0.number)] \($0.label)" }.joined(separator: ", "))")
        #expect(outcome.citations.first?.label == "Lease 2418 Mission St.pdf")

        await find.shutdown()
        let text = report.joined(separator: "\n")
        print(text)
        if let path = Self.environment["MFERENCE_FIND_REPORT"] {
            try Data(text.utf8).write(to: URL(filePath: path))
        }
    }

    // MARK: Photos, video and audio

    struct MediaExpectation {
        var query: String
        var expected: String
        /// For video and audio: the match must be in this stretch (seconds).
        var during: ClosedRange<Double>?
    }

    /// Photos named the way a camera names them, so only what they show can
    /// match. The pictures every Mac has under /Library/User Pictures.
    static let photos: [(source: String, name: String)] = [
        ("Animals/Zebra", "IMG_2041.heic"), ("Flowers/Sunflower", "IMG_2042.heic"),
        ("Animals/Parrot", "IMG_2043.heic"), ("Nature/Earth", "IMG_2044.heic"),
        ("Nature/Cactus", "IMG_2045.heic"), ("Fun/Fortune Cookie", "IMG_2046.heic"),
        ("Instruments/Guitar", "IMG_2047.heic"), ("Animals/Owl", "IMG_2048.heic"),
        ("Flowers/Red Rose", "IMG_2049.heic"), ("Instruments/Drum", "IMG_2050.heic"),
        ("Fun/Medal", "IMG_2051.heic"), ("Nature/Leaf", "IMG_2052.heic"),
    ]

    static let mediaExpectations = [
        MediaExpectation(query: "photo of a zebra", expected: "IMG_2041.heic"),
        MediaExpectation(query: "yellow flower", expected: "IMG_2042.heic"),
        MediaExpectation(query: "a colourful bird", expected: "IMG_2043.heic"),
        MediaExpectation(query: "the planet seen from space", expected: "IMG_2044.heic"),
        MediaExpectation(query: "cactus", expected: "IMG_2045.heic"),
        MediaExpectation(query: "fortune cookie", expected: "IMG_2046.heic"),
        MediaExpectation(query: "old newspaper about the moon landing", expected: "IMG_0001.jpg"),
        MediaExpectation(query: "penguin at the zoo", expected: "Zoo day.mov", during: 0...6),
        MediaExpectation(query: "someone playing the piano", expected: "Zoo day.mov", during: 6...12),
        MediaExpectation(query: "lightning storm", expected: "Zoo day.mov", during: 12...18),
        MediaExpectation(query: "voice memo about the leaking kitchen sink", expected: "Voice memo 3.aiff"),
        MediaExpectation(query: "recording of the budget meeting", expected: "Recording 12.aiff"),
        // Words only spoken, never in a name.
        MediaExpectation(query: "call the plumber", expected: "Voice memo 3.aiff"),
        MediaExpectation(query: "marketing spend and hiring plans", expected: "Recording 12.aiff"),
    ]

    @Test(.enabled(if: modelPath != nil && mediaModelPath != nil, "set MFERENCE_FIND_MODEL and _MEDIA_MODEL"))
    func photosVideoAndAudioEndToEnd() async throws {
        let folder = try FindTempFolder()
        let indexFolder = try FindTempFolder()
        try await Self.makeMediaCorpus(in: folder.url)
        var report: [String] = ["# Find live-model run: photos, video and audio", ""]

        let settings = FindSettings(store: InMemorySettingsStore())
        settings.learnOnlyOnPower = false
        settings.readMedia = true
        let helper = Self.environment["MFERENCE_FIND_HELPER"].map { URL(filePath: $0) }
        // The app's readers: a pool of helper processes.
        let extractor: any FindExtracting = helper.map { executable in
            FindReaderPool((0..<FindReaderPool.size()).map { _ in HelperFindExtractor(executable: executable) })
        } ?? InProcessFindExtractor()
        let find = FindService(settings: settings, dependencies: .init(
            indexDirectory: indexFolder.url, model: PresentFindModel(url: URL(filePath: Self.modelPath!)),
            mediaModel: PresentFindModel(url: URL(filePath: Self.mediaModelPath!)),
            makeEmbedder: { FindEmbedderLive(modelPath: $0, mediaModelPath: $1) }, extractor: extractor,
            watchFolders: false))
        report.append("- reader: \(helper == nil ? "in process" : "helper process (\(helper!.lastPathComponent))")")

        let started = Date()
        await find.turnOn(roots: [folder.url])
        for _ in 0..<6_000 {
            let status = find.status
            if status.totalFiles > 0, status.learnedFiles + status.skippedFiles == status.totalFiles,
               status.phase == FindIndexStatus.Phase.upToDate { break }
            try await Task.sleep(for: .milliseconds(100))
        }
        let learned = Date().timeIntervalSince(started)
        let status = find.status
        report.append("- files: \(status.totalFiles), learned \(status.learnedFiles), skipped \(status.skippedFiles)")
        report.append(String(format: "- everything learned after %.1f s (%d photos, a 18 s narrated video, 2 voice memos, fillers)",
                             learned, Self.photos.count + 1))
        report.append("- index on disk: \(status.indexBytes) bytes")
        let skipped = await find.skippedFiles()
        for file in skipped { report.append("  - skipped \(file.url.lastPathComponent): \(file.reason)") }
        #expect(skipped.isEmpty)
        #expect(status.learnedFiles == status.totalFiles, "not every file was learned")

        report.append("")
        report.append("| query | top 3 | #1 reasons | #1 at | ms | pass |")
        report.append("|---|---|---|---|---|---|")
        var passed = 0
        for expectation in Self.mediaExpectations {
            let clock = ContinuousClock()
            var results = FindResults.empty
            let duration = await clock.measure { results = await find.search(FindQuery(expectation.query)) }
            let top = Array(results.strong.prefix(3))
            let hit = top.first { $0.name == expectation.expected }
            var ok = hit != nil
            if ok, let during = expectation.during { ok = hit?.time.map { during.contains($0) } ?? false }
            if ok { passed += 1 }
            let first = results.strong.first
            report.append("| \(expectation.query) | \(top.map(\.name).joined(separator: "; ")) | "
                + "\(first.map { $0.reasons.map(\.title).joined(separator: ", ") } ?? "—") | \(first?.location ?? "—") | "
                + String(format: "%.0f | %@ |", Double(duration.components.attoseconds) / 1e15
                         + Double(duration.components.seconds) * 1000, ok ? "yes" : "NO"))
            #expect(ok, "“\(expectation.query)” → \(top.map { "\($0.name)@\($0.location ?? "-")" })")
        }
        report.append("")
        report.append("\(passed) of \(Self.mediaExpectations.count) queries found the expected file (and moment) in the top three.")

        // Words and documents still work beside media.
        let words = await find.search(FindQuery("dentist cleaning on thursday"))
        report.append("- text beside media: “dentist cleaning on thursday” → \(words.strong.first?.name ?? "nothing")")
        #expect(words.strong.first?.name == "Dentist.txt")
        let none = await find.search(FindQuery("volcanic eruptions on mars"))
        report.append("- no-answer query “volcanic eruptions on mars”: \(none.strong.count) strong results"
            + (none.strong.isEmpty ? "" : " (\(none.strong.prefix(3).map(\.name).joined(separator: ", ")))"))

        await find.shutdown()
        let text = report.joined(separator: "\n")
        print(text)
        if let path = Self.environment["MFERENCE_FIND_MEDIA_REPORT"] {
            try Data(text.utf8).write(to: URL(filePath: path))
        }
    }

    static func makeMediaCorpus(in root: URL) async throws {
        let manager = FileManager.default
        let photos = root.appending(path: "Pictures/Camera Roll")
        try manager.createDirectory(at: photos, withIntermediateDirectories: true)
        for photo in Self.photos {
            // The account pictures are links into /System; copy the pictures.
            try manager.copyItem(at: URL(filePath: "/Library/User Pictures/\(photo.source).heic").resolvingSymlinksInPath(),
                                 to: photos.appending(path: photo.name))
        }
        // A real photo of a newspaper (llama.cpp's own sample, when present).
        if let newspaper = environment["MFERENCE_FIND_NEWSPAPER"] {
            try manager.copyItem(at: URL(filePath: newspaper), to: photos.appending(path: "IMG_0001.jpg"))
        }
        let audio = root.appending(path: "Recordings")
        try manager.createDirectory(at: audio, withIntermediateDirectories: true)
        try speak("Note to self. Call the plumber tomorrow morning about the leaking kitchen sink, the cabinet underneath is soaked.",
                  to: audio.appending(path: "Voice memo 3.aiff"))
        try speak("Welcome everyone to the quarterly budget meeting. First on the agenda is marketing spend for next year, then hiring.",
                  to: audio.appending(path: "Recording 12.aiff"))
        try speak("Here we are at the zoo! Look at the penguins, then the piano player by the gate, and wow, a storm is coming.",
                  to: audio.appending(path: "narration.aiff"))
        let movies = root.appending(path: "Movies")
        try manager.createDirectory(at: movies, withIntermediateDirectories: true)
        try await slideshow(["Animals/Penguin", "Instruments/Piano", "Nature/Lightning"].map {
            URL(filePath: "/Library/User Pictures/\($0).heic").resolvingSymlinksInPath()
        }, seconds: 6, narration: audio.appending(path: "narration.aiff"), to: movies.appending(path: "Zoo day.mov"))
        try manager.removeItem(at: audio.appending(path: "narration.aiff"))
        try Data("Dentist: cleaning with Dr. Lin on Thursday at 9:30.".utf8).write(to: root.appending(path: "Dentist.txt"))
        for (index, text) in ["Grocery list: oats, milk, apples.", "Gym: legs on Tuesday.",
                              "Call the bank about the card.", "Book the flight to Lisbon."].enumerated() {
            try Data(text.utf8).write(to: root.appending(path: "note \(index + 1).txt"))
        }
    }

    /// Speech from the Mac's own voice.
    static func speak(_ text: String, to url: URL) throws {
        let say = Process()
        say.executableURL = URL(filePath: "/usr/bin/say")
        say.arguments = ["-o", url.path, text]
        try say.run()
        say.waitUntilExit()
        #expect(say.terminationStatus == 0)
    }

    /// A video of `images`, `seconds` each, with `narration` as its sound.
    static func slideshow(_ images: [URL], seconds: Int, narration: URL, to url: URL) async throws {
        let silent = url.deletingLastPathComponent().appending(path: "silent.mov")
        let width = 640, height = 480, fps: Int32 = 2
        let writer = try AVAssetWriter(outputURL: silent, fileType: .mov)
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: [
            AVVideoCodecKey: AVVideoCodecType.h264, AVVideoWidthKey: width, AVVideoHeightKey: height,
        ])
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input, sourcePixelBufferAttributes: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32ARGB,
            kCVPixelBufferWidthKey as String: width, kCVPixelBufferHeightKey as String: height,
        ])
        writer.add(input)
        #expect(writer.startWriting())
        writer.startSession(atSourceTime: .zero)
        var frame: Int64 = 0
        for imageURL in images {
            let source = try #require(CGImageSourceCreateWithURL(imageURL as CFURL, nil))
            let image = try #require(CGImageSourceCreateImageAtIndex(source, 0, nil))
            for _ in 0..<(seconds * Int(fps)) {
                while !input.isReadyForMoreMediaData { try await Task.sleep(for: .milliseconds(5)) }
                var buffer: CVPixelBuffer?
                CVPixelBufferCreate(nil, width, height, kCVPixelFormatType_32ARGB, nil, &buffer)
                let pixels = try #require(buffer)
                CVPixelBufferLockBaseAddress(pixels, [])
                let context = try #require(CGContext(
                    data: CVPixelBufferGetBaseAddress(pixels), width: width, height: height, bitsPerComponent: 8,
                    bytesPerRow: CVPixelBufferGetBytesPerRow(pixels), space: CGColorSpaceCreateDeviceRGB(),
                    bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue))
                // Aspect fill.
                let scale = max(Double(width) / Double(image.width), Double(height) / Double(image.height))
                let drawn = CGSize(width: Double(image.width) * scale, height: Double(image.height) * scale)
                context.draw(image, in: CGRect(x: (Double(width) - drawn.width) / 2, y: (Double(height) - drawn.height) / 2,
                                               width: drawn.width, height: drawn.height))
                CVPixelBufferUnlockBaseAddress(pixels, [])
                adaptor.append(pixels, withPresentationTime: CMTime(value: frame, timescale: fps))
                frame += 1
            }
        }
        input.markAsFinished()
        await writer.finishWriting()
        #expect(writer.status == .completed)

        // Mux the narration in.
        let composition = AVMutableComposition()
        let video = AVURLAsset(url: silent)
        let sound = AVURLAsset(url: narration)
        let videoTrack = try #require(try await video.loadTracks(withMediaType: .video).first)
        let soundTrack = try #require(try await sound.loadTracks(withMediaType: .audio).first)
        let length = try await video.load(.duration)
        let soundLength = try await sound.load(.duration)
        try composition.addMutableTrack(withMediaType: .video, preferredTrackID: kCMPersistentTrackID_Invalid)?
            .insertTimeRange(CMTimeRange(start: .zero, duration: length), of: videoTrack, at: .zero)
        try composition.addMutableTrack(withMediaType: .audio, preferredTrackID: kCMPersistentTrackID_Invalid)?
            .insertTimeRange(CMTimeRange(start: .zero, duration: CMTimeMinimum(length, soundLength)), of: soundTrack,
                             at: .zero)
        let export = try #require(AVAssetExportSession(asset: composition, presetName: AVAssetExportPresetHighestQuality))
        try await export.export(to: url, as: .mov)
        try FileManager.default.removeItem(at: silent)
    }

    // MARK: Corpus

    static func makeCorpus(in root: URL) throws {
        func write(_ path: String, _ text: String) throws {
            let url = root.appending(path: path)
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try Data(text.utf8).write(to: url)
        }
        try pdf(root.appending(path: "Home/Lease 2418 Mission St.pdf"), pages: [
            "RESIDENTIAL LEASE AGREEMENT. 13. ALTERATIONS. Tenant shall not paint, wallpaper or make alterations without the prior written consent of Landlord. Rent is due on the first of each month.",
            "14. ANIMALS. Tenant may keep one cat or one dog under 30 lb, subject to a 500 dollar refundable deposit and written notice to Landlord. 15. NOISE. Tenant shall not disturb other occupants.",
        ])
        try pdf(root.appending(path: "Insurance/Renters insurance policy.pdf"), pages: [
            "SECTION II. PERSONAL LIABILITY. Coverage E pays for bodily injury to others caused by a domestic animal you own, up to the limit shown in the declarations.",
        ])
        try pdf(root.appending(path: "Downloads/Cedar Cabin house manual.pdf"), pages: [
            "Welcome to the cabin. Check-in is after 4 pm. The hot tub is open from 8 am to 10 pm. Please keep the doors locked when you leave.",
        ])
        try screenshot(root.appending(path: "Desktop/Screenshot 2026-08-19 at 21.03.png"), lines: [
            "Welcome to the Cedar Cabin", "Network: Cedar-Guest-5G", "Password: lakeside2026", "Checkout 11 am",
        ])
        try document(root.appending(path: "Home/Renovation contract.docx"),
                     "RENOVATION CONTRACT. 6. SCHEDULE. Work shall be completed within ten weeks of the permit date, barring change orders signed by both parties. 7. WARRANTY. One year on workmanship.",
                     type: .officeOpenXML)
        try document(root.appending(path: "Work/Investor letter draft.docx"),
                     "Dear investors, this quarter we grew sales 18 percent while holding spending flat. Our new plan launches in January.",
                     type: .officeOpenXML)
        try document(root.appending(path: "Health/Smile Dental confirmation.rtf"),
                     "Smile Dental. Your cleaning with Dr. Lin is booked for Thursday, December 4 at 9:30. Please arrive ten minutes early.",
                     type: .rtf)
        try office(root.appending(path: "Work/Board update Q3.pptx"), members: [
            "ppt/slides/slide1.xml": #"<p:sld><a:p><a:t>Board update, third quarter</a:t></a:p></p:sld>"#,
            "ppt/slides/slide2.xml": #"<p:sld><a:p><a:t>Financials: revenue 4.2 million dollars, up 18 percent on the second quarter</a:t></a:p></p:sld>"#,
        ])
        try office(root.appending(path: "Work/Q3 close.xlsx"), members: [
            "xl/workbook.xml": #"<workbook><sheets><sheet name="Summary" sheetId="1"/></sheets></workbook>"#,
            "xl/sharedStrings.xml": "<sst><si><t>Total revenue for the quarter</t></si><si><t>Gross margin</t></si></sst>",
            "xl/worksheets/sheet1.xml": #"<worksheet><sheetData><row><c t="s"><v>0</v></c><c><v>4218400</v></c></row><row><c t="s"><v>1</v></c><c><v>0.714</v></c></row></sheetData></worksheet>"#,
        ])
        try write("Notes/Camping packing list.txt", "Packing list for the camping trip: tent, sleeping bags, headlamps, water filter, rain jackets, bug spray, and the camp stove with two fuel canisters.")
        try write("Notes/Banana bread.md", "# Banana bread\nMash three ripe bananas, whisk in melted butter, sugar and one egg, fold in flour and baking soda, and bake for an hour at 175 degrees.")
        // Everyday filler, so the index has a realistic spread of topics.
        let filler = [
            "Car service: the oil change and tire rotation are due at 60,000 miles; the brake pads were replaced in May.",
            "Garden plan: plant tomatoes and basil after the last frost, water the roses twice a week in summer.",
            "Book club picks for spring: a novel about a lighthouse keeper and a history of the printing press.",
            "Running log: five kilometres in 27 minutes on Sunday, knee felt fine, try intervals on Wednesday.",
            "Notes from the history lecture on the Roman aqueducts and how gravity carried water into the city.",
            "Wedding gift ideas for Sam and Priya: a pasta maker, a weekend cabin voucher, or framed map prints.",
            "Bike repair: the rear derailleur needs adjusting, and the chain should be cleaned and oiled.",
            "Meeting notes: the design review moved to Friday; ship the onboarding changes after QA signs off.",
            "Japanese practice: hiragana drills every morning, then twenty new vocabulary cards before bed.",
            "Moving checklist: book the movers, forward the mail, cancel the gym, and measure the new hallway.",
            "Birdwatching at the lake: two herons, a kingfisher and a flock of cedar waxwings near the pier.",
            "Tax prep reminders: collect the W-2, the charity receipts and the 1099 forms by the end of February.",
            "Podcast ideas: an episode on urban beekeeping and one on the history of public libraries.",
            "Vacation planning: compare train and flight times to Lisbon, and look for a flat near the river.",
            "Laptop setup: install the editor, sign in to the password manager, and turn on disk encryption.",
            "Sourdough starter: feed it equal parts flour and water each morning; it doubles in about six hours.",
            "Kids' school calendar: picture day is October 9 and the science fair is the second week of March.",
            "Workout plan: squats, push-ups and planks on Monday, Wednesday and Friday, stretching every day.",
            "Recipe for lentil soup: onions, carrots, celery, red lentils, cumin and a squeeze of lemon at the end.",
            "Photography notes: shoot the sunrise at f/8, keep ISO low, and bring the tripod for long exposures.",
            "Electric bill went up 12 percent this winter; consider a smart thermostat and sealing the windows.",
            "Guitar practice: learn the F barre chord, then the strumming pattern for the second verse.",
            "Neighbourhood association minutes: the street fair is in June and the park cleanup is in April.",
            "Plant care: the fiddle leaf fig wants bright indirect light and water only when the top soil is dry.",
            "Interview prep: questions about distributed systems, a story about a hard bug, and salary research.",
        ]
        for (index, text) in filler.enumerated() {
            try write("Notes/Misc/note \(index + 1).txt", text)
        }
    }

    static func pdf(_ url: URL, pages: [String]) throws {
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        var box = CGRect(x: 0, y: 0, width: 612, height: 792)
        let context = try #require(CGContext(url as CFURL, mediaBox: &box, nil))
        for page in pages {
            context.beginPDFPage(nil)
            let framesetter = CTFramesetterCreateWithAttributedString(NSAttributedString(string: page,
                attributes: [.font: NSFont.systemFont(ofSize: 12)]))
            let frame = CTFramesetterCreateFrame(framesetter, CFRange(location: 0, length: 0),
                                                 CGPath(rect: CGRect(x: 54, y: 54, width: 504, height: 684), transform: nil), nil)
            CTFrameDraw(frame, context)
            context.endPDFPage()
        }
        context.closePDF()
    }

    static func screenshot(_ url: URL, lines: [String]) throws {
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        let width = 1_400, height = 900
        let context = try #require(CGContext(data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0,
                                             space: CGColorSpaceCreateDeviceRGB(),
                                             bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue))
        context.setFillColor(CGColor(gray: 1, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: width, height: height))
        for (index, line) in lines.enumerated() {
            let text = CTLineCreateWithAttributedString(NSAttributedString(string: line,
                attributes: [.font: NSFont.systemFont(ofSize: 44), .foregroundColor: NSColor.black]))
            context.textPosition = CGPoint(x: 80, y: CGFloat(height - 140 - index * 90))
            CTLineDraw(text, context)
        }
        let image = try #require(context.makeImage())
        let destination = try #require(CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil))
        CGImageDestinationAddImage(destination, image, nil)
        #expect(CGImageDestinationFinalize(destination))
    }

    static func document(_ url: URL, _ text: String, type: NSAttributedString.DocumentType) throws {
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        let string = NSAttributedString(string: text)
        try string.data(from: NSRange(location: 0, length: string.length), documentAttributes: [.documentType: type])
            .write(to: url)
    }

    static func office(_ url: URL, members: [String: String]) throws {
        let staging = try FindTempFolder()
        for (path, xml) in members {
            try staging.write(path, xml)
        }
        try staging.write("[Content_Types].xml", "<Types/>")
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        let zip = Process()
        zip.executableURL = URL(filePath: "/usr/bin/zip")
        zip.currentDirectoryURL = staging.url
        zip.arguments = ["-qr", url.path, "."]
        try zip.run()
        zip.waitUntilExit()
    }
}

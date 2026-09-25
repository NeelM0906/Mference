import Foundation
import Testing
@testable import Mference
@testable import MferenceServerCore

private func entry(_ modelID: String,
                   family: ModelFamily = .gemma4,
                   path: String) -> ServerLibraryEntry {
    ServerLibraryEntry(modelID: modelID,
                       familyModelID: modelID,
                       basename: URL(fileURLWithPath: path).lastPathComponent,
                       directory: URL(fileURLWithPath: path, isDirectory: true),
                       family: family)
}

// MARK: - --list-models

@Suite("--list-models arguments")
struct ServerListModelsArgumentTests {
    @Test func listModelsIsOffByDefault() throws {
        #expect(try ServerArguments.parse(["--model", "m.gturbo"]).listModels == false)
        #expect(try ServerArguments.parse(["--library"]).listModels == false)
    }

    @Test func listModelsIsAcceptedWithLibrary() throws {
        #expect(try ServerArguments.parse(["--library", "--list-models"]).listModels)
        let withRoot = try ServerArguments.parse(["--library", "/a", "--list-models"])
        #expect(withRoot.listModels)
        #expect(withRoot.library?.roots == ["/a"])
    }

    /// It takes no value, so the flag after it must survive — the same hazard
    /// bare `--library` has.
    @Test func listModelsDoesNotEatTheNextFlag() throws {
        let arguments = try ServerArguments.parse([
            "--library", "--list-models", "--port", "9100",
        ])
        #expect(arguments.listModels)
        #expect(arguments.port == 9100)
    }

    /// Single-model mode already names its model on the command line, so there
    /// is nothing for a listing to add and the combination is a mistake.
    @Test func listModelsWithoutLibraryIsRejected() {
        #expect(throws: ServerArgumentError.invalid("--list-models requires --library")) {
            try ServerArguments.parse(["--model", "m.gturbo", "--list-models"])
        }
    }

    @Test func usageDocumentsListModels() {
        #expect(ServerArguments.usage.contains("--list-models"))
    }
}

// MARK: - Listing text

@Suite("Library listing")
struct ServerLibraryListingTests {
    @Test func emptyLibraryPrintsTheEmptySentinelAlone() {
        let text = ServerLibraryListing.text(for: ServerLibraryIndex(entries: []))
        #expect(text == ServerLibraryListing.emptyListing)
        #expect(!text.contains("\n"))
    }

    /// Header, one row per install, `/v1/models` order, and columns that line
    /// up: the launcher prints this verbatim.
    @Test func listingIsAHeaderAndOneRowPerInstallInModelOrder() {
        let index = ServerLibraryIndex(entries: [
            entry("qwen3.6-35b-a3b", family: .qwen36, path: "/roots/qwen36.gturbo"),
            entry("gemma-4-26b-a4b-it", family: .gemma4, path: "/roots/gemma4.gturbo"),
        ])
        let text = ServerLibraryListing.text(for: index) { _ in 14_291_921_884 }
        let lines = text.split(separator: "\n", omittingEmptySubsequences: false)
        #expect(lines.count == 3)
        #expect(lines[0].hasPrefix("MODEL"))
        #expect(lines[0].contains("FAMILY"))
        #expect(lines[0].contains("BYTES"))
        #expect(lines[0].hasSuffix("PATH"))
        // Sorted by identifier, exactly as /v1/models lists them.
        #expect(lines[1].hasPrefix("gemma-4-26b-a4b-it"))
        #expect(lines[2].hasPrefix("qwen3.6-35b-a3b   "))
        #expect(lines[1].contains("gemma4"))
        #expect(lines[2].contains("qwen36"))
        #expect(lines[1].hasSuffix("/roots/gemma4.gturbo"))
        #expect(lines[2].hasSuffix("/roots/qwen36.gturbo"))
        #expect(lines[1].contains("14291921884"))
        // Every row starts its PATH column at the same offset.
        let pathColumns = lines.map { line -> Int in
            line.distance(from: line.startIndex,
                          to: line.range(of: "PATH")?.lowerBound
                              ?? line.range(of: "/roots")!.lowerBound)
        }
        #expect(Set(pathColumns).count == 1)
    }

    /// A receipt that records no sizes must not be reported as a zero-byte
    /// install.
    @Test func unknownSizePrintsADashNotZero() throws {
        let index = ServerLibraryIndex(entries: [
            entry("gemma-4-26b-a4b-it", path: "/roots/gemma4.gturbo"),
        ])
        let text = ServerLibraryListing.text(for: index) { _ in nil }
        let row = try #require(text.split(separator: "\n").last)
        #expect(row.contains(" -  "))
        #expect(!row.contains(" 0 "))
    }

    /// Sizes come from the install receipt, so listing a 148 GB install reads
    /// one small JSON file and never walks the directory.
    @Test func installedBytesComeFromTheReceipt() throws {
        let root = try ServerLibraryFixture.makeRoot("listing")
        defer { try? FileManager.default.removeItem(at: root) }
        let directory = try ServerLibraryFixture.makeCompleteInstall(in: root, named: "gemma4")
        let listed = entry("gemma-4-26b-a4b-it", path: directory.path)
        // The fixture's receipt records no file sizes at all.
        #expect(ServerLibraryListing.receiptInstalledBytes(listed) == nil)

        let receiptURL = directory.appendingPathComponent("verified-install.json")
        var receipt = try #require(
            JSONSerialization.jsonObject(with: Data(contentsOf: receiptURL))
                as? [String: Any])
        receipt["files"] = [
            "model_weights.bin": ["size": 1_000, "sha256": String(repeating: "0", count: 64)],
            "packed_experts/layer_00.bin": ["size": 24,
                                            "sha256": String(repeating: "0", count: 64)],
        ]
        try JSONSerialization.data(withJSONObject: receipt, options: [.sortedKeys])
            .write(to: receiptURL)
        #expect(ServerLibraryListing.receiptInstalledBytes(listed) == 1_024)
    }

    @Test func aMissingReceiptIsReportedAsUnknownRatherThanCrashing() {
        let listed = entry("gemma-4-26b-a4b-it",
                           path: "/nonexistent/\(UUID().uuidString).gturbo")
        #expect(ServerLibraryListing.receiptInstalledBytes(listed) == nil)
    }
}

// MARK: - Default roots without the app

@Suite("Default library roots")
struct ServerLibraryDefaultRootsTests {
    private func defaults() -> UserDefaults {
        UserDefaults(suiteName: "mference-server-tests-\(UUID().uuidString)")!
    }

    @Test func homeModelLibraryIsScannedWithoutACheckout() {
        let roots = ServerLibraryDiscovery.defaultRoots(
            userDefaults: defaults(), environment: [:], executableURL: nil,
            currentDirectoryURL: URL(fileURLWithPath: "/not-a-checkout"),
            applicationSupportURL: nil, fileExists: { _ in false })
        #expect(roots.map(\.path) == [FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("llm-models").standardizedFileURL.path])
    }

    /// The checkout is recognized by `Package.swift` beside the runtime and
    /// server source trees. `Sources/MferenceApp` is absent here on purpose:
    /// the Mac app is gone, and `scratch/` must still be scanned without it.
    @Test func scratchIsFoundWithNoAppSourcesOnDisk() {
        let present: Set<String> = [
            "/checkout/Package.swift",
            "/checkout/Sources/Mference",
            "/checkout/Sources/MferenceServer",
        ]
        let roots = ServerLibraryDiscovery.defaultRoots(
            userDefaults: defaults(),
            environment: [:],
            executableURL: URL(fileURLWithPath: "/checkout/.build/release/MferenceServer"),
            currentDirectoryURL: URL(fileURLWithPath: "/elsewhere", isDirectory: true),
            applicationSupportURL: nil,
            homeDirectoryURL: URL(fileURLWithPath: "/Users/library user"),
            fileExists: { present.contains($0) })
        #expect(roots.map(\.path) == ["/Users/library user/llm-models", "/checkout/scratch"])
    }

    /// The old marker — `Package.swift` plus the app's Mac sources — is not
    /// enough on its own, so nothing can quietly start depending on the app
    /// tree being present again.
    @Test func appSourcesAloneDoNotIdentifyACheckout() {
        let present: Set<String> = [
            "/checkout/Package.swift",
            "/checkout/Sources/MferenceApp/Mac",
        ]
        let roots = ServerLibraryDiscovery.defaultRoots(
            userDefaults: defaults(),
            environment: [:],
            executableURL: URL(fileURLWithPath: "/checkout/.build/release/MferenceServer"),
            currentDirectoryURL: URL(fileURLWithPath: "/checkout", isDirectory: true),
            applicationSupportURL: nil,
            homeDirectoryURL: URL(fileURLWithPath: "/Users/library user"),
            fileExists: { present.contains($0) })
        #expect(roots.map(\.path) == ["/Users/library user/llm-models"])
    }

    @Test func configuredRootComesFirstAndApplicationSupportLast() {
        let present: Set<String> = [
            "/checkout/Package.swift",
            "/checkout/Sources/Mference",
            "/checkout/Sources/MferenceServer",
        ]
        let roots = ServerLibraryDiscovery.defaultRoots(
            userDefaults: defaults(),
            environment: [ServerLibraryDiscovery.libraryRootEnvironmentKey: "/models"],
            executableURL: URL(fileURLWithPath: "/checkout/.build/release/MferenceServer"),
            currentDirectoryURL: URL(fileURLWithPath: "/checkout", isDirectory: true),
            applicationSupportURL: URL(fileURLWithPath: "/Users/x/Library/Application Support",
                                       isDirectory: true),
            homeDirectoryURL: URL(fileURLWithPath: "/Users/library user"),
            fileExists: { present.contains($0) })
        #expect(roots.map(\.path) == [
            "/models",
            "/Users/library user/llm-models",
            "/checkout/scratch",
            "/Users/x/Library/Application Support/Mference",
        ])
    }

    @Test func configuredHomeLibraryIsNotDuplicated() {
        let roots = ServerLibraryDiscovery.defaultRoots(userDefaults: defaults(),
            environment: [ServerLibraryDiscovery.libraryRootEnvironmentKey: "/Users/library user/llm-models"],
            executableURL: nil, applicationSupportURL: nil,
            homeDirectoryURL: URL(fileURLWithPath: "/Users/library user"),
            fileExists: { _ in false })
        #expect(roots.map(\.path) == ["/Users/library user/llm-models"])
    }
}

// MARK: - Starting with nothing installed

@Suite("Empty library", .serialized)
struct ServerEmptyLibraryTests {
    /// Library mode is the UI's backend, and the UI has to come up before the
    /// first install exists. The server starts, `/v1/models` is empty, and a
    /// request naming anything gets the ordinary 404 — the loader is never
    /// reached.
    @Test func emptyLibraryStartsAndServesAnEmptyModelList() async throws {
        let library = ServerModelLibrary(index: ServerLibraryIndex(entries: [])) { _ in
            Issue.record("an empty library must never load anything")
            throw ServerRequestError.unknownModel
        }
        let server = MferenceHTTPServer(library: library, queueLimit: 1)
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)

        let listed = try await URLSession.shared.data(
            from: URL(string: "http://127.0.0.1:\(port)/v1/models")!).0
        let object = try #require(
            JSONSerialization.jsonObject(with: listed) as? [String: Any])
        #expect(object["object"] as? String == "list")
        #expect((object["data"] as? [[String: Any]])?.isEmpty == true)

        let health = try await URLSession.shared.data(
            from: URL(string: "http://127.0.0.1:\(port)/health")!).0
        #expect(String(decoding: health, as: UTF8.self).contains("\"ok\""))

        var request = URLRequest(
            url: URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "content-type")
        request.httpBody = Data(
            #"{"model":"gemma-4-26b-a4b-it","messages":[{"role":"user","content":"hi"}]}"#
                .utf8)
        let (body, response) = try await URLSession.shared.data(for: request)
        #expect((response as? HTTPURLResponse)?.statusCode == 404)
        #expect(String(decoding: body, as: UTF8.self).contains("model_not_found"))

        try await server.shutdown()
    }
}

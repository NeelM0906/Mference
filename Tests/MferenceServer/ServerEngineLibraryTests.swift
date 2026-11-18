import Foundation
import Synchronization
import Testing
@testable import Mference
@testable import MferenceEngine
@testable import MferenceServerCore

// Library mode on the engine: the catalog it serves, the swap log, the bridge
// between server and engine types, and the HTTP surface with an engine model
// that is not the server's own session.

private final class Events: Sendable {
    private let lines = Mutex<[String]>([])
    func append(_ line: String) { lines.withLock { $0.append(line) } }
    var all: [String] { lines.withLock { $0 } }
}

/// Appends `release <id>` on deinit: the observable point at which a bridged
/// session's memory is gone.
private final class Sentinel {
    let events: Events
    let id: String
    init(_ events: Events, _ id: String) { self.events = events; self.id = id }
    deinit { events.append("release \(id)") }
}

private actor BridgedModel: ServerLoadedModel {
    nonisolated let chatDialect: ChatDialect = .chatml
    private let sentinel: Sentinel
    private let id: String

    init(_ events: Events, _ id: String) {
        self.sentinel = Sentinel(events, id)
        self.id = id
    }

    func prepare(_ request: ValidatedChatRequest) async throws -> PreparedGeneration {
        if request.messages.first?.content == "too long" {
            throw ServerRequestError.invalid(message: "prompt exceeds the configured context",
                                             param: "messages", code: "context_length_exceeded")
        }
        return PreparedGeneration(request: request, promptIDs: [1, 2, 3])
    }

    func generate(_ prepared: PreparedGeneration,
                  onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void) async throws -> ServerCompletion {
        onEvent(.content(id))
        var completion = ServerCompletion(
            content: id, toolCalls: [], finishReason: "stop",
            usage: OpenAIUsage(promptTokens: 3, completionTokens: 1, totalTokens: 4, cachedTokens: 2,
                               completionTokensDetails: .init(reasoningTokens: 0, visibleTokens: 1)))
        completion.prefillSeconds = 0.25
        completion.decodeSeconds = 0.5
        return completion
    }
}

private func entry(_ id: String, family: ModelFamily, basename: String,
                   familyID: String? = nil) -> ServerLibraryEntry {
    ServerLibraryEntry(modelID: id, familyModelID: familyID ?? id, basename: basename,
                       directory: URL(fileURLWithPath: "/models/\(basename).gturbo", isDirectory: true),
                       family: family)
}

private func bridgedLibrary(_ ids: [String], events: Events) -> ServerModelLibrary {
    ServerModelLibrary(index: ServerLibraryIndex(entries: ids.map {
        entry($0, family: .qwen36, basename: $0)
    })) { directory in
        let id = directory.deletingPathExtension().lastPathComponent
        events.append("load \(id)")
        return BridgedModel(events, id)
    }
}

// MARK: - Catalog

/// A library on disk, for `EngineCatalogDiscovery`: install directories by
/// family, and GGUF files by directory. Nothing is read from disk.
private func fakeFileSystem(installs: [String: ModelFamily],
                            ggufs: [String: [String]] = [:]) -> EngineCatalogDiscovery.FileSystem {
    EngineCatalogDiscovery.FileSystem(
        childDirectories: { root in
            installs.keys.filter { ($0 as NSString).deletingLastPathComponent == root.path }
                .sorted().map { URL(fileURLWithPath: $0, isDirectory: true) }
        },
        ggufFiles: { directory in
            (ggufs[directory.path] ?? []).map { directory.appendingPathComponent($0) }
        },
        probe: { url in installs[url.standardizedFileURL.path].map { .complete(family: $0) } ?? .notAModelDirectory },
        checkpointID: { _ in nil },
        readGGUF: { url in
            EngineGGUFMetadata(architecture: "qwen35", contextLength: 262_144, tensorCount: 10,
                               nextNTensorCount: url.lastPathComponent.contains("-mtp") ? 4 : 0)
        },
        size: { _ in 42 })
}

private let libraryInstalls: [String: ModelFamily] = [
    "/models/qwen36.gturbo": .qwen36,
    "/models/minicpm5.gturbo": .minicpm5,
    "/models/maple.gturbo": .maple,
    "/models/qwen38flashnextudiq3xxs.gturbo": .qwen38flashnextudiq3xxs,
    "/models/gemma4.gturbo": .gemma4,
]
private let libraryGGUFs = ["/models/gguf": ["Qwen3.5-4B-UD-Q4_K_XL.gguf",
                                             "Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf",
                                             "Some-Other-Model.Q4_K_M.gguf"]]

private func servingLibrary(_ arguments: [String],
                            installs: [String: ModelFamily] = libraryInstalls,
                            explicitModel: URL? = nil,
                            backends: [any EngineBackend] = [FakeEngineBackend()]) throws -> ServerModelLibrary {
    ServerModelLibrary.serving(roots: [URL(fileURLWithPath: "/models", isDirectory: true)],
                               explicitModel: explicitModel,
                               arguments: try ServerArguments.parse(arguments),
                               ggufDirectories: [URL(fileURLWithPath: "/models/gguf", isDirectory: true)],
                               fileSystem: fakeFileSystem(installs: installs, ggufs: libraryGGUFs),
                               backends: backends)
}

@Suite("Engine catalog for library mode")
struct ServerEngineCatalogTests {
    @Test func productScopeServesTheNativeProductModelsWithANativeBackend() throws {
        let library = try servingLibrary(["--library"])
        let catalog = library.snapshot.catalog
        #expect(catalog.models.map(\.id) == ["maple-preview-2bit-mlx", "minicpm5-2b-int4g64", "qwen3.6-35b-a3b"])
        #expect(catalog.models.allSatisfy { $0.engine == .mference && $0.product != nil })
        #expect(library.readyDetail == "catalog=product models=3")
        // The GGUF product models wait for a llama.cpp backend, and say so.
        #expect(catalog.unavailableReason(for: "qwen3.5-4b") == "no llama.cpp backend in this build")
        #expect(catalog.unavailableReason(for: "qwen3.8-27b") == "no llama.cpp backend in this build")
        // So does a non-product GGUF: --catalog all would not help without it.
        #expect(catalog.unavailableReason(for: "some-other-model.q4_k_m") == "no llama.cpp backend in this build")
        // Installs the product does not offer name the server's switch.
        let flashNext = try #require(catalog.unavailableReason(for: "unsloth-qwen3.8-flash-next-ud-iq3_xxs"))
        #expect(flashNext.hasSuffix("--catalog all lists it"))
        #expect(catalog.unavailableReason(for: "gemma-4-26b-a4b-it")?.hasSuffix("--catalog all lists it") == true)
        #expect(!catalog.skipped.contains { $0.reason.contains("developer catalog scope") })
        #expect(library.snapshot.health.model == nil)
    }

    @Test func aLlamaBackendBringsTheGGUFProductModelsIn() throws {
        let library = try servingLibrary(["--library"],
                                         backends: [FakeEngineBackend(), FakeEngineBackend(engine: .llamaCpp)])
        let models = library.snapshot.catalog.models
        #expect(models.map(\.id) == ["maple-preview-2bit-mlx", "minicpm5-2b-int4g64", "qwen3.5-4b",
                                     "qwen3.6-35b-a3b"])
        #expect(library.snapshot.modelList.data.map(\.engine)
                == ["mference", "mference", "llama.cpp", "mference"])
        // GGUF context defaults to 32768, native to --max-context.
        #expect(library.snapshot.modelList.data.map(\.maxModelLen) == [16_384, 16_384, 32_768, 16_384])
        // Deep waits for MTP draft decoding in the llama.cpp backend.
        #expect(library.snapshot.unavailableReason(for: "qwen3.8-27b") == ServerEngineCatalog.deepNeedsMTP)
        let withMTP = ServerEngineCatalog.servable(
            EngineCatalogDiscovery.discover(
                roots: [URL(fileURLWithPath: "/models", isDirectory: true)],
                ggufDirectories: [URL(fileURLWithPath: "/models/gguf", isDirectory: true)],
                fileSystem: fakeFileSystem(installs: libraryInstalls, ggufs: libraryGGUFs)),
            engines: [.mference, .llamaCpp], deepWithMTP: true)
        #expect(withMTP.descriptor(for: "qwen3.8-27b")?.engine == .llamaCpp)
    }

    /// The server's flags reach the engine host, and the default backends are
    /// the engine's own: `NativeBackend` and `LlamaBackend`, so the GGUF
    /// product models are served too (nothing is loaded here).
    @Test func servingPassesTheFlagsAndDefaultsToTheEngineBackends() throws {
        let arguments = try ServerArguments.parse(["--library", "--max-context", "8192",
                                                    "--prompt-cache-mode", "off"])
        let library = ServerModelLibrary.serving(
            roots: [URL(fileURLWithPath: "/models", isDirectory: true)],
            arguments: arguments,
            ggufDirectories: [URL(fileURLWithPath: "/models/gguf", isDirectory: true)],
            fileSystem: fakeFileSystem(installs: libraryInstalls, ggufs: libraryGGUFs))
        #expect(library.host.loadOptions == arguments.engineLoadOptions)
        let list = library.snapshot.modelList.data
        #expect(list.map(\.id) == ["maple-preview-2bit-mlx", "minicpm5-2b-int4g64", "qwen3.5-4b",
                                   "qwen3.6-35b-a3b"])
        // --max-context sets native models; GGUF models keep 32768.
        #expect(list.map(\.maxModelLen) == [8_192, 8_192, 32_768, 8_192])
        #expect(list.map(\.engine) == ["mference", "mference", "llama.cpp", "mference"])
        #expect(library.snapshot.unavailableReason(for: "qwen3.8-27b") == ServerEngineCatalog.deepNeedsMTP)
        // A non-product GGUF is declined with the switch that lists it.
        #expect(library.snapshot.unavailableReason(for: "some-other-model.q4_k_m")?
            .hasSuffix("--catalog all lists it") == true)
    }

    @Test func allScopeListsEveryRunnableInstall() throws {
        let library = try servingLibrary(["--library", "--catalog", "all"])
        #expect(library.snapshot.catalog.models.count == 5)
        #expect(library.snapshot.descriptor(for: "gemma-4-26b-a4b-it") != nil)
        #expect(library.readyDetail == "catalog=all models=5")
    }

    @Test func explicitModelIsTheInstallServedForItsProductModel() throws {
        var installs = libraryInstalls
        installs["/models/qwen36-ourquant.gturbo"] = .qwen36
        let preferred = URL(fileURLWithPath: "/models/qwen36-ourquant.gturbo", isDirectory: true)
        let library = try servingLibrary(["--library"], installs: installs, explicitModel: preferred)
        #expect(library.snapshot.descriptor(for: "qwen3.6-35b-a3b")?.location.standardizedFileURL.path
                == preferred.path)
        let plain = try servingLibrary(["--library"], installs: installs)
        #expect(plain.snapshot.descriptor(for: "qwen3.6-35b-a3b")?.location.lastPathComponent
                == "qwen36.gturbo")
    }

    @Test func runtimeFlagsBecomeEngineLoadOptions() throws {
        let options = try ServerArguments.parse([
            "--library", "--max-context", "8192", "--prompt-cache-mode", "single-prefix",
            "--prompt-cache-bytes", "1024", "--verify", "full-sha256", "--shadow-budget", "2",
            "--prefill-chunk", "512", "--kv-reserve",
        ]).engineLoadOptions
        #expect(options == EngineLoadOptions(nativeMaxContext: 8_192,
                                             promptCacheMode: .singlePrefix,
                                             promptCacheBytes: .bytes(1_024),
                                             integrityPolicy: try #require(ModelIntegrityPolicy(verifyFlag: "full-sha256")),
                                             shadowPrefetchBudget: 2,
                                             prefillChunkTokens: 512,
                                             reserveFullKV: true))
        let defaults = try ServerArguments.parse(["--library"]).engineLoadOptions
        #expect(defaults == EngineLoadOptions())
        #expect(try ServerArguments.parse(["--library", "--max-context", "max"]).engineLoadOptions
                .nativeMaxContext == nil)
    }

    @Test func listingShowsTheEngineColumn() throws {
        let library = try servingLibrary(["--library"])
        let lines = library.listing.split(separator: "\n").map(String.init)
        #expect(lines.first?.hasPrefix("MODEL") == true)
        #expect(lines.first?.contains("ENGINE") == true)
        #expect(lines.count == 4)
        #expect(lines.dropFirst().allSatisfy { $0.contains("mference") && $0.contains("42") })
        #expect(ServerLibraryListing.text(for: .empty) == ServerLibraryListing.emptyListing)
        let onlyOthers = try servingLibrary(["--library"], installs: ["/models/gemma4.gturbo": .gemma4])
        // The GGUF files (two product, one other) are not counted: without a
        // llama.cpp backend --catalog all would not list them either.
        #expect(onlyOthers.listing == "no product models installed (1 other install; --catalog all lists them)")
    }

    @Test func catalogArgument() throws {
        #expect(try ServerArguments.parse(["--library"]).catalogScope == .product)
        #expect(try ServerArguments.parse(["--library", "--catalog", "all"]).catalogScope == .all)
        #expect(throws: ServerArgumentError.self) {
            _ = try ServerArguments.parse(["--library", "--catalog", "everything"])
        }
        #expect(throws: ServerArgumentError.self) {
            _ = try ServerArguments.parse(["--model", "/m", "--catalog", "all"])
        }
        #expect(ServerArguments.usage.contains("--catalog <product|all>"))
    }

    /// The engine's wording for `--catalog all`; if it changes, the server's
    /// rewrite (and its 400 messages) must change with it.
    @Test func engineScopeWordingIsTheOneTheServerRewrites() {
        let catalog = EngineCatalogDiscovery.discover(
            roots: [URL(fileURLWithPath: "/models", isDirectory: true)], ggufDirectories: [],
            fileSystem: fakeFileSystem(installs: ["/models/gemma4.gturbo": .gemma4]))
        #expect(catalog.skipped.first?.reason.hasSuffix(ServerEngineCatalog.engineScopeHint) == true)
    }
}

// MARK: - Swap log

@Suite("Engine host swap log")
struct ServerHostLogTests {
    @Test func messagesKeepTheServerFormatAndAddTheReleaseLine() {
        #expect(ServerLog.hostEventMessage(.loadStarted(modelID: "a", replacing: nil)) == "swap started to=a")
        #expect(ServerLog.hostEventMessage(.unloadStarted(modelID: "a", reason: .swap(to: "b")))
                == "swap started from=a to=b")
        #expect(ServerLog.hostEventMessage(.unloadFinished(modelID: "a", reason: .swap(to: "b")))
                == "model released model=a reason=swap")
        #expect(ServerLog.hostEventMessage(.loadStarted(modelID: "b", replacing: "a")) == nil)
        #expect(ServerLog.hostEventMessage(.loadFinished(modelID: "b", seconds: 12.34))
                == "swap finished model=b in 12.3s")
        #expect(ServerLog.hostEventMessage(.loadFailed(modelID: "b", message: "boom"))
                == "swap failed model=b error=boom")
        #expect(ServerLog.hostEventMessage(.unloadStarted(modelID: "b", reason: .idle)) == nil)
        #expect(ServerLog.hostEventMessage(.unloadFinished(modelID: "b", reason: .shutdown))
                == "model released model=b reason=shutdown")
    }

    /// The order memory validation (`Scripts/engine-validate.sh`) checks: the
    /// old session is gone and logged as released before the next load starts.
    @Test func aSwapLogsReleaseBetweenStartAndFinish() async throws {
        let events = Events()
        let host = EngineHost(
            catalog: ServerEngineCatalog.catalog(
                for: ServerLibraryIndex(entries: [entry("a", family: .qwen36, basename: "a"),
                                                  entry("b", family: .qwen36, basename: "b")])),
            backends: [ServerLoaderBackend { directory in
                let id = directory.deletingPathExtension().lastPathComponent
                events.append("load \(id)")
                return BridgedModel(events, id)
            }],
            observer: { event in ServerLog.hostEventMessage(event).map(events.append) })
        try await host.load("a")
        try await host.load("b")
        // Durations depend on the machine; compare the lines without them.
        let lines = events.all.map {
            $0.hasPrefix("swap finished") ? $0.components(separatedBy: " in ")[0] : $0
        }
        #expect(lines == [
            "swap started to=a", "load a", "swap finished model=a",
            "swap started from=a to=b", "release a", "model released model=a reason=swap",
            "load b", "swap finished model=b",
        ])
        await host.shutdown()
        #expect(events.all.suffix(2) == ["release b", "model released model=b reason=shutdown"])
    }
}

// MARK: - Bridge

@Suite("Server and engine type bridge")
struct ServerEngineBridgeTests {
    @Test func bridgedSessionKeepsItsTraitsAndRunsUnchanged() async throws {
        let events = Events()
        let backend = ServerLoaderBackend { _ in BridgedModel(events, "a") }
        let descriptor = FakeEngineBackend.descriptor(id: "a")
        let model = try await backend.load(descriptor, options: EngineLoadOptions(nativeMaxContext: 2_048))
        #expect(model.traits.chatDialect == .chatml)
        #expect(model.traits.maxContext == 2_048)

        let request = EngineRequest(messages: [.init(role: .user, content: "hi")],
                                    maximumCompletionTokens: 8)
        let prepared = try await model.prepare(request)
        #expect(prepared.promptTokens == [1, 2, 3])
        let streamed = Events()
        let completion = try await model.generate(prepared) { event in
            if case .content(let text) = event { streamed.append(text) }
        }
        #expect(streamed.all == ["a"])
        #expect(completion.content == "a")
        #expect(completion.usage.cachedPromptTokens == 2)
        #expect(completion.usage.visibleTokens == 1)
        #expect(completion.timings.prefillSeconds == 0.25)

        // The server path sees the very same session object.
        // (Not bound to a local: that would keep the session alive past unload.)
        #expect(try serverLoadedModel(for: model) is BridgedModel)

        // Unload drops the session at once.
        await model.unload()
        #expect(events.all == ["release a"])
        #expect(throws: EngineError.modelUnloaded("a")) { _ = try serverLoadedModel(for: model) }
    }

    @Test func bridgedInvalidRequestBecomesAnEngineInvalidRequest() async throws {
        let model = try await ServerLoaderBackend { _ in BridgedModel(Events(), "a") }
            .load(FakeEngineBackend.descriptor(id: "a"), options: EngineLoadOptions())
        await #expect(throws: EngineError.invalidRequest(message: "prompt exceeds the configured context",
                                                        param: "messages",
                                                        code: "context_length_exceeded")) {
            _ = try await model.prepare(EngineRequest(messages: [.init(role: .user, content: "too long")],
                                                      maximumCompletionTokens: 8))
        }
    }

    @Test func engineModelAdapterMapsEventsUsageAndErrors() async throws {
        let call = ParsedToolCall(id: "call_0", name: "get_weather",
                                  arguments: .object(["city": .string("Lisbon")]),
                                  argumentsJSON: #"{"city":"Lisbon"}"#)
        let backend = FakeEngineBackend(defaultScript: FakeEngineScript(
            events: [.reasoning("think"), .content("Hi"), .toolCall(call)]))
        let model = try await backend.load(FakeEngineBackend.descriptor(id: "m", contextLength: 64),
                                           options: EngineLoadOptions(nativeMaxContext: 64))
        let adapter = try serverLoadedModel(for: model)
        #expect(adapter.chatDialect == .chatml)

        let request = ValidatedChatRequest(messages: [.init(role: .user, content: "hello there")],
                                           tools: [], stream: true, includeUsage: true,
                                           generationConfig: .defaults, maximumCompletionTokens: 16)
        let prepared = try await adapter.prepare(request)
        #expect(prepared.promptIDs.count == 3)
        let streamed = Events()
        let completion = try await adapter.generate(prepared) { event in
            switch event {
            case .reasoning(let text): streamed.append("r:\(text)")
            case .content(let text): streamed.append("c:\(text)")
            case .toolCall(let call): streamed.append("t:\(call.name)")
            }
        }
        #expect(streamed.all == ["r:think", "c:Hi", "t:get_weather"])
        #expect(completion.finishReason == "tool_calls")
        #expect(completion.reasoningContent == "think")
        #expect(completion.toolCalls == [call])
        #expect(completion.usage.promptTokens == 3)
        #expect(completion.usage.completionTokens == 3)
        // No reasoning count from the backend: no invented details object.
        #expect(completion.usage.completionTokensDetails == nil)

        let long = ValidatedChatRequest(
            messages: [.init(role: .user, content: Array(repeating: "w", count: 80).joined(separator: " "))],
            tools: [], stream: false, includeUsage: false,
            generationConfig: .defaults, maximumCompletionTokens: 16)
        await #expect(throws: ServerRequestError.invalid(message: "prompt exceeds the configured context",
                                                        param: "messages",
                                                        code: "context_length_exceeded")) {
            _ = try await adapter.prepare(long)
        }
    }

    @Test func engineErrorsMapToTheServersErrors() {
        #expect(serverError(EngineError.unknownModel("x")) as? ServerRequestError == .unknownModel)
        #expect(serverError(EngineError.queueFull) as? ServerRequestError == .queueFull)
        #expect(serverError(EngineError.modelNotRunnable(id: "x", reason: "why")) as? ServerRequestError
                == .invalid(message: "why", param: "model", code: "model_not_runnable"))
        #expect(serverError(EngineError.shuttingDown) is CancellationError)
        struct Other: Error {}
        #expect(serverError(Other()) is Other)
    }
}

// MARK: - Library and HTTP

@Suite("Library mode on the engine host", .serialized)
struct ServerEngineLibraryTests {
    @Test func shutdownReleasesTheResidentModel() async throws {
        let events = Events()
        let library = bridgedLibrary(["a"], events: events)
        _ = try await library.resolve(modelID: "a")
        #expect(library.loadedModelID == "a")
        await library.shutdown()
        #expect(events.all == ["load a", "release a"])
        #expect(library.snapshot.health.model == nil)
        await #expect(throws: CancellationError.self) { _ = try await library.resolve(modelID: "a") }
    }

    @Test func preloadByDirectory() async throws {
        let events = Events()
        let library = bridgedLibrary(["a", "b"], events: events)
        try await library.preload(directory: URL(fileURLWithPath: "/models/b.gturbo", isDirectory: true))
        #expect(events.all == ["load b"])
        await #expect(throws: ServerArgumentError.self) {
            try await library.preload(directory: URL(fileURLWithPath: "/elsewhere/c.gturbo"))
        }
    }

    /// `--library --model` naming an install the product catalog declines
    /// fails with the catalog's reason, which names `--catalog all`.
    @Test func preloadOfADeclinedInstallSaysWhy() async throws {
        let gemma = URL(fileURLWithPath: "/models/gemma4.gturbo", isDirectory: true)
        let library = try servingLibrary(["--library"], explicitModel: gemma)
        do {
            try await library.preload(directory: gemma)
            Issue.record("preload of a declined install succeeded")
        } catch let error as ServerArgumentError {
            #expect(String(describing: error).contains("--catalog all"))
        }
    }

    /// Production library mode end to end over HTTP, with a scripted engine
    /// backend in the native slot: a product model is served through the
    /// engine adapter, a declined install is a 400 that says why, and the
    /// listing names only product models.
    @Test func productCatalogOverDiscoveryServesAndDeclines() async throws {
        let journal = FakeEngineJournal()
        let library = try servingLibrary(
            ["--library"],
            backends: [FakeEngineBackend(journal: journal,
                                         defaultScript: FakeEngineScript(events: [.content("native")]))])
        let server = MferenceHTTPServer(library: library, queueLimit: 1)
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)
        func post(_ model: String) async throws -> (Int, String) {
            var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!)
            request.httpMethod = "POST"
            request.setValue("application/json", forHTTPHeaderField: "content-type")
            request.httpBody = Data(#"{"model":"\#(model)","messages":[{"role":"user","content":"hi"}]}"#.utf8)
            let (data, response) = try await URLSession.shared.data(for: request)
            return ((response as? HTTPURLResponse)?.statusCode ?? 0, String(decoding: data, as: UTF8.self))
        }
        let listing = String(decoding: try await URLSession.shared.data(
            from: URL(string: "http://127.0.0.1:\(port)/v1/models")!).0, as: UTF8.self)
        #expect(listing.contains(#""id":"qwen3.6-35b-a3b""#))
        #expect(!listing.contains("gemma") && !listing.contains("qwen3.5-4b"))

        for declined in ["gemma-4-26b-a4b-it", "qwen3.5-4b"] {
            let (status, body) = try await post(declined)
            #expect(status == 400, "\(declined): \(body)")
            #expect(body.contains("model_not_runnable"))
        }
        #expect(journal.entries.isEmpty)

        let (servedStatus, servedBody) = try await post("qwen3.6-35b-a3b")
        #expect(servedStatus == 200, "\(servedBody)")
        #expect(servedBody.contains(#""content":"native""#))
        try await server.shutdown()
        #expect(journal.maxLiveModels == 1)
        #expect(journal.entries.contains(.unloaded("qwen3.6-35b-a3b")))
    }

    @Test func modelsCarryTheEngineAndAnEngineModelServesChat() async throws {
        let catalog = EngineCatalog(models: [
            FakeEngineBackend.descriptor(id: "native", contextLength: 4_096, product: .qwen36),
            FakeEngineBackend.descriptor(id: "gguf", engine: .llamaCpp, contextLength: 65_536,
                                         product: .quick),
        ], skipped: [EngineCatalogSkip(location: URL(fileURLWithPath: "/m/flash"),
                                       reason: "not offered", modelID: "flash")])
        let library = ServerModelLibrary(
            catalog: catalog,
            backends: [FakeEngineBackend(),
                       FakeEngineBackend(engine: .llamaCpp,
                                         defaultScript: FakeEngineScript(events: [.content("q")]))],
            loadOptions: EngineLoadOptions(nativeMaxContext: 2_048))
        let server = MferenceHTTPServer(library: library, queueLimit: 1)
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)

        let listing = try await URLSession.shared.data(
            from: URL(string: "http://127.0.0.1:\(port)/v1/models")!).0
        let object = try #require(JSONSerialization.jsonObject(with: listing) as? [String: Any])
        let models = try #require(object["data"] as? [[String: Any]])
        #expect(models.map { $0["id"] as? String } == ["gguf", "native"])
        #expect(models.map { $0["engine"] as? String } == ["llama.cpp", "mference"])
        // GGUF context defaults to 32768; native follows --max-context.
        #expect(models.map { $0["max_model_len"] as? Int } == [32_768, 2_048])

        for stream in [false, true] {
            var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!)
            request.httpMethod = "POST"
            request.setValue("application/json", forHTTPHeaderField: "content-type")
            request.httpBody = Data(#"{"model":"gguf","stream":\#(stream),"messages":[{"role":"user","content":"hi"}]}"#.utf8)
            let (data, response) = try await URLSession.shared.data(for: request)
            let text = String(decoding: data, as: UTF8.self)
            #expect((response as? HTTPURLResponse)?.statusCode == 200, "\(text)")
            #expect(text.contains(#""content":"q""#))
            #expect(text.contains(#""model":"gguf""#))
        }
        #expect(library.snapshot.health.model == "gguf")

        var declined = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!)
        declined.httpMethod = "POST"
        declined.setValue("application/json", forHTTPHeaderField: "content-type")
        declined.httpBody = Data(#"{"model":"flash","messages":[{"role":"user","content":"hi"}]}"#.utf8)
        let (data, response) = try await URLSession.shared.data(for: declined)
        #expect((response as? HTTPURLResponse)?.statusCode == 400)
        #expect(String(decoding: data, as: UTF8.self).contains("model_not_runnable"))

        try await server.shutdown()
        #expect(library.snapshot.health.model == nil)
    }
}

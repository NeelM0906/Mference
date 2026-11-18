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

@Suite("Engine catalog for library mode")
struct ServerEngineCatalogTests {
    private static let index = ServerLibraryIndex(
        entries: [
            entry("qwen3.6-35b-a3b@qwen36", family: .qwen36, basename: "qwen36",
                  familyID: "qwen3.6-35b-a3b"),
            entry("qwen3.6-35b-a3b@a-quant", family: .qwen36, basename: "a-quant",
                  familyID: "qwen3.6-35b-a3b"),
            entry("minicpm5-2b-int4g64", family: .minicpm5, basename: "minicpm5"),
            entry("maple-preview-2bit-mlx", family: .maple, basename: "maple"),
            entry("unsloth-qwen3.8-flash-next-ud-iq3_xxs", family: .qwen38flashnextudiq3xxs,
                  basename: "qwen38flashnextudiq3xxs"),
        ],
        skipped: [ServerLibrarySkip(directory: URL(fileURLWithPath: "/models/partial.gturbo"),
                                    reason: "incomplete install: x")])

    @Test func productScopeListsOnlyTheProductModelsUnderTheirSlotIdentifiers() throws {
        let catalog = ServerEngineCatalog.catalog(for: Self.index, scope: .product,
                                                  installedBytes: { _ in 42 })
        #expect(catalog.models.map(\.id) == ["maple-preview-2bit-mlx", "minicpm5-2b-int4g64",
                                             "qwen3.6-35b-a3b"])
        #expect(catalog.models.allSatisfy { $0.engine == .mference && $0.sizeBytes == 42 })
        #expect(catalog.models.map(\.product) == [.maple, .minicpm5, .qwen36])
        // The install named after the slot wins over another of the same checkpoint.
        let qwen = try #require(catalog.descriptor(for: "qwen3.6-35b-a3b"))
        #expect(qwen.location.lastPathComponent == "qwen36.gturbo")
        #expect(qwen.displayName == EngineProductSlot.qwen36.displayName)
    }

    @Test func productScopeDeclinesEverythingElseWithAReason() throws {
        let catalog = ServerEngineCatalog.catalog(for: Self.index, scope: .product)
        let flashNext = try #require(catalog.unavailableReason(for: "unsloth-qwen3.8-flash-next-ud-iq3_xxs"))
        #expect(flashNext.contains("--catalog all"))
        #expect(catalog.unavailableReason(for: "qwen3.6-35b-a3b@a-quant")?
            .contains("another install") == true)
        // The served install's discovery identifier explains the stable one.
        #expect(catalog.unavailableReason(for: "qwen3.6-35b-a3b@qwen36")?
            .contains("served as qwen3.6-35b-a3b") == true)
        #expect(catalog.skipped.contains { $0.reason == "incomplete install: x" })
        #expect(throws: EngineError.self) { _ = try catalog.resolve("unsloth-qwen3.8-flash-next-ud-iq3_xxs") }
        #expect(catalog.descriptor(for: "qwen3.6-35b-a3b@qwen36") == nil)
    }

    @Test func allScopeListsEveryRunnableInstallUnderItsDiscoveryIdentifier() {
        let catalog = ServerEngineCatalog.catalog(for: Self.index, scope: .all)
        #expect(catalog.models.map(\.id) == Self.index.entries.map(\.modelID).sorted())
        #expect(catalog.descriptor(for: "minicpm5-2b-int4g64")?.product == .minicpm5)
        #expect(catalog.descriptor(for: "unsloth-qwen3.8-flash-next-ud-iq3_xxs")?.product == nil)
    }

    @Test func explicitModelDirectoryWinsItsProductSlot() throws {
        let catalog = ServerEngineCatalog.catalog(
            for: Self.index, scope: .product,
            preferredDirectory: URL(fileURLWithPath: "/models/a-quant.gturbo"))
        #expect(try #require(catalog.descriptor(for: "qwen3.6-35b-a3b")).location.lastPathComponent
                == "a-quant.gturbo")
        #expect(catalog.unavailableReason(for: "qwen3.6-35b-a3b@qwen36")?
            .contains("another install") == true)
    }

    @Test func earlierRootWinsBetweenInstallsWithTheSameName() throws {
        func install(_ root: String) -> ServerLibraryEntry {
            ServerLibraryEntry(modelID: "minicpm5-2b-int4g64@minicpm5", familyModelID: "minicpm5-2b-int4g64",
                               basename: "minicpm5",
                               directory: URL(fileURLWithPath: "\(root)/minicpm5.gturbo", isDirectory: true),
                               family: .minicpm5)
        }
        let index = ServerLibraryIndex(entries: [install("/a-scratch"), install("/z-models")])
        let catalog = ServerEngineCatalog.catalog(
            for: index, scope: .product,
            rootOrder: [URL(fileURLWithPath: "/z-models"), URL(fileURLWithPath: "/a-scratch")])
        #expect(try #require(catalog.descriptor(for: "minicpm5-2b-int4g64")).location.path
                == "/z-models/minicpm5.gturbo")
    }

    @Test func contextIsTheConfiguredLengthOrTheFamilyMaximum() throws {
        let catalog = ServerEngineCatalog.catalog(for: Self.index, scope: .product)
        let maple = try #require(catalog.descriptor(for: "maple-preview-2bit-mlx"))
        #expect(maple.resolvedContext(options: EngineLoadOptions(nativeMaxContext: 8_192)) == 8_192)
        #expect(maple.resolvedContext(options: EngineLoadOptions(nativeMaxContext: nil))
                == ModelFamily.maple.maximumContext)
    }

    @Test func listingShowsTheEngineColumn() {
        let catalog = ServerEngineCatalog.catalog(for: Self.index, scope: .product,
                                                  installedBytes: { _ in 1_000 })
        let lines = ServerLibraryListing.text(for: catalog).split(separator: "\n").map(String.init)
        #expect(lines.first?.hasPrefix("MODEL") == true)
        #expect(lines.first?.contains("ENGINE") == true)
        #expect(lines.count == 4)
        #expect(lines.dropFirst().allSatisfy { $0.contains("mference") && $0.contains("1000") })
        #expect(ServerLibraryListing.text(for: .empty) == ServerLibraryListing.emptyListing)
        let onlyOthers = ServerEngineCatalog.catalog(
            for: ServerLibraryIndex(entries: [entry("gemma-4-26b-a4b-it", family: .gemma4, basename: "gemma4")]),
            scope: .product)
        #expect(ServerLibraryListing.text(for: onlyOthers)
                == "no product models installed (1 other install; --catalog all lists them)")
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

    /// `serving` is the production wiring: the default product catalog, the
    /// explicit `--model` preference and root order, and nothing loaded.
    @Test func servingUsesTheProductCatalogByDefault() throws {
        let product = ServerModelLibrary.serving(Self.index, arguments: try ServerArguments.parse(["--library"]))
        #expect(product.snapshot.catalog.models.map(\.id)
                == ["maple-preview-2bit-mlx", "minicpm5-2b-int4g64", "qwen3.6-35b-a3b"])
        #expect(product.readyDetail == "catalog=product models=3")
        #expect(product.snapshot.health.model == nil)
        let preferred = ServerModelLibrary.serving(
            Self.index,
            arguments: try ServerArguments.parse(["--library", "--model", "/models/a-quant.gturbo"]))
        #expect(preferred.snapshot.descriptor(for: "qwen3.6-35b-a3b")?.location.lastPathComponent
                == "a-quant.gturbo")
        let all = ServerModelLibrary.serving(Self.index,
                                             arguments: try ServerArguments.parse(["--library", "--catalog", "all"]))
        #expect(all.snapshot.catalog.models.count == Self.index.entries.count)
        #expect(all.readyDetail == "catalog=all models=5")
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
                                                  entry("b", family: .qwen36, basename: "b")]),
                scope: .all),
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
        let index = ServerLibraryIndex(entries: [entry("gemma-4-26b-a4b-it", family: .gemma4, basename: "gemma4")])
        let library = ServerModelLibrary(
            catalog: ServerEngineCatalog.catalog(for: index, scope: .product),
            backends: [ServerLoaderBackend { _ in BridgedModel(Events(), "x") }],
            loadOptions: EngineLoadOptions())
        do {
            try await library.preload(directory: URL(fileURLWithPath: "/models/gemma4.gturbo"))
            Issue.record("preload of a declined install succeeded")
        } catch let error as ServerArgumentError {
            #expect(String(describing: error).contains("--catalog all"))
        }
    }

    /// The default product catalog over discovery, through HTTP: a product
    /// model is served by the bridged session, anything else is a 400 that
    /// says why, and the listing names only the product model.
    @Test func productCatalogOverDiscoveryServesAndDeclines() async throws {
        let events = Events()
        let index = ServerLibraryIndex(entries: [
            entry("qwen3.6-35b-a3b", family: .qwen36, basename: "qwen36"),
            entry("gemma-4-26b-a4b-it", family: .gemma4, basename: "gemma4"),
        ])
        let library = ServerModelLibrary(
            catalog: ServerEngineCatalog.catalog(for: index, scope: .product),
            backends: [ServerLoaderBackend { directory in
                let id = directory.deletingPathExtension().lastPathComponent
                events.append("load \(id)")
                return BridgedModel(events, id)
            }],
            loadOptions: EngineLoadOptions())
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
        let listing = try await URLSession.shared.data(
            from: URL(string: "http://127.0.0.1:\(port)/v1/models")!).0
        #expect(String(decoding: listing, as: UTF8.self).contains(#""id":"qwen3.6-35b-a3b""#))
        #expect(!String(decoding: listing, as: UTF8.self).contains("gemma"))

        let (declinedStatus, declinedBody) = try await post("gemma-4-26b-a4b-it")
        #expect(declinedStatus == 400)
        #expect(declinedBody.contains("model_not_runnable") && declinedBody.contains("--catalog all"))
        #expect(events.all.isEmpty)

        let (servedStatus, servedBody) = try await post("qwen3.6-35b-a3b")
        #expect(servedStatus == 200, "\(servedBody)")
        #expect(servedBody.contains(#""content":"qwen36""#))
        #expect(events.all == ["load qwen36"])
        try await server.shutdown()
        #expect(events.all == ["load qwen36", "release qwen36"])
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

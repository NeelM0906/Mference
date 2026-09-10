import Darwin
import Foundation
import MferenceServerCore

let arguments: ServerArguments
do {
    arguments = try ServerArguments.parse(Array(CommandLine.arguments.dropFirst()))
} catch ServerArgumentError.help {
    print(ServerArguments.usage)
    exit(0)
} catch {
    FileHandle.standardError.write(Data("error: \(error)\n\n\(ServerArguments.usage)\n".utf8))
    exit(2)
}

do {
    let signals = ServerTerminationSignals()
    let host = try arguments.bindMode.host()
    let explicitModelURL = arguments.model.map {
        URL(fileURLWithPath: $0).standardizedFileURL
    }
    let server: MferenceHTTPServer
    let readyDetail: String

    if let libraryOption = arguments.library {
        let roots = libraryOption.resolvedRoots()
        let index = ServerLibraryDiscovery.discover(
            roots: roots,
            explicitModelDirectory: explicitModelURL)
        index.logDiscovery()
        guard !index.entries.isEmpty else {
            let searched = roots.map(\.path).joined(separator: ", ")
            throw ServerArgumentError.invalid(
                "--library found no completed installs under: \(searched)")
        }
        let maxContext = arguments.maxContext
        let promptCacheMode = arguments.promptCacheMode
        let library = ServerModelLibrary(index: index) { directory in
            try await ServerModelSession.load(modelDirectory: directory,
                                              maxContext: maxContext,
                                              promptCacheMode: promptCacheMode)
        }
        // `--model` alongside `--library` preloads one install; without it the
        // first request pays the load. Either way exactly one model is ever
        // resident, and no second model process is started.
        if let explicitModelURL {
            guard let preload = index.entries.first(where: {
                $0.directory == explicitModelURL
            }) else {
                throw ServerArgumentError.invalid(
                    "--model \(explicitModelURL.path) is not a completed install")
            }
            try await library.preload(modelID: preload.modelID)
        }
        server = MferenceHTTPServer(library: library, queueLimit: arguments.queueLimit)
        readyDetail = "mode=library models=\(index.entries.count)"
    } else {
        guard let explicitModelURL else {
            throw ServerArgumentError.invalid("--model is required")
        }
        let backend = try await ServerModelSession.load(
            modelDirectory: explicitModelURL,
            maxContext: arguments.maxContext,
            promptCacheMode: arguments.promptCacheMode)
        let modelID = arguments.modelIDOverride ?? backend.defaultModelID
        server = MferenceHTTPServer(
            modelID: modelID,
            queueLimit: arguments.queueLimit,
            backend: backend,
            chatDialect: backend.chatDialect)
        readyDetail = "model=\(modelID)"
    }

    _ = try await server.start(host: host, port: arguments.port)
    print("MferenceServer ready at http://\(host):\(arguments.port) \(readyDetail) context=\(arguments.maxContext) prompt_cache=\(arguments.promptCacheMode.rawValue)")
    // Supervisors watch for the ready line through a pipe or log file, where
    // stdout is block-buffered and would otherwise hold it back indefinitely.
    fflush(stdout)

    _ = await signals.wait()
    try await server.shutdown()
    await signals.cancel()
} catch {
    FileHandle.standardError.write(Data("error: \(error)\n".utf8))
    exit(1)
}

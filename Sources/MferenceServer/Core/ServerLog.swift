import Foundation
import MferenceEngine

/// Request lifecycle logging. Responses stay deliberately generic so runtime
/// details never reach a client; the operator needs the opposite, so the log
/// carries the underlying error verbatim. Local stderr only.
///
/// Start and completion are both logged: a long prefill emits no output for
/// minutes, and without a start line that is indistinguishable from a wedged
/// server.
enum ServerLog {
    static func requestStarted(id: String, streaming: Bool) {
        write("request \(id) started streaming=\(streaming)")
    }

    static func requestCompleted(id: String,
                                 duration: Duration,
                                 completion: ServerCompletion) {
        let usage = completion.usage
        let prefill = completion.prefillSeconds.map { " prefill=" + String(format: "%.3fs", $0) } ?? ""
        write("""
        request \(id) completed in \(format(duration)) \
        prompt=\(usage.promptTokens) \
        cached=\(usage.promptTokensDetails.cachedTokens) \
        completion=\(usage.completionTokens) \
        finish=\(completion.finishReason)\(prefill)
        """)
        if let diagnostics = completion.diagnostics, let json = try? diagnostics.jsonLine() {
            write("request \(id) runtime-diagnostics \(json)")
        }
    }

    /// The multi-prefix cache gave memory back after memory pressure or swap
    /// growth.
    static func promptCacheShed(reason: String, releasedBytes: Int) {
        write("prompt cache shed reason=\(reason) released=\(releasedBytes)B")
    }

    static func requestFailed(id: String, status: UInt, streaming: Bool, error: any Error) {
        write(requestFailureMessage(id: id, status: status, streaming: streaming, error: error))
    }

    /// Cancellation is an expected lifecycle outcome (disconnect or shutdown),
    /// not evidence of a model/server fault. This only classifies operator
    /// logs; it does not change the HTTP/SSE response or suppress other errors.
    static func requestFailureMessage(id: String, status: UInt, streaming: Bool, error: any Error) -> String {
        if error is CancellationError {
            return "request \(id) cancelled streaming=\(streaming)"
        }
        let detail = switch error {
        case let error as ServerRequestError: describe(error)
        default: String(describing: error)
        }
        return "request \(id) failed status=\(status) streaming=\(streaming) error=\(detail)"
    }

    static func streamAborted(id: String, reason: String) {
        write("request \(id) stream aborted: \(reason)")
    }

    /// Library mode: the engine host's lifecycle in the server's log format.
    /// A swap from A to B reads
    ///
    ///     swap started from=A to=B
    ///     model released model=A reason=swap
    ///     swap finished model=B in 12.3s
    ///
    /// The release line comes once A's memory is gone and before B's load
    /// begins, which is the order memory validation checks. A cold load is
    /// `swap started to=B` then `swap finished` (or `swap failed`). A load emits
    /// nothing for as long as first-touch SHA-256 verification takes, so the
    /// start line is what distinguishes a swap in progress from a stalled
    /// request.
    static func hostEventMessage(_ event: EngineHostEvent) -> String? {
        switch event {
        case .unloadStarted(let modelID, .swap(let target)):
            "swap started from=\(modelID) to=\(target)"
        case .unloadStarted:
            nil
        case .unloadFinished(let modelID, let reason):
            "model released model=\(modelID) reason=\(reason.logName)"
        case .loadStarted(let modelID, nil):
            "swap started to=\(modelID)"
        case .loadStarted:
            nil
        case .loadFinished(let modelID, let seconds):
            "swap finished model=\(modelID) in \(String(format: "%.1fs", seconds))"
        case .loadFailed(let modelID, let message):
            "swap failed model=\(modelID) error=\(message)"
        }
    }

    static func hostEvent(_ event: EngineHostEvent) {
        if let message = hostEventMessage(event) { write(message) }
    }

    /// Reported once per candidate the library declined, so a partial or gated
    /// install is visibly absent rather than silently missing.
    static func librarySkipped(directory: String, reason: String) {
        write("library skipped \(directory): \(reason)")
    }

    static func libraryReady(models: [String]) {
        write("library ready models=\(models.joined(separator: ","))")
    }

    /// `--verify auto` fell back to full SHA-256 for this load (see
    /// `ModelIntegrityPolicy.autoFallbackNotice`).
    static func verifyAutoFallback(_ notice: String) {
        write(notice)
    }

    private static func format(_ duration: Duration) -> String {
        let seconds = Double(duration.components.seconds)
            + Double(duration.components.attoseconds) / 1e18
        return String(format: "%.1fs", seconds)
    }

    private static func describe(_ error: ServerRequestError) -> String {
        let detail = error.envelope.error
        return "\(detail.code): \(detail.message)"
    }

    private static func write(_ message: String) {
        let line = "[\(Date().formatted(.iso8601))] \(message)\n"
        FileHandle.standardError.write(Data(line.utf8))
    }
}

extension EngineUnloadReason {
    var logName: String {
        switch self {
        case .swap: "swap"
        case .idle: "idle"
        case .requested: "requested"
        case .shutdown: "shutdown"
        }
    }
}

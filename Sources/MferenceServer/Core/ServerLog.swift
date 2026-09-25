import Foundation

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

    /// Library mode only. A load emits nothing for as long as first-touch
    /// SHA-256 verification takes, so the start line is what distinguishes a
    /// swap in progress from a stalled request.
    static func modelLoadStarted(_ modelID: String) {
        write("swap started to=\(modelID)")
    }

    static func modelSwapStarted(from: String, to: String) {
        write("swap started from=\(from) to=\(to)")
    }

    static func modelSwapFinished(model: String, duration: Duration) {
        write("swap finished model=\(model) in \(format(duration))")
    }

    static func modelSwapFailed(model: String, error: any Error) {
        write("swap failed model=\(model) error=\(String(describing: error))")
    }

    /// Reported once per candidate the library declined, so a partial or gated
    /// install is visibly absent rather than silently missing.
    static func librarySkipped(directory: String, reason: String) {
        write("library skipped \(directory): \(reason)")
    }

    static func libraryReady(models: [String]) {
        write("library ready models=\(models.joined(separator: ","))")
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

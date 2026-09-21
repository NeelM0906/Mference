/// Frames streamed reasoning for standard error (`--show-reasoning`).
///
/// Thoughts arrive as deltas before, and for tool-free chat only before, the
/// visible answer. Each thought becomes one `[reasoning]` ... `[/reasoning]`
/// block, so it stays readable next to the prompt, notices and timing footer
/// that share standard error, and never touches standard output.
struct ReasoningEcho {
    private var open = false
    private var endsLine = true

    /// Text to write for a reasoning delta; the first one opens the block.
    mutating func reasoning(_ delta: String) -> String {
        guard !delta.isEmpty else { return "" }
        let prefix = open ? "" : "[reasoning]\n"
        open = true
        endsLine = delta.hasSuffix("\n")
        return prefix + delta
    }

    /// Text to write before visible output and at the end of the turn; closes
    /// an open block on its own line.
    mutating func close() -> String {
        guard open else { return "" }
        open = false
        return (endsLine ? "" : "\n") + "[/reasoning]\n"
    }
}

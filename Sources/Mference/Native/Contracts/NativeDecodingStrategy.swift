extension Native {
    /// How the next tokens are proposed. This program ships only plain
    /// decoding (no drafts); MTP, DFlash2 and prompt lookup implement this
    /// hook later. Every proposal is verified exactly (D1: speculative output
    /// is byte-identical to the same engine's plain greedy decode).
    public protocol DecodingStrategy: AnyObject {
        var name: String { get }
        /// Maximum draft rows per verify step; 0 means plain decoding.
        var maxDraftRows: Int { get }
        /// Start a generation over `prompt`.
        func reset(prompt: [Int32])
        /// Draft tokens to verify after `history` (empty for plain).
        func draft(history: [Int32]) -> [Int32]
        /// Feedback after a verify step.
        func observe(proposed: Int, accepted: Int)
    }
}

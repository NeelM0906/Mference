extension Native {
    /// The frozen per-type pack layout (D3). A pack writer stores every tensor
    /// of `type` in `layout(for: type)` and falls back to `.gguf` only where
    /// the layout does not `fit` the tensor; the fast `QuantGEMM` reads
    /// exactly these layouts. The table is the same on every GPU family.
    ///
    /// Admissibility: an entry must fit every Quick and Deep tensor of its
    /// type (so neither model ever falls back), which a test checks against
    /// both models' tensor shapes (required with any non-identity entry).
    /// `MferenceRepackCore` cannot import this module, so the production writer (W1) mirrors `TileLayout`,
    /// `QuantBlockFormat` and this table, and a test in `MferenceTestsCore`
    /// (which links both targets) pins the mirror to these definitions.
    ///
    /// Version 1 is the identity for every type. Changing an entry (for
    /// example once W2's kernels show a tiled layout is faster) is a
    /// coordinator contract PR naming W1 and W2; existing packs record their
    /// layouts per tensor, so they stay readable, and are rewritten by a
    /// repack.
    public enum PackLayoutTable {
        public static let version = 1

        public static func layout(for type: GGMLType) -> TileLayout {
            switch type {
            default: return .gguf
            }
        }

        /// The layout a writer records for a tensor of `type` with `rows`
        /// rows of `blocksPerRow` blocks.
        public static func layout(for type: GGMLType, rows: Int, blocksPerRow: Int) -> TileLayout {
            let preferred = layout(for: type)
            return preferred.fits(rows: rows, blocksPerRow: blocksPerRow) ? preferred : .gguf
        }
    }
}

import Foundation
import Metal

/// How the runtime must treat RMSNorm weights for the loaded family.
///
/// Qwen4-Exp (Flash-Next) initializes every RMSNorm weight at zero and applies
/// `(1 + weight)`. The port's decision is to bake the `+1` into the stored
/// weights so the runtime's standard RMSNorm applies unchanged — the same
/// conversion the Qwen 3.8 MTP attach already performs at repack
/// (`MTPAttachTool`, `addOne`).
///
/// The W2 install path copies norms verbatim today, so the bake happens here,
/// at load, gated on the family. Once the repacker folds it in, the install
/// publishes `manifest.zeroCenteredNormsBakedAtInstall = true` and this policy
/// resolves to `.storedInFullForm`, leaving the loader a pass-through. The
/// two paths must never both apply: that is what the manifest flag decides.
public enum ZeroCenteredNormPolicy: Sendable, Equatable {
    /// Stored weights are already `1 + w`; hand them to the kernels as they are.
    case storedInFullForm
    /// Stored weights are the zero-centered `w`; widen to `1 + w` at load.
    case bakeAtLoad
}

extension Model {

    // MARK: - Zero-centered norm policy

    /// Families whose checkpoint RMSNorm weights are zero-centered, i.e. whose
    /// reference implementation applies `(1 + w)`.
    ///
    /// Qwen 3.8's own trunk is *not* here: its shipped MLX conversion stores
    /// norms in full form already, and only the separately attached MTP shard
    /// carries the zero-centered convention, which `MTPAttachTool` converts at
    /// attach time. Adding a family to this set changes the bytes every kernel
    /// sees, so it is an explicit list rather than a heuristic.
    static let zeroCenteredNormFamilies: Set<ModelFamily> = [.qwen38flashnext]

    /// Whether this load must apply the `(1 + w)` bake itself.
    public var zeroCenteredNormPolicy: ZeroCenteredNormPolicy {
        guard Self.zeroCenteredNormFamilies.contains(config.family) else {
            return .storedInFullForm
        }
        return manifest.zeroCenteredNormsBakedAtInstall == true
            ? .storedInFullForm : .bakeAtLoad
    }

    /// Tensor-name suffixes of the norms the reference zero-centers.
    ///
    /// Every `Qwen4ExpTextRMSNorm` instance in the text stack: the attention
    /// per-head q/k norms, the QSA indexer's q/k layernorms, the
    /// hyper-connection group norms (per-site and the global mixer's), and the
    /// PLE block's three norms. All of them are weight-zero-initialized and
    /// applied as `(1 + w)`.
    ///
    /// Deliberately **excluded**:
    ///   * `linear_attn.norm.weight` — the gated DeltaNet norm is
    ///     `Qwen3_5RMSNormGated`, which is **ones**-initialized and applies
    ///     `w` directly. Confirmed against the reference implementation
    ///     (2026-09-01 parity harness), not inferred: it is the one norm in
    ///     this stack that is not zero-centered, and baking it would add one
    ///     to an already-full-form weight.
    ///   * the `mtp.*` sidecar's norms: MTP draft decode is out of scope for
    ///     v1, and its norms need the same treatment when it lands.
    ///
    /// The reference RMSNorm upcasts internally —
    /// `_norm(x.float()) * (1 + w.float())`, cast back afterwards — so a
    /// kernel consuming these baked weights must accumulate in fp32 to match.
    static let zeroCenteredNormSuffixes: [String] = [
        ".self_attn.q_norm.weight",
        ".self_attn.k_norm.weight",
        ".self_attn.indexer.q_layernorm.weight",
        ".self_attn.indexer.k_layernorm.weight",
        ".attn_hyper_connection.hc_norm.weight",
        ".mlp_hyper_connection.hc_norm.weight",
        ".hyper_connection_mixer.hc_norm.weight",
        ".ple.norm_conv.weight",
        ".ple.norm_key.weight",
        ".ple.norm_query.weight",
    ]

    /// Whether `name` is one of the norms the `(1 + w)` bake applies to.
    static func isZeroCenteredNorm(_ name: String) -> Bool {
        zeroCenteredNormSuffixes.contains { name.hasSuffix($0) }
    }

    /// Resolve a norm weight, applying the family's `(1 + w)` bake when the
    /// install has not already done it.
    ///
    /// The resident buffer is a read-only mapping of the install, so the bake
    /// materializes a BF16 copy in a private shared-storage buffer, cached by
    /// name for the model's lifetime. Norms are `[10240]` at worst — 20 KB —
    /// and there are a few hundred of them, so the copies cost single-digit
    /// megabytes against a 175 GB install.
    ///
    /// Names outside `zeroCenteredNormSuffixes`, and every family outside
    /// `zeroCenteredNormFamilies`, return the mapped tensor untouched: this
    /// accessor is safe to route all norm reads through.
    public func normWeight(name: String) throws -> TensorView {
        let source = try resident(name: name)
        guard zeroCenteredNormPolicy == .bakeAtLoad,
              Self.isZeroCenteredNorm(name) else { return source }
        guard source.dtype == 1 else {
            throw ModelError.indexCorrupt(
                detail: "zero-centered norm \(name) is dtype \(source.dtype), not BF16")
        }
        let key = "1+w:" + name
        if let cached = streamersQueue.sync(execute: { convertedBox.views[key] }) {
            return cached
        }
        let count = Int(source.length) / MemoryLayout<UInt16>.stride
        guard let buffer = device.makeBuffer(
            length: max(1, count * MemoryLayout<UInt16>.stride),
            options: .storageModeShared) else {
            throw ModelError.residentBufferWrapFailed
        }
        let src = source.buffer.contents().advanced(by: Int(source.offset))
            .assumingMemoryBound(to: UInt16.self)
        let dst = buffer.contents().assumingMemoryBound(to: UInt16.self)
        for i in 0..<count {
            // Identical to the MTP attach conversion, bit for bit: widen the
            // stored BF16 to float, add one, round back to BF16.
            dst[i] = Quantization.bf16Bits(Quantization.bf16ToFloat(src[i]) + 1.0)
        }
        let view = TensorView(
            buffer: buffer, offset: 0, length: source.length,
            scaleOffset: 0, scaleLength: 0, biasOffset: 0, biasLength: 0,
            shape: source.shape, dtype: 1)
        streamersQueue.sync { convertedBox.views[key] = view }
        return view
    }

    // MARK: - Low-rank hyper-connections

    private func hcSite(_ site: HyperConnectionSite, layer L: Int) -> String {
        "\(trunkPrefix)layers.\(L).\(site.rawValue)."
    }

    /// The two per-layer hyper-connection sites. Each owns a complete
    /// `GatedResidual`: mix down/up, block inject, and a group norm.
    public enum HyperConnectionSite: String, Sendable {
        case attention = "attn_hyper_connection"
        case mlp = "mlp_hyper_connection"
    }

    /// `[hcLowRank, hcCount * hidden]` — collapses the normed residual bundle
    /// to the low-rank mix vector.
    public func hcMixDown(site: HyperConnectionSite, layer L: Int) throws -> TensorView {
        try resident(name: hcSite(site, layer: L) + "input_mix_weight_down.weight")
    }
    /// `[hcCount * hidden, hcLowRank]` — expands the mix vector back to a
    /// per-channel sigmoid gate over the bundle.
    public func hcMixUp(site: HyperConnectionSite, layer L: Int) throws -> TensorView {
        try resident(name: hcSite(site, layer: L) + "input_mix_weight_up.weight")
    }
    /// `[hcCount, hcCount * hidden]` — the per-stream placement weights the
    /// block output is injected with.
    public func hcInject(site: HyperConnectionSite, layer L: Int) throws -> TensorView {
        try resident(name: hcSite(site, layer: L) + "block_inject_weight.weight")
    }
    /// `[hcCount * hidden]` — group RMSNorm weight, group size `hidden`: each
    /// stream is normalized independently over its own channels against one
    /// shared bundle-wide vector.
    public func hcNorm(site: HyperConnectionSite, layer L: Int) throws -> TensorView {
        try normWeight(name: hcSite(site, layer: L) + "hc_norm.weight")
    }

    /// The global mixer that collapses the residual bundle to `hidden` after
    /// the last layer. It is a `GatedResidual` without the inject path, and it
    /// stands in for the final norm — this family has none.
    public var hcGlobalMixDown: TensorView {
        get throws { try resident(name: "\(trunkPrefix)hyper_connection_mixer.input_mix_weight_down.weight") }
    }
    public var hcGlobalMixUp: TensorView {
        get throws { try resident(name: "\(trunkPrefix)hyper_connection_mixer.input_mix_weight_up.weight") }
    }
    public var hcGlobalNorm: TensorView {
        get throws { try normWeight(name: "\(trunkPrefix)hyper_connection_mixer.hc_norm.weight") }
    }

    // MARK: - QSA indexer

    private func indexerPrefix(layer L: Int) -> String {
        "\(trunkPrefix)layers.\(L).self_attn.indexer."
    }

    /// `[(indexerNumHeads + indexerNumKVHeads) * indexerHeadDim, hidden]`,
    /// fused: the query heads first, then the single key head.
    public func indexerQKProj(layer L: Int) throws -> TensorView {
        try resident(name: indexerPrefix(layer: L) + "index_qk_proj.weight")
    }
    /// `[indexerHeadDim]`, zero-centered.
    public func indexerQNorm(layer L: Int) throws -> TensorView {
        try normWeight(name: indexerPrefix(layer: L) + "q_layernorm.weight")
    }
    /// `[indexerHeadDim]`, zero-centered. Applied to the pooled block key
    /// *after* the float32 mean, not to the raw per-token keys.
    ///
    /// Note for whatever consumes these: the indexer does **not** guarantee
    /// that a query's own block survives selection. There is no "always keep
    /// self" rule anywhere in the reference — only the top-k over block scores
    /// plus the always-selected incomplete tail — so no layer of this plumbing
    /// may add one.
    public func indexerKNorm(layer L: Int) throws -> TensorView {
        try normWeight(name: indexerPrefix(layer: L) + "k_layernorm.weight")
    }

    // MARK: - PLE n-gram embedding

    private func plePrefix(layer L: Int) -> String {
        "\(trunkPrefix)layers.\(L).ple."
    }

    /// `[hcCount * hidden, hidden]` — projects the gathered n-gram embedding
    /// to the per-stream key.
    public func pleKeyProj(layer L: Int) throws -> TensorView {
        try resident(name: plePrefix(layer: L) + "key_proj.weight")
    }
    /// `[hidden, hidden]` — the value the per-stream gate scales.
    public func pleValueProj(layer L: Int) throws -> TensorView {
        try resident(name: plePrefix(layer: L) + "value_proj.weight")
    }
    /// `[hcCount * hidden, 1, pleConvKernelSize]` — depthwise causal conv over
    /// the gated value, dilated by the n-gram size.
    public func pleConv1d(layer L: Int) throws -> TensorView {
        try resident(name: plePrefix(layer: L) + "conv1d.weight")
    }
    public func pleNormConv(layer L: Int) throws -> TensorView {
        try normWeight(name: plePrefix(layer: L) + "norm_conv.weight")
    }
    public func pleNormKey(layer L: Int) throws -> TensorView {
        try normWeight(name: plePrefix(layer: L) + "norm_key.weight")
    }
    public func pleNormQuery(layer L: Int) throws -> TensorView {
        try normWeight(name: plePrefix(layer: L) + "norm_query.weight")
    }

    /// The three splitmix64-derived n-gram hash multipliers, I64 `[3]`.
    ///
    /// **Loaded, never re-derived.** The reference derives them from a seed and
    /// the layer index, but the installed values are the contract: a
    /// re-derivation that disagreed by one bit would index a different row and
    /// be undetectable at load. Their count is also the n-gram size.
    public func pleLayerMultipliers(layer L: Int) throws -> [Int64] {
        try residentInt64(name: plePrefix(layer: L) + "ple_embedding.layer_multipliers")
    }
    /// Per-head base row offsets into the pool, I64 `[ngramHeads]`. Loaded,
    /// never re-derived.
    public func pleNgramHeadOffsets(layer L: Int) throws -> [Int64] {
        try residentInt64(name: plePrefix(layer: L) + "ple_embedding.ngram_heads_offsets")
    }
    /// Per-head vocabulary sizes (consecutive primes), I64 `[ngramHeads]`.
    /// A head's row is `hash % vocabSizes[h] + offsets[h]`.
    public func pleNgramHeadVocabSizes(layer L: Int) throws -> [Int64] {
        try residentInt64(name: plePrefix(layer: L) + "ple_embedding.ngram_heads_vocab_sizes")
    }

    /// N-gram size for a PLE layer, i.e. `layer_multipliers.count`. The token
    /// history the runtime must cache is one less than this.
    public func pleNgramSize(layer L: Int) throws -> Int {
        try pleLayerMultipliers(layer: L).count
    }

    /// Read an I64 lookup table out of the resident mapping.
    ///
    /// These are CPU-side hash tables, not kernel operands: the loader hands
    /// back a typed `[Int64]` rather than a `TensorView` so a caller cannot
    /// accidentally bind a 64-bit integer buffer as float data.
    func residentInt64(name: String) throws -> [Int64] {
        let view = try resident(name: name)
        guard view.dtype == 4 else {
            throw ModelError.indexCorrupt(
                detail: "\(name) is dtype \(view.dtype); expected I64 (4)")
        }
        let count = Int(view.length) / MemoryLayout<Int64>.stride
        guard count > 0, UInt64(count * MemoryLayout<Int64>.stride) == view.length else {
            throw ModelError.tensorSizeMismatch(
                name: name,
                expected: UInt64(count * MemoryLayout<Int64>.stride),
                actual: view.length)
        }
        let base = view.buffer.contents().advanced(by: Int(view.offset))
        return (0..<count).map { index in
            var value: Int64 = 0
            withUnsafeMutableBytes(of: &value) { destination in
                destination.copyMemory(from: UnsafeRawBufferPointer(
                    start: base.advanced(by: index * MemoryLayout<Int64>.stride),
                    count: MemoryLayout<Int64>.stride))
            }
            return Int64(littleEndian: value)
        }
    }

    // MARK: - PLE row pool

    /// Validated geometry of the layer's streamed n-gram row pool.
    public func plePoolGeometry(layer L: Int) throws -> PleRowPoolGeometry {
        guard let pool = manifest.plePool else {
            throw ModelError.plePoolMissing(layer: L)
        }
        guard pool.kind == PleRowPool.supportedKind else {
            throw ModelError.plePoolInvalid(detail: "unknown kind \"\(pool.kind)\"")
        }
        guard let entry = pool.layers.first(where: { $0.layer == L }) else {
            throw ModelError.plePoolMissing(layer: L)
        }
        return try PleRowPoolGeometry(layer: entry, hiddenSize: config.hiddenSize)
    }

    /// Open the layer's row pool. The caller owns the returned reader (and its
    /// file descriptor and cache slab); one per PLE layer is enough.
    ///
    /// The installed `ngram_heads_*` tables are cross-checked against the
    /// pool's derived head count here, because that is the one place both
    /// facts are in hand: a table of the wrong length would otherwise surface
    /// as a silently wrong row index at decode.
    public func openPleRowPool(layer L: Int,
                               cacheRows: Int = PleRowPool.defaultCacheRows) throws -> PleRowPool {
        let geometry = try plePoolGeometry(layer: L)
        let offsets = try pleNgramHeadOffsets(layer: L)
        let vocabSizes = try pleNgramHeadVocabSizes(layer: L)
        guard offsets.count == geometry.ngramHeads,
              vocabSizes.count == geometry.ngramHeads else {
            throw ModelError.plePoolInvalid(
                detail: "layer \(L): pool geometry implies \(geometry.ngramHeads) "
                    + "n-gram heads but the checkpoint carries "
                    + "\(offsets.count) offsets / \(vocabSizes.count) vocab sizes")
        }
        for head in 0..<geometry.ngramHeads {
            let last = offsets[head] + vocabSizes[head]
            guard offsets[head] >= 0, vocabSizes[head] > 0,
                  last <= Int64(geometry.rows) else {
                throw ModelError.plePoolInvalid(
                    detail: "layer \(L): n-gram head \(head) spans rows "
                        + "[\(offsets[head]), \(last)) outside the pool's "
                        + "\(geometry.rows) rows")
            }
        }
        return try PleRowPool(directoryURL: directoryURL,
                              geometry: geometry,
                              cacheRows: cacheRows)
    }

    // MARK: - Sidecars

    /// Whether the install carried an optional source tensor group (`mtp`,
    /// `vision`). Absent means the install predates the sidecar policy.
    public func sidecarCarried(_ group: String) -> Bool? {
        manifest.sidecars?[group]?.carried
    }

    /// A routed-expert pool installed outside `packed_experts/` — today only
    /// the MTP draft layer's own 512 experts. Nothing reads these yet; the
    /// accessor exists so the streaming layer can be pointed at them when MTP
    /// draft decode lands, without another manifest change.
    public func auxiliaryExpertPool(named name: String) -> ManifestAuxiliaryExpertPool? {
        manifest.auxiliaryExpertPools?.first { $0.name == name }
    }
}

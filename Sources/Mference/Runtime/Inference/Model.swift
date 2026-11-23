import Foundation
import Metal
import Darwin

public struct ModelLoadStats: Sendable {
    public var manifestSha256Nanos: UInt64
    public var receiptValidationNanos: UInt64
    public var eagerSha256Nanos: UInt64
    /// Resident mode only: opening every routed layer and touching its
    /// mapping at load, so page-in cost is not billed to decode.
    public var residentWarmupNanos: UInt64

    public init(manifestSha256Nanos: UInt64 = 0,
                receiptValidationNanos: UInt64 = 0,
                eagerSha256Nanos: UInt64 = 0,
                residentWarmupNanos: UInt64 = 0) {
        self.manifestSha256Nanos = manifestSha256Nanos
        self.receiptValidationNanos = receiptValidationNanos
        self.eagerSha256Nanos = eagerSha256Nanos
        self.residentWarmupNanos = residentWarmupNanos
    }
}

/// Bounded routed-expert cache configuration.
public enum ExpertStreamingMode: Sendable {
    /// Read each expert into one of `slotCount` 2 MB-aligned cache slots.
    case pread(slotCount: Int)
    /// Map every layer file once and serve experts as subregion views.
    /// No slots, no bookkeeping, no reads on the hot path.
    case resident
}

/// Per-layer routed-expert backend selected by `ExpertStreamingMode`.
enum ExpertBackend {
    case pread(PreadExpertStreamer)
    case resident(ResidentExpertStreamer)

    func loadExpert(layer: Int, expert: Int) throws
        -> (buffer: MTLBuffer, offset: UInt64, size: UInt64) {
        switch self {
        case .pread(let streamer):
            return try streamer.loadExpert(layer: layer, expert: expert)
        case .resident(let streamer):
            return try streamer.expertBuffer(layer: layer, expert: expert)
        }
    }
}

/// Loaded `.gturbo/` model. Resident weights live behind one mmap'd
/// `MTLBuffer`; routed expert weights live behind per-layer streaming
/// backends opened lazily on first touch.
public struct Model {
    public let device: MTLDevice
    public let config: ArchConfig
    public let streamingMode: ExpertStreamingMode
    public let expertCachePolicy: ExpertCachePolicy
    public let integrityPolicy: ModelIntegrityPolicy
    /// Under `.trustedReceiptWhenValid`, the receipt check that failed and
    /// sent the load to full SHA-256; nil when no fallback happened.
    public internal(set) var integrityFallbackDetail: String?
    /// The one-line notice for `integrityFallbackDetail`, if any.
    public var verifyAutoFallbackNotice: String? {
        integrityFallbackDetail.map(ModelIntegrityPolicy.autoFallbackNotice(detail:))
    }
    public var modelID: String { manifest.modelID }
    public var sourceSnapshotHash: String? { manifest.sourceSnapshotHash }
    public var sharedExpertWeightBits: Int { manifest.quant?.sharedExpert.weightBits ?? 8 }
    public var routedExpertWeightBits: Int { manifest.quant?.routedExpert.weightBits ?? 4 }

    /// Every install uses the group-64 specialization, including families
    /// whose attention weights are not INT4 at all.
    var affineInt4GroupSize: Int { Quantization.groupSize }

    let residentBuffer: ResidentBuffer
    let residentIndex: ResidentIndex
    let packedExpertsLayout: PackedExpertsLayout
    let manifest: Manifest
    let directoryURL: URL

    /// Lazy state. Held inside a reference box so `Model` can stay a struct
    /// while still letting accessors mutate layer state via a serial queue.
    let streamersBox: StreamersBox
    let streamersQueue: DispatchQueue
    /// Whether prefill sweeps read their experts past the page cache, decided
    /// once at load by `RuntimeConfiguration.prefillSweepBypassesPageCache`.
    let prefillSweepBypassesPageCache: Bool

    /// Lazily converted FP32 copies of small BF16 tensors that kernels bind
    /// as `device const float*` (DeepSeek V4's compressor `ape` tables).
    final class ConvertedTensorBox: @unchecked Sendable {
        var views: [String: TensorView] = [:]
    }
    let convertedBox = ConvertedTensorBox()

    final class StreamersBox: @unchecked Sendable {
        var streamers: [ExpertBackend?]
        var layerVerified: [Bool]
        init(numLayers: Int) {
            self.streamers = Array(repeating: nil, count: numLayers)
            self.layerVerified = Array(repeating: false, count: numLayers)
        }
    }

    init(device: MTLDevice,
         config: ArchConfig,
         streamingMode: ExpertStreamingMode,
         expertCachePolicy: ExpertCachePolicy,
         integrityPolicy: ModelIntegrityPolicy,
         residentBuffer: ResidentBuffer,
         residentIndex: ResidentIndex,
         packedExpertsLayout: PackedExpertsLayout,
         manifest: Manifest,
         directoryURL: URL) {
        self.device = device
        self.config = config
        self.streamingMode = streamingMode
        self.expertCachePolicy = expertCachePolicy
        self.integrityPolicy = integrityPolicy
        self.residentBuffer = residentBuffer
        self.residentIndex = residentIndex
        self.packedExpertsLayout = packedExpertsLayout
        self.manifest = manifest
        self.directoryURL = directoryURL
        self.streamersBox = StreamersBox(numLayers: packedExpertsLayout.numLayers)
        self.streamersQueue = DispatchQueue(label: "mference.expert-streamers")
        self.prefillSweepBypassesPageCache = Self.prefillSweepBypassesPageCache(
            family: config.family,
            streamingMode: streamingMode,
            layout: packedExpertsLayout,
            coreWeightsBytes: residentIndex.header.indexSize + residentIndex.header.residentSize)
    }

    /// Feeds the regime rule this install's sizes. Only `.pread` sweeps exist;
    /// `indexSize + residentSize` is `model_weights.bin`, which load checks.
    static func prefillSweepBypassesPageCache(
        family: ModelFamily,
        streamingMode: ExpertStreamingMode,
        layout: PackedExpertsLayout,
        coreWeightsBytes: UInt64,
        physicalMemoryBytes: UInt64 = ProcessInfo.processInfo.physicalMemory
    ) -> Bool {
        guard case .pread(let slotCount) = streamingMode else { return false }
        // File bytes, since those are what the page cache would hold.
        let poolBytes = layout.layers.reduce(UInt64(0)) { total, layer in
            total + UInt64(layer.experts.count) * layout.storedExpertStride
        }
        let routedLayers = UInt64(layout.layers.filter { !$0.experts.isEmpty }.count)
        let slotBytes = UInt64(PreadExpertStreamer.slotAllocationSize(
            expertStride: layout.expertStride))
        return RuntimeConfiguration.prefillSweepBypassesPageCache(
            for: family,
            physicalMemoryBytes: physicalMemoryBytes,
            expertPoolBytes: poolBytes,
            wiredSlotBytes: UInt64(slotCount) * routedLayers * slotBytes,
            coreWeightsBytes: coreWeightsBytes)
    }

    // MARK: - Resident accessors

    /// Tensor-name prefix for the language-model trunk. The Qwen conversions
    /// ship inside a multimodal container (`language_model.model.`); Maple
    /// and MiniCPM5 are text-only and use the bare `model.` prefix.
    var trunkPrefix: String {
        switch config.family {
        case .qwen36, .qwen38: return "language_model.model."
        case .maple: return "model."
        case .minicpm5: return "model."
        // Installed under the GGUF's own names (`blk.N.*`, `token_embd`).
        case .inkling2Loop: return ""
        }
    }
    private var lmHeadName: String {
        switch config.family {
        case .qwen36, .qwen38: return "language_model.lm_head.weight"
        case .maple: return "lm_head.weight"
        case .minicpm5: return "lm_head.weight"
        case .inkling2Loop: return Self.inkling2OutputHeadName
        }
    }
    /// Maple names the token embedding `word_embeddings`, not `embed_tokens`.
    private var embeddingName: String {
        switch config.family {
        case .qwen36, .qwen38:
            return "\(trunkPrefix)embed_tokens.weight"
        case .maple:
            return "\(trunkPrefix)word_embeddings.weight"
        case .minicpm5:
            return "\(trunkPrefix)embed_tokens.weight"
        case .inkling2Loop:
            return Self.inkling2EmbeddingName
        }
    }

    public var embedding: TensorView {
        try! resident(name: embeddingName)
    }

    /// A tied checkpoint reuses the embedding; Qwen 3.6 carries a separate
    /// `lm_head` tensor. The transpose for the lm_head GEMV path is the
    /// kernel's job, not the loader's.
    public var lmHead: TensorView {
        if config.tieWordEmbeddings { return embedding }
        return try! resident(name: lmHeadName)
    }

    public func qProj(layer L: Int) throws -> TensorView {
        return try resident(name: "\(trunkPrefix)layers.\(L).self_attn.q_proj.weight")
    }
    public func kProj(layer L: Int) throws -> TensorView {
        return try resident(name: "\(trunkPrefix)layers.\(L).self_attn.k_proj.weight")
    }
    public func vProj(layer L: Int) throws -> TensorView {
        return try resident(name: "\(trunkPrefix)layers.\(L).self_attn.v_proj.weight")
    }
    public func oProj(layer L: Int) throws -> TensorView {
        return try resident(name: "\(trunkPrefix)layers.\(L).self_attn.o_proj.weight")
    }
    /// The router is the source-named `.mlp.gate.weight`.
    public func router(layer L: Int) throws -> TensorView {
        switch config.family {
        case .qwen36, .qwen38:
            // Qwen 3.8 is dense and has no router tensor; the accessor throws
            // tensorNotFound if a caller ever asks.
            return try resident(name: "\(trunkPrefix)layers.\(L).mlp.gate.weight")
        case .maple:
            return try resident(name: "model.layers.\(L).mlp.gate.weight")
        case .minicpm5:
            // Dense: no router tensor exists; the accessor throws
            // tensorNotFound if a caller ever asks.
            return try resident(name: "\(trunkPrefix)layers.\(L).mlp.gate.weight")
        case .inkling2Loop:
            return try inkling2Router(layer: L)
        }
    }
    /// Shared-expert FFN. Qwen 3.6 and Maple keep the source's
    /// `.mlp.shared_expert.{gate,up,down}_proj.weight` names; the dense
    /// families' per-layer MLP is `.mlp.{gate,up,down}_proj.weight`.
    public func sharedExpertGate(layer L: Int) throws -> TensorView {
        try resident(name: sharedExpertName("gate_proj", layer: L))
    }
    public func sharedExpertUp(layer L: Int) throws -> TensorView {
        try resident(name: sharedExpertName("up_proj", layer: L))
    }
    public func sharedExpertDown(layer L: Int) throws -> TensorView {
        try resident(name: sharedExpertName("down_proj", layer: L))
    }
    private func sharedExpertName(_ proj: String, layer L: Int) -> String {
        switch config.family {
        case .qwen38:
            // For dense Qwen 3.8 these accessors serve the per-layer MLP.
            return "language_model.model.layers.\(L).mlp.\(proj).weight"
        case .qwen36:
            return "\(trunkPrefix)layers.\(L).mlp.shared_expert.\(proj).weight"
        case .maple:
            return "model.layers.\(L).mlp.shared_expert.\(proj).weight"
        case .minicpm5:
            // Dense llama: the per-layer MLP under the bare `.mlp.` names.
            return "\(trunkPrefix)layers.\(L).mlp.\(proj).weight"
        case .inkling2Loop:
            // No shared expert: llama.cpp's shared-expert name, which the
            // install never carries, so the accessor throws tensorNotFound.
            return Self.inkling2LayerName(L, "ffn_\(proj.replacingOccurrences(of: "_proj", with: ""))_shexp.weight")
        }
    }
    /// Qwen-only scalar gate on the shared-expert branch: a `[1, hidden]`
    /// 8-bit projection whose sigmoid multiplies the shared FFN output.
    public func sharedExpertScalarGate(layer L: Int) throws -> TensorView {
        return try resident(name: "\(trunkPrefix)layers.\(L).mlp.shared_expert_gate.weight")
    }
    /// Pre-attention norm.
    public func inputNorm(layer L: Int) throws -> TensorView {
        try resident(name: "\(trunkPrefix)layers.\(L).input_layernorm.weight")
    }
    /// Pre-FFN norm.
    public func postAttnNorm(layer L: Int) throws -> TensorView {
        try resident(name: "\(trunkPrefix)layers.\(L).post_attention_layernorm.weight")
    }
    /// Final trunk norm before `lm_head`.
    public var finalNorm: TensorView {
        try! resident(name: "\(trunkPrefix)norm.weight")
    }
    /// Whether the architecture carries a final trunk norm at all.
    public var hasFinalNorm: Bool { !config.hasLowRankHyperConnections }

    // MARK: - Per-head attention norms (Q/K only)
    //
    // `q_norm` and `k_norm` are RMSNorm with learnable scale, applied per head
    // before RoPE. `v_norm` has **no learnable weight** (no-scale RMSNorm) and
    // is therefore not stored as a tensor — the runtime uses an
    // explicit no-scale variant rather than consuming a unit-weight buffer.

    public func qNorm(layer L: Int) throws -> TensorView {
        try resident(name: "\(trunkPrefix)layers.\(L).self_attn.q_norm.weight")
    }
    public func kNorm(layer L: Int) throws -> TensorView {
        try resident(name: "\(trunkPrefix)layers.\(L).self_attn.k_norm.weight")
    }

    // MARK: - Feed-forward norms
    //
    // The Gemma 4 sandwich wraps two parallel FFN branches:
    //   pre_feedforward_layernorm        -> dense MLP input
    //   pre_feedforward_layernorm_2      -> routed expert input
    //   post_feedforward_layernorm_1     -> dense MLP output
    //   post_feedforward_layernorm_2     -> routed expert output
    //   post_feedforward_layernorm       -> combined (h1+h2) output

    public func preFFN(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).pre_feedforward_layernorm.weight")
    }
    public func preFFN2(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).pre_feedforward_layernorm_2.weight")
    }
    public func postFFN1(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).post_feedforward_layernorm_1.weight")
    }
    public func postFFN2(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).post_feedforward_layernorm_2.weight")
    }
    public func postFFN(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).post_feedforward_layernorm.weight")
    }

    // MARK: - Router auxiliaries
    //
    // `router.scale` is a per-feature multiplier on the router's input
    // (post-RMSNorm), fused with 1/sqrt(hidden_size). `per_expert_scale` is
    // applied to the top-k routing weights after softmax over top-k.

    public func routerScale(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).router.scale")
    }
    public func routerPerExpertScale(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).router.per_expert_scale")
    }

    /// Per-layer scalar gain applied to the entire residual stream at the end
    /// of the layer; shape `[1]`, BF16.
    public func layerScalar(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).layer_scalar")
    }

    // MARK: - Gated-DeltaNet linear attention (Qwen 3.6)
    //
    // Layers whose mask value is 2 replace full/sliding attention with the
    // gated delta rule. Projections are 4-bit affine; the depthwise conv
    // weight, A_log, dt_bias, and the gated output norm are BF16.

    public func linearInProjQKV(layer L: Int) throws -> TensorView {
        return try resident(name: "\(trunkPrefix)layers.\(L).linear_attn.in_proj_qkv.weight")
    }
    public func linearInProjZ(layer L: Int) throws -> TensorView {
        return try resident(name: "\(trunkPrefix)layers.\(L).linear_attn.in_proj_z.weight")
    }
    public func linearInProjA(layer L: Int) throws -> TensorView {
        return try resident(name: "\(trunkPrefix)layers.\(L).linear_attn.in_proj_a.weight")
    }
    public func linearInProjB(layer L: Int) throws -> TensorView {
        return try resident(name: "\(trunkPrefix)layers.\(L).linear_attn.in_proj_b.weight")
    }
    public func linearOutProj(layer L: Int) throws -> TensorView {
        return try resident(name: "\(trunkPrefix)layers.\(L).linear_attn.out_proj.weight")
    }
    /// Depthwise causal conv weight, source shape `[convDim, kernel, 1]`, BF16.
    public func linearConv1d(layer L: Int) throws -> TensorView {
        return try resident(name: "\(trunkPrefix)layers.\(L).linear_attn.conv1d.weight")
    }
    /// Per-value-head decay base, shape `[numVHeads]`, BF16.
    public func linearALog(layer L: Int) throws -> TensorView {
        return try resident(name: "\(trunkPrefix)layers.\(L).linear_attn.A_log")
    }
    /// Per-value-head dt bias, shape `[numVHeads]`, BF16.
    public func linearDtBias(layer L: Int) throws -> TensorView {
        return try resident(name: "\(trunkPrefix)layers.\(L).linear_attn.dt_bias")
    }
    /// Gated RMSNorm weight over the value head dim, shape `[valueHeadDim]`.
    public func linearNorm(layer L: Int) throws -> TensorView {
        return try resident(name: "\(trunkPrefix)layers.\(L).linear_attn.norm.weight")
    }

    /// Resolve a resident tensor, widening BF16 to a cached FP32 copy when
    /// the source dtype is BF16. Intended for small tables only (the copy
    /// lives in a dedicated shared-storage buffer).
    func residentAsF32(name: String) throws -> TensorView {
        if let cached = streamersQueue.sync(execute: { convertedBox.views[name] }) {
            return cached
        }
        let src = try resident(name: name)
        guard src.dtype == 1 else {
            streamersQueue.sync { convertedBox.views[name] = src }
            return src
        }
        let count = Int(src.length) / 2
        guard let buf = device.makeBuffer(
            length: count * MemoryLayout<Float>.stride,
            options: .storageModeShared) else {
            throw ModelError.residentBufferWrapFailed
        }
        let srcPtr = src.buffer.contents().advanced(by: Int(src.offset))
            .assumingMemoryBound(to: UInt16.self)
        let dstPtr = buf.contents().assumingMemoryBound(to: Float.self)
        for i in 0..<count { dstPtr[i] = Quantization.bf16ToFloat(srcPtr[i]) }
        let view = TensorView(
            buffer: buf, offset: 0, length: UInt64(count * 4),
            scaleOffset: 0, scaleLength: 0,
            biasOffset: 0, biasLength: 0,
            shape: src.shape, dtype: 3)
        streamersQueue.sync { convertedBox.views[name] = view }
        return view
    }

    /// Resolve a tensor name to a `TensorView` against the resident buffer.
    /// `fileOffset` (absolute) is converted to a buffer-relative offset by
    /// subtracting the resident region's file offset (which equals
    /// `header.indexSize`).
    func resident(name: String) throws -> TensorView {
        guard let entry = residentIndex.entries[name] else {
            throw ModelError.tensorNotFound(name: name)
        }
        let residentFileOffset = residentIndex.header.indexSize
        guard entry.fileOffset >= residentFileOffset,
              entry.scaleSize == 0 || entry.scaleOffset >= residentFileOffset,
              entry.biasSize == 0 || entry.biasOffset >= residentFileOffset else {
            throw ModelError.indexCorrupt(
                detail: "resident tensor \(name) has an offset before the resident region")
        }
        let relativeOffset = entry.fileOffset - residentFileOffset
        let scaleRel: UInt64 = entry.scaleSize > 0
            ? entry.scaleOffset - residentFileOffset : 0
        let biasRel: UInt64 = entry.biasSize > 0
            ? entry.biasOffset - residentFileOffset : 0
        // Resolve the chunk holding the tensor's full span; offsets in the
        // returned view are relative to that chunk's buffer.
        var spanLo = relativeOffset
        var spanHi = relativeOffset + entry.sizeBytes
        if entry.scaleSize > 0 {
            spanLo = min(spanLo, scaleRel)
            spanHi = max(spanHi, scaleRel + entry.scaleSize)
        }
        if entry.biasSize > 0 {
            spanLo = min(spanLo, biasRel)
            spanHi = max(spanHi, biasRel + entry.biasSize)
        }
        guard let chunk = residentBuffer.chunk(containing: spanLo, spanHi) else {
            throw ModelError.indexCorrupt(
                detail: "resident tensor \(name) straddles a chunk boundary")
        }
        return TensorView(
            buffer: chunk.buffer,
            offset: relativeOffset - chunk.start + chunk.bufferOffset,
            length: entry.sizeBytes,
            scaleOffset: entry.scaleSize > 0 ? scaleRel - chunk.start + chunk.bufferOffset : 0,
            scaleLength: entry.scaleSize,
            biasOffset:  entry.biasSize > 0 ? biasRel - chunk.start + chunk.bufferOffset : 0,
            biasLength:  entry.biasSize,
            shape: entry.shape,
            dtype: entry.dtype)
    }

    // MARK: - Routed expert (lazy)

    /// First touch of layer L opens its backend + verifies SHA-256; subsequent
    /// touches reuse the open backend. The backend resolves the expert to an
    /// cache-slot `(MTLBuffer, offset)` pair.
    public func routedExpert(layer L: Int, expert E: Int) throws -> TensorView {
        try ensureLayerOpened(L)
        let backend = streamersQueue.sync { streamersBox.streamers[L]! }
        let r = try backend.loadExpert(layer: 0, expert: E)
        return TensorView(
            buffer: r.buffer,
            offset: r.offset,
            length: r.size,
            scaleOffset: 0, scaleLength: 0,
            biasOffset:  0, biasLength:  0,
            shape: (UInt32(L), UInt32(E), 0, 0),
            dtype: 0)
    }

    /// Open layer L's file + verify SHA, idempotent.
    func ensureLayerOpened(_ L: Int) throws {
        try streamersQueue.sync {
            try openLayerLocked(L)
        }
    }

    /// Best-effort overlap hook for prefill: starts the same lazy layer open on
    /// the model's streamer queue without waiting for the first expert fetch.
    public func beginOpeningRoutedExpertStreamer(layer L: Int) {
        nonisolated(unsafe) let model = self
        streamersQueue.async {
            try? model.openLayerLocked(L)
        }
    }

    private func openLayerLocked(_ L: Int) throws {
        if streamersBox.streamers[L] != nil {
            return
        }
        // Leading dense-FFN layers have no routed experts and no blob file.
        if packedExpertsLayout.layers[L].experts.isEmpty { return }
        let basename = packedExpertsLayout.layers[L].file
        let url = directoryURL
            .appendingPathComponent("packed_experts")
            .appendingPathComponent(basename)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw ModelError.missingFile(name: "packed_experts/\(basename)")
        }
        if !streamersBox.layerVerified[L] {
            let manifestRel = "packed_experts/\(basename)"
            guard let entry = manifest.files[manifestRel] else {
                throw ModelError.missingFile(name: manifestRel)
            }
            switch integrityPolicy {
            case .fullSha256, .trustedReceiptWhenValid:
                try Sha256Verifier.verifyFile(at: url, named: manifestRel,
                                              expectedHex: entry.sha256)
            case .sizeCheckTrustedReceipt:
                try Self.verifyTrustedReceiptFileSize(url: url,
                                                      relativePath: manifestRel,
                                                      expectedSize: entry.size)
            }
            streamersBox.layerVerified[L] = true
        }
        let streamSize = UInt64(packedExpertsLayout.expertsPerLayer)
            * packedExpertsLayout.storedExpertStride
        let layout = StreamLayout(
            path: url.path,
            streamOffset: 0,
            streamSize: streamSize,
            expertsPerLayer: packedExpertsLayout.expertsPerLayer,
            expertStride: packedExpertsLayout.expertStride,
            expertOffsets: packedExpertsLayout.layers[L].experts.map(\.offset),
            storage: packedExpertsLayout.storage)
        switch streamingMode {
        case .pread(let slotCount):
            streamersBox.streamers[L] = .pread(try PreadExpertStreamer(
                layout: layout,
                device: device,
                slotCount: slotCount,
                cachePolicy: expertCachePolicy,
                unbufferedSweep: prefillSweepBypassesPageCache))
        case .resident:
            streamersBox.streamers[L] = .resident(try ResidentExpertStreamer(
                layout: layout,
                device: device,
                strategy: Self.residentStrategy(for: config.family)))
        }
    }

    /// How `.resident` holds a layer's experts: every kept family maps it
    /// (see `ResidentExpertStreamer`; a layout with `storage` still copies).
    static func residentStrategy(for family: ModelFamily) -> ResidentExpertStreamer.Strategy {
        .mapped
    }

    /// Resident mode: open every routed layer and touch its mapping so page-in
    /// cost lands in model load, not the first prefill or decode step.
    func warmUpResidentBackends() throws {
        guard case .resident = streamingMode else { return }
        for L in 0..<packedExpertsLayout.numLayers {
            if packedExpertsLayout.layers[L].experts.isEmpty { continue }
            try ensureLayerOpened(L)
            let backend = streamersQueue.sync { streamersBox.streamers[L]! }
            if case .resident(let streamer) = backend {
                streamer.warmUp()
            }
        }
    }

    /// Test hook: how many layer files have been opened so far.
    public func openLayerFileCount() -> Int {
        streamersQueue.sync { streamersBox.streamers.compactMap { $0 }.count }
    }

}

extension Model {

    /// Open a `.gturbo/` directory with the architecture auto-detected from
    /// `manifest.json -> arch.family`.
    public static func load(directoryURL: URL,
                            device: MTLDevice,
                            streamingMode: ExpertStreamingMode = .pread(slotCount: 16),
                            expertCachePolicy: ExpertCachePolicy = PreadExpertStreamer.cachePolicyDefault,
                            integrityPolicy: ModelIntegrityPolicy? = nil,
                            loadStats: UnsafeMutablePointer<ModelLoadStats>? = nil) throws -> Model {
        let family = try ManifestReader.peekFamily(directoryURL: directoryURL)
        guard let baseline = ArchConfig.knownArchitectures[family] else {
            throw ModelError.indexCorrupt(detail: "no baseline for family \(family.rawValue)")
        }
        return try load(directoryURL: directoryURL,
                        device: device,
                        expecting: baseline,
                        streamingMode: streamingMode,
                        expertCachePolicy: expertCachePolicy,
                        integrityPolicy: integrityPolicy,
                        loadStats: loadStats)
    }

    /// Open a `.gturbo/` directory and return a typed handle. Eagerly verifies
    /// SHA-256 of `model_weights.bin` and `packed_experts/layout.json`; layer
    /// files are verified lazily on first `routedExpert(...)` touch.
    public static func load(directoryURL: URL,
                            device: MTLDevice,
                            expecting: ArchConfig,
                            streamingMode: ExpertStreamingMode = .pread(slotCount: 16),
                            expertCachePolicy: ExpertCachePolicy = PreadExpertStreamer.cachePolicyDefault,
                            integrityPolicy: ModelIntegrityPolicy? = nil,
                            loadStats: UnsafeMutablePointer<ModelLoadStats>? = nil) throws -> Model {
        var stats = ModelLoadStats()
        defer {
            loadStats?.pointee = stats
        }
        let requestedIntegrityPolicy = integrityPolicy ?? .fullSha256
        // `.trustedReceiptWhenValid` starts on the receipt path and settles on
        // full SHA-256 as soon as a receipt check fails, so an install without
        // a usable receipt loads exactly as it does under `.fullSha256`.
        var resolvedIntegrityPolicy: ModelIntegrityPolicy =
            requestedIntegrityPolicy == .trustedReceiptWhenValid ? .sizeCheckTrustedReceipt : requestedIntegrityPolicy
        var fallbackDetail: String?
        func receiptCheck(_ check: () throws -> Void) throws {
            guard resolvedIntegrityPolicy == .sizeCheckTrustedReceipt else { return }
            let start = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            defer { stats.receiptValidationNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - start }
            do {
                try check()
            } catch ModelError.trustedReceiptInvalid(let detail)
                        where requestedIntegrityPolicy == .trustedReceiptWhenValid {
                resolvedIntegrityPolicy = .fullSha256
                fallbackDetail = detail
            }
        }

        let manifestURL = directoryURL.appendingPathComponent("manifest.json")
        guard FileManager.default.fileExists(atPath: manifestURL.path) else {
            throw ModelError.partialInstall(path: directoryURL.path)
        }
        let manifestShaStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        let manifestSha = try Sha256Verifier.hashFile(at: manifestURL)
        stats.manifestSha256Nanos = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - manifestShaStart
        let manifestSize = try Self.fileSize(at: manifestURL,
                                             relativePath: "manifest.json")
        var receipt: VerifiedInstallReceipt?
        try receiptCheck {
            let loadedReceipt = try VerifiedInstallReceiptReader.load(directoryURL: directoryURL)
            try VerifiedInstallReceiptReader.validateManifestBinding(
                loadedReceipt,
                directoryURL: directoryURL,
                manifestSha256: manifestSha)
            receipt = loadedReceipt
        }

        let manifest = try ManifestReader.load(directoryURL: directoryURL,
                                               expecting: expecting)
        if let receipt {
            try receiptCheck {
                try VerifiedInstallReceiptReader.validate(receipt,
                                                          directoryURL: directoryURL,
                                                          manifest: manifest,
                                                          manifestSha256: manifestSha,
                                                          manifestSize: manifestSize)
            }
        }

        // Verify the small, always-touched files before mapping model data.
        let weightsURL = directoryURL.appendingPathComponent("model_weights.bin")
        let layoutURL  = directoryURL
            .appendingPathComponent("packed_experts")
            .appendingPathComponent("layout.json")
        guard let weightsEntry = manifest.files["model_weights.bin"] else {
            throw ModelError.missingFile(name: "model_weights.bin")
        }
        guard let layoutEntry = manifest.files["packed_experts/layout.json"] else {
            throw ModelError.missingFile(name: "packed_experts/layout.json")
        }
        let eagerShaStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        try Sha256Verifier.verifyFile(at: weightsURL, named: "model_weights.bin",
                                      expectedHex: weightsEntry.sha256)
        try Sha256Verifier.verifyFile(at: layoutURL, named: "packed_experts/layout.json",
                                      expectedHex: layoutEntry.sha256)
        stats.eagerSha256Nanos = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - eagerShaStart

        let residentIndex = try ResidentIndexReader.load(fileURL: weightsURL)
        let layout = try PackedExpertsLayoutReader.load(directoryURL: directoryURL)

        // The resident index must account for the complete weights file.
        let attrs = try FileManager.default.attributesOfItem(atPath: weightsURL.path)
        if let fileSize = attrs[.size] as? UInt64 {
            let expected = residentIndex.header.indexSize + residentIndex.header.residentSize
            if fileSize != expected {
                throw ModelError.indexCorrupt(detail: """
                    model_weights.bin size \(fileSize) != indexSize \
                    \(residentIndex.header.indexSize) + residentSize \
                    \(residentIndex.header.residentSize) = \(expected)
                    """)
            }
        }

        // Region-relative spans (tensor + companions) so an oversized region
        // can be cut into multiple buffers without splitting any tensor.
        let regionStart = residentIndex.header.indexSize
        let tensorSpans = residentIndex.entries.values
            .filter { $0.fileOffset >= regionStart }
            .map { entry -> (start: UInt64, end: UInt64) in
                var lo = entry.fileOffset
                var hi = entry.fileOffset + entry.sizeBytes
                if entry.scaleSize > 0 {
                    lo = min(lo, entry.scaleOffset)
                    hi = max(hi, entry.scaleOffset + entry.scaleSize)
                }
                if entry.biasSize > 0 {
                    lo = min(lo, entry.biasOffset)
                    hi = max(hi, entry.biasOffset + entry.biasSize)
                }
                return (lo - regionStart, hi - regionStart)
            }
        let residentBuffer = try ResidentBuffer(
            fileURL: weightsURL,
            fileOffset: residentIndex.header.indexSize,
            residentSize: residentIndex.header.residentSize,
            device: device,
            tensorSpans: tensorSpans)

        try receiptCheck {
            try validateTrustedReceiptLayerLayout(directoryURL: directoryURL,
                                                  manifest: manifest,
                                                  layout: layout)
        }

        var model = Model(
            device: device,
            config: expecting,
            streamingMode: streamingMode,
            expertCachePolicy: expertCachePolicy,
            integrityPolicy: resolvedIntegrityPolicy,
            residentBuffer: residentBuffer,
            residentIndex: residentIndex,
            packedExpertsLayout: layout,
            manifest: manifest,
            directoryURL: directoryURL)
        model.integrityFallbackDetail = fallbackDetail
        if case .resident = streamingMode {
            let warmupStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            try model.warmUpResidentBackends()
            stats.residentWarmupNanos =
                clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - warmupStart
        }
        return model
    }

    private static func verifyTrustedReceiptFileSize(url: URL,
                                                     relativePath: String,
                                                     expectedSize: UInt64) throws {
        let actualSize = try fileSize(at: url, relativePath: relativePath)
        guard actualSize == expectedSize else {
            throw ModelError.trustedReceiptInvalid(
                detail: "\(relativePath) size \(actualSize) != \(expectedSize)")
        }
    }

    private static func fileSize(at url: URL, relativePath: String) throws -> UInt64 {
        let attrs = try FileManager.default.attributesOfItem(atPath: url.path)
        guard let sizeValue = attrs[.size] else {
            throw ModelError.trustedReceiptInvalid(detail: "missing size for \(relativePath)")
        }
        let actualSize: UInt64
        if let number = sizeValue as? NSNumber {
            actualSize = number.uint64Value
        } else if let value = sizeValue as? UInt64 {
            actualSize = value
        } else if let value = sizeValue as? Int {
            actualSize = UInt64(value)
        } else {
            throw ModelError.trustedReceiptInvalid(detail: "invalid size for \(relativePath)")
        }
        return actualSize
    }

    private static func validateTrustedReceiptLayerLayout(directoryURL: URL,
                                                          manifest: Manifest,
                                                          layout: PackedExpertsLayout) throws {
        let pageSize = UInt64(getpagesize())
        guard layout.expertStride % pageSize == 0,
              layout.storedExpertStride % pageSize == 0 else {
            throw ModelError.trustedReceiptInvalid(
                detail: "expertStride \(layout.expertStride) or stored stride "
                    + "\(layout.storedExpertStride) is not page-aligned")
        }
        guard layout.numLayers == manifest.numLayers,
              layout.expertsPerLayer == manifest.expertsPerLayer,
              layout.expertStride == manifest.expertStride else {
            throw ModelError.trustedReceiptInvalid(detail: "layout does not match manifest dimensions")
        }
        for layer in layout.layers {
            guard layer.layer >= 0 && layer.layer < manifest.numLayers else {
                throw ModelError.trustedReceiptInvalid(detail: "layout layer out of range")
            }
            // Dense-FFN layers ship no expert blob; nothing to size-check.
            if layer.experts.isEmpty { continue }
            let relativePath = "packed_experts/\(layer.file)"
            guard let manifestEntry = manifest.files[relativePath] else {
                throw ModelError.trustedReceiptInvalid(detail: "manifest missing \(relativePath)")
            }
            let storedStride = layout.storedExpertStride
            guard storedStride % pageSize == 0 else {
                throw ModelError.trustedReceiptInvalid(
                    detail: "\(relativePath) stored stride \(storedStride) is not page-aligned")
            }
            let expectedSize = UInt64(layout.expertsPerLayer) * storedStride
            guard manifestEntry.size == expectedSize else {
                throw ModelError.trustedReceiptInvalid(
                    detail: "\(relativePath) manifest size \(manifestEntry.size) != \(expectedSize)")
            }
            let url = directoryURL
                .appendingPathComponent("packed_experts")
                .appendingPathComponent(layer.file)
            let actualSize = try fileSize(at: url, relativePath: relativePath)
            guard actualSize == expectedSize else {
                throw ModelError.trustedReceiptInvalid(
                    detail: "\(relativePath) size \(actualSize) != \(expectedSize)")
            }
            guard layer.experts.count == layout.expertsPerLayer else {
                throw ModelError.trustedReceiptInvalid(detail: "\(relativePath) expert count mismatch")
            }
            for expert in layer.experts {
                guard expert.size == storedStride else {
                    throw ModelError.trustedReceiptInvalid(
                        detail: "\(relativePath) expert \(expert.expert) size mismatch")
                }
                guard expert.offset % pageSize == 0 else {
                    throw ModelError.trustedReceiptInvalid(
                        detail: "\(relativePath) expert \(expert.expert) offset is not page-aligned")
                }
                guard expert.offset <= actualSize,
                      expert.size <= actualSize - expert.offset else {
                    throw ModelError.trustedReceiptInvalid(
                        detail: "\(relativePath) expert \(expert.expert) range exceeds file size")
                }
            }
        }
    }
}

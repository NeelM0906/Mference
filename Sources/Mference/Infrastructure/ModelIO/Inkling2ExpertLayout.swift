import Foundation

/// The routed experts of an `inkling2Loop` install, validated against the
/// family baseline: what the expert kernels read, layer by layer.
///
/// One blob per expert in `packed_experts/layer_LL.bin`, at `expert *
/// expertStride` of its layer. A blob holds two projections, each the GGUF's
/// own GGML block rows, copied unchanged:
///
/// - `gateUp`: `[2 * width, hidden]` (rows `0 ..< width` are the gate, rows
///   `width ..< 2 * width` the up projection), at blob offset 0;
/// - `down`: `[hidden, width]`, at the next 256-byte boundary.
///
/// A row of `n` values is `n / blockElements` blocks back to back
/// (`GGMLBlockFormat`). The blob is padded to the 16 KiB page, so the stride
/// is the layer's own: 4,489,216 bytes for NVFP4 and 8,470,528 for Q8_0. An
/// NVFP4 projection's output is further multiplied by its expert's F32 factor,
/// which stays resident (`Model.inkling2ExpertGateUpScale(layer:)`, `...Down`).
public struct Inkling2ExpertLayout: Sendable, Equatable {
    public struct Projection: Sendable, Equatable {
        public let format: GGMLBlockFormat
        /// Output rows.
        public let rows: Int
        /// Input values per row.
        public let columns: Int
        /// Byte offset inside an expert blob.
        public let offset: UInt64
        /// Bytes: `rows * rowBytes`.
        public let size: UInt64
        /// Bytes of one row of `columns` values.
        public let rowBytes: Int
    }

    public struct Layer: Sendable, Equatable {
        public let format: GGMLBlockFormat
        /// Bytes from one expert's blob to the next in the layer file.
        public let expertStride: UInt64
        public let gateUp: Projection
        public let down: Projection
    }

    /// One entry per stored layer, indexed by layer.
    public let layers: [Layer]

    /// Offset alignment of the second projection inside a blob.
    public static let projectionAlignment: UInt64 = 256
    /// The page the installer pads every blob to, whatever the host's page.
    public static let blobPage: UInt64 = 16_384

    /// The stride the installer gives one expert blob of `format`: gate-up at
    /// 0, down at the next 256-byte boundary, padded to `blobPage`
    /// (4,489,216 bytes for NVFP4 and 8,470,528 for Q8_0 at the shipped
    /// shape). Nil for a format without whole block rows at this shape.
    public static func blobStride(format: GGMLBlockFormat, config: ArchConfig) -> UInt64? {
        let hidden = config.hiddenSize, width = config.moeIntermediateSize
        guard let gateUpRow = format.rowBytes(elements: hidden),
              let downRow = format.rowBytes(elements: width) else { return nil }
        let gateUp = UInt64(2 * width * gateUpRow)
        let downStart = (gateUp + projectionAlignment - 1) / projectionAlignment * projectionAlignment
        let end = downStart + UInt64(hidden * downRow)
        return (end + blobPage - 1) / blobPage * blobPage
    }

    /// Builds the view and proves it: every layer present with all experts at
    /// its own page-aligned stride, its format equal to
    /// `config.inkling2.routedExpertFormats`, both projections in that format
    /// with the baseline's shapes, sizes that are whole block rows, and no
    /// byte outside the stride.
    static func validated(layout: PackedExpertsLayout, config: ArchConfig,
                          pageSize: UInt64 = UInt64(getpagesize())) throws -> Inkling2ExpertLayout {
        func invalid(_ detail: String) -> ModelError {
            .indexCorrupt(detail: "inkling2 experts: " + detail)
        }
        let formats = config.inkling2.routedExpertFormats
        guard config.family == .inkling2Loop, formats.count == config.numLayers else {
            throw invalid("the baseline names no per-layer expert formats")
        }
        guard layout.numLayers == config.numLayers, layout.layers.count == config.numLayers,
              layout.expertsPerLayer == config.numExperts else {
            throw invalid("layout has \(layout.layers.count) layers of \(layout.expertsPerLayer) experts")
        }
        let hidden = config.hiddenSize, width = config.moeIntermediateSize
        var out: [Layer] = []
        out.reserveCapacity(config.numLayers)
        for (index, entry) in layout.layers.enumerated() {
            guard entry.layer == index, entry.experts.count == config.numExperts else {
                throw invalid("layer \(index) is not a full layer")
            }
            guard let stride = entry.expertStride, stride > 0, stride % pageSize == 0 else {
                throw invalid("layer \(index) has no page-aligned stride of its own")
            }
            guard let format = GGMLBlockFormat(name: formats[index]),
                  format == .nvfp4 || format == .q8_0,
                  entry.format == nil || entry.format == format.name else {
                throw invalid("layer \(index) format \(entry.format ?? "nil") is not \(formats[index])")
            }
            let tensors = entry.experts[0].subTensors
            guard Set(tensors.keys) == ["gate_up", "down"] else {
                throw invalid("layer \(index) needs exactly gate_up and down, has \(tensors.keys.sorted())")
            }
            func projection(_ role: String, rows: Int, columns: Int) throws -> Projection {
                let t = tensors[role]!
                guard t.dtype == "GGML", t.format == format.name, t.shape == [rows, columns],
                      let rowBytes = format.rowBytes(elements: columns),
                      t.size == UInt64(rows * rowBytes),
                      t.offset <= stride, t.size <= stride - t.offset else {
                    throw invalid("layer \(index) \(role) is not \(format.name) [\(rows), \(columns)] "
                        + "inside its stride")
                }
                return Projection(format: format, rows: rows, columns: columns,
                                  offset: t.offset, size: t.size, rowBytes: rowBytes)
            }
            let gateUp = try projection("gate_up", rows: 2 * width, columns: hidden)
            let down = try projection("down", rows: hidden, columns: width)
            let downStart = (gateUp.size + Self.projectionAlignment - 1)
                / Self.projectionAlignment * Self.projectionAlignment
            guard gateUp.offset == 0, down.offset == downStart else {
                throw invalid("layer \(index) projections are not at 0 and \(downStart)")
            }
            for expert in entry.experts where expert.subTensors != tensors {
                throw invalid("layer \(index) expert \(expert.expert) differs from expert 0")
            }
            out.append(Layer(format: format, expertStride: stride, gateUp: gateUp, down: down))
        }
        return Inkling2ExpertLayout(layers: out)
    }
}

extension Model {
    /// The routed-expert layout of an `inkling2Loop` install, validated
    /// against the baseline once at load; validated now for a model built
    /// another way.
    public func inkling2ExpertLayout() throws -> Inkling2ExpertLayout {
        if let validatedInkling2Experts { return validatedInkling2Experts }
        return try Inkling2ExpertLayout.validated(layout: packedExpertsLayout, config: config)
    }
}

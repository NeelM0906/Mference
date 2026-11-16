import Foundation

/// Routed experts stored as native GGUF blocks and executed as LUT4 slots
/// (WS4 U2 R2, `layout.json -> expertStorage.format == "ggufNativeBlocksV1"`).
///
/// The layer files hold each expert's three projections as the GGUF's own
/// block bytes, at a per-layer stored stride. A slot holds the LUT4 expansion
/// (U1 §3a), which is larger than every stored expert, so slots are sized from
/// the per-layer `slotStrides` and never from the top-level `expertStride`
/// (the largest stored stride).
///
/// Slot sections, per projection in gate, up, down order: the packed 4-bit
/// indices (`n / 2` bytes, IQ4_NL nibble order per 32 weights), the fp16 `D`
/// (`2 n / bd`), then the int8 `M` (`n / bm`, absent when `bm == 0`). Every
/// section starts 16-byte aligned.
public struct GGUFExpertLayout: Sendable, Equatable {
    public static let format = "ggufNativeBlocksV1"
    public static let slotFormat = "lut4"

    public struct Projection: Sendable, Equatable {
        public let type: GGUFLUT4.SourceType
        /// Rows x columns of the projection, as `[rows, columns]`.
        public let rows: Int
        public let columns: Int
        /// Byte offset and size of the GGUF blocks inside a stored expert.
        public let storedOffset: UInt64
        public let storedSize: UInt64
        /// LUT4 section offsets inside a slot.
        public let indexOffset: UInt64
        public let dOffset: UInt64
        public let mOffset: UInt64

        public var weights: Int { rows * columns }
        public var blocks: Int { weights / type.blockWeights }
    }

    public struct Layer: Sendable, Equatable {
        public let storedStride: UInt64
        public let slotStride: UInt64
        public let gate: Projection
        public let up: Projection
        public let down: Projection
        /// Bytes the LUT4 sections occupy (at most `slotStride`).
        public let slotBytes: UInt64

        public var projections: [Projection] { [gate, up, down] }
    }

    /// One entry per layer, indexed by layer.
    public let layers: [Layer]

    static func sourceType(named name: String) -> GGUFLUT4.SourceType? {
        switch name {
        case "IQ3_XXS": .iq3_xxs
        case "IQ4_NL": .iq4_nl
        case "IQ2_S": .iq2_s
        case "IQ4_XS": .iq4_xs
        case "Q2_0": .q2_0
        default: nil
        }
    }

    /// LUT4 section bytes of one projection: indices, `D`, `M`.
    static func sectionBytes(_ type: GGUFLUT4.SourceType, weights n: Int) -> (index: Int, d: Int, m: Int) {
        let p = GGUFLUT4.parameters(type)
        return (n / 2, 2 * (n / p.bd), p.bm == 0 ? 0 : n / p.bm)
    }

    @inline(__always) static func align16(_ n: UInt64) -> UInt64 { (n + 15) / 16 * 16 }

    /// Parses `expertStorage` (format `ggufNativeBlocksV1`) against the layer
    /// entries and proves every expert matches its layer's single block layout.
    static func parse(_ obj: [String: Any], numLayers: Int, expertsPerLayer: Int,
                      layers: [LayerLayout]) throws -> GGUFExpertLayout {
        func invalid(_ detail: String) -> ModelError {
            .indexCorrupt(detail: "layout.json: expertStorage (\(format)) " + detail)
        }
        guard obj["format"] as? String == format else { throw invalid("has another format") }
        guard obj["slotFormat"] as? String == slotFormat else {
            throw invalid("needs slotFormat \"\(slotFormat)\"")
        }
        func strides(_ key: String) throws -> [UInt64] {
            guard let values = obj[key] as? [NSNumber], values.count == numLayers,
                  values.allSatisfy({ $0.int64Value > 0 }) else {
                throw invalid("needs \(numLayers) positive \(key)")
            }
            return values.map(\.uint64Value)
        }
        let stored = try strides("storedExpertStrides")
        let slots = try strides("slotStrides")
        guard layers.count == numLayers else { throw invalid("has \(layers.count) layers, expected \(numLayers)") }

        var out: [Layer] = []
        out.reserveCapacity(numLayers)
        for (L, layer) in layers.enumerated() {
            guard layer.layer == L, layer.experts.count == expertsPerLayer else {
                throw invalid("layer \(L) is not a full layer of \(expertsPerLayer) experts")
            }
            let first = layer.experts[0].subTensors
            guard Set(first.keys) == ["gate", "up", "down"] else {
                throw invalid("layer \(L) needs exactly gate, up and down")
            }
            for (e, expert) in layer.experts.enumerated() {
                guard expert.expert == e, expert.size == stored[L],
                      expert.offset == UInt64(e) * stored[L], expert.subTensors == first else {
                    throw invalid("layer \(L) expert \(e) is not at its fixed stride")
                }
            }
            var cursor: UInt64 = 0
            func projection(_ role: String) throws -> Projection {
                let t = first[role]!
                guard t.dtype == "GGML", let shape = t.shape, shape.count == 2,
                      shape[0] > 0, shape[1] > 0 else {
                    throw invalid("layer \(L) \(role) is not a 2-D GGML tensor")
                }
                let n = shape[0] * shape[1]
                guard let name = t.format, let type = sourceType(named: name) else {
                    throw invalid("layer \(L) \(role) has no LUT4 format")
                }
                let p = GGUFLUT4.parameters(type)
                guard shape[1].isMultiple(of: type.blockWeights), n.isMultiple(of: 32),
                      p.bm == 0 || n.isMultiple(of: p.bm),
                      t.size == UInt64(n / type.blockWeights * type.blockBytes),
                      t.offset + t.size <= stored[L] else {
                    throw invalid("layer \(L) \(role) has blocks that do not tile its shape")
                }
                let s = sectionBytes(type, weights: n)
                let index = align16(cursor)
                let d = align16(index + UInt64(s.index))
                let m = align16(d + UInt64(s.d))
                cursor = m + UInt64(s.m)
                return Projection(type: type, rows: shape[0], columns: shape[1],
                                  storedOffset: t.offset, storedSize: t.size,
                                  indexOffset: index, dOffset: d, mOffset: m)
            }
            let gate = try projection("gate")
            let up = try projection("up")
            let down = try projection("down")
            guard gate.type == up.type, gate.rows == up.rows, gate.columns == up.columns,
                  down.rows == gate.columns, down.columns == gate.rows else {
                throw invalid("layer \(L) gate/up/down shapes or types disagree")
            }
            guard cursor <= slots[L] else {
                throw invalid("layer \(L) LUT4 sections need \(cursor) bytes, slot stride is \(slots[L])")
            }
            out.append(Layer(storedStride: stored[L], slotStride: slots[L],
                             gate: gate, up: up, down: down, slotBytes: cursor))
        }
        return GGUFExpertLayout(layers: out)
    }
}

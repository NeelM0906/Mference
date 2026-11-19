import Metal

extension Native {
    /// ggml tensor types. Raw values are ggml's `enum ggml_type` at the pinned
    /// llama.cpp (`Sources/LlamaCpp`, b1b770d9) and are what GGUF stores.
    public enum GGMLType: UInt32, Codable, Sendable, CaseIterable {
        case f32 = 0
        case f16 = 1
        case q4_0 = 2
        case q4_1 = 3
        case q5_0 = 6
        case q5_1 = 7
        case q8_0 = 8
        case q8_1 = 9
        case q2_K = 10
        case q3_K = 11
        case q4_K = 12
        case q5_K = 13
        case q6_K = 14
        case q8_K = 15
        case iq2_xxs = 16
        case iq2_xs = 17
        case iq3_xxs = 18
        case iq1_s = 19
        case iq4_nl = 20
        case iq3_s = 21
        case iq2_s = 22
        case iq4_xs = 23
        case i8 = 24
        case i16 = 25
        case i32 = 26
        case i64 = 27
        case f64 = 28
        case iq1_m = 29
        case bf16 = 30
    }

    /// One field of a quant block, in ggml struct order (`ggml-common.h`).
    public struct BlockField: Equatable, Sendable {
        public let name: String
        public let bytes: Int
        public init(name: String, bytes: Int) {
            self.name = name
            self.bytes = bytes
        }
    }

    /// Block geometry of a ggml type: `elements` weights per block stored in
    /// `bytes` bytes, made of `fields` in struct order (their sizes sum to
    /// `bytes`). Unquantized types are one-element blocks with one field.
    ///
    /// Normative for the types Quick and Deep use; `TileLayout.planes` splits
    /// blocks along these fields.
    public struct QuantBlockFormat: Equatable, Sendable {
        public let type: GGMLType
        public let elements: Int
        public let bytes: Int
        public let fields: [BlockField]

        public init(type: GGMLType, elements: Int, bytes: Int, fields: [BlockField]) {
            self.type = type
            self.elements = elements
            self.bytes = bytes
            self.fields = fields
        }

        /// The block format of `type`, or nil for types the native engine
        /// does not carry. Values mirror `ggml-common.h` at the pin.
        public static func of(_ type: GGMLType) -> QuantBlockFormat? {
            func f(_ n: String, _ b: Int) -> BlockField { BlockField(name: n, bytes: b) }
            switch type {
            case .f32: return .init(type: type, elements: 1, bytes: 4, fields: [f("v", 4)])
            case .f16: return .init(type: type, elements: 1, bytes: 2, fields: [f("v", 2)])
            case .bf16: return .init(type: type, elements: 1, bytes: 2, fields: [f("v", 2)])
            case .q8_0: return .init(type: type, elements: 32, bytes: 34,
                                     fields: [f("d", 2), f("qs", 32)])
            case .q2_K: return .init(type: type, elements: 256, bytes: 84,
                                     fields: [f("scales", 16), f("qs", 64), f("d", 2), f("dmin", 2)])
            case .q4_K: return .init(type: type, elements: 256, bytes: 144,
                                     fields: [f("d", 2), f("dmin", 2), f("scales", 12), f("qs", 128)])
            case .q5_K: return .init(type: type, elements: 256, bytes: 176,
                                     fields: [f("d", 2), f("dmin", 2), f("scales", 12), f("qh", 32), f("qs", 128)])
            case .q6_K: return .init(type: type, elements: 256, bytes: 210,
                                     fields: [f("ql", 128), f("qh", 64), f("scales", 16), f("d", 2)])
            case .iq4_xs: return .init(type: type, elements: 256, bytes: 136,
                                       fields: [f("d", 2), f("scales_h", 2), f("scales_l", 4), f("qs", 128)])
            case .iq3_xxs: return .init(type: type, elements: 256, bytes: 98,
                                        fields: [f("d", 2), f("qs", 96)])
            case .iq3_s: return .init(type: type, elements: 256, bytes: 110,
                                      fields: [f("d", 2), f("qs", 64), f("qh", 8), f("signs", 32), f("scales", 4)])
            case .iq2_xxs: return .init(type: type, elements: 256, bytes: 66,
                                        fields: [f("d", 2), f("qs", 64)])
            case .iq2_xs: return .init(type: type, elements: 256, bytes: 74,
                                       fields: [f("d", 2), f("qs", 64), f("scales", 8)])
            case .iq2_s: return .init(type: type, elements: 256, bytes: 82,
                                      fields: [f("d", 2), f("qs", 64), f("qh", 8), f("scales", 8)])
            case .iq1_m: return .init(type: type, elements: 256, bytes: 56,
                                      fields: [f("qs", 32), f("qh", 16), f("scales", 8)])
            default: return nil
            }
        }
    }

    /// GPU family class used for kernel calibration (D5: Apple7 and Apple8 are
    /// first-class; no fast path may require Apple9+).
    public enum GPUFamilyClass: Int, Codable, Sendable, Comparable, CaseIterable {
        /// M1 / A14 generation.
        case apple7 = 7
        /// M2 / A15-A16 generation.
        case apple8 = 8
        /// M3 / A17 generation.
        case apple9 = 9
        /// M4 / A18 and later (tensor-ops capable families included).
        case apple10 = 10

        /// The highest class `device` supports.
        public init(device: MTLDevice) {
            if device.supportsFamily(.apple10) { self = .apple10 }
            else if device.supportsFamily(.apple9) { self = .apple9 }
            else if device.supportsFamily(.apple8) { self = .apple8 }
            else { self = .apple7 }
        }

        public static func < (a: GPUFamilyClass, b: GPUFamilyClass) -> Bool { a.rawValue < b.rawValue }
    }
}

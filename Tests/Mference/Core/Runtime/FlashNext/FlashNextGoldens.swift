import Foundation
@testable import Mference

/// Reader for the committed `qwen4_exp` reference goldens — either
/// `Tests/Mference/Fixtures/qwen4exp/` (fp32 weights) or
/// `Tests/Mference/Fixtures/qwen4exp-bf16/` (the weights the checkpoint
/// carries; see `WeightDtype`). Float tensors come from the `.safetensors`
/// captures, integer sets from the `integers_*.json` files.
///
/// The layout of each file is documented in `Scripts/parity/README.md`; the
/// only subtlety this reader encodes is the decode off-by-one: row `i` of a
/// `decode_*.safetensors` tensor is decode step `i + 1`, because the first
/// generated token comes from the cached prefill leg.
struct FlashNextGoldens {

    enum Prompt: String, CaseIterable {
        case short, long
    }

    enum Phase: String {
        case prefill, decode
    }

    /// Which capture the goldens were taken from.
    ///
    /// `fp32` is the original set: the reference ran with the float32 weights
    /// `Qwen4ExpForCausalLM(cfg)` initialized, and the checkpoint it emitted is a
    /// **lossy bfloat16 copy of those weights**. No consumer of the checkpoint
    /// can reproduce them — loading the shipped bytes moves the logits by
    /// 1.2e-3 (SHORT) / 3.1e-2 (LONG) max-abs, flips router top-k indices and
    /// indexer selected sets, and diverges the LONG greedy rollout at step 7.
    ///
    /// `bf16` is captured after rounding every parameter through bfloat16, so it
    /// describes the weights the checkpoint actually carries. That is the set a
    /// port which loads an install can gate on at 1e-4, and the one this suite
    /// uses. See `Scripts/parity/qwen4exp_make_goldens.py --weight-dtype`.
    enum WeightDtype: String {
        case fp32 = "qwen4exp"
        case bf16 = "qwen4exp-bf16"
    }

    struct Tensor {
        let shape: [Int]
        let values: [Float]

        /// Row `r` of a rank-2 tensor.
        func row(_ r: Int) -> [Float] {
            let width = shape[shape.count - 1]
            return Array(values[(r * width)..<((r + 1) * width)])
        }
    }

    let prompt: Prompt
    let phase: Phase
    let dtype: WeightDtype
    let tensors: [String: Tensor]
    let integers: [String: Any]

    static func directory(_ dtype: WeightDtype) -> URL {
        FlashNextParity.repoRoot
            .appendingPathComponent("Tests/Mference/Fixtures")
            .appendingPathComponent(dtype.rawValue)
    }

    static func manifest(_ dtype: WeightDtype = .bf16) throws -> [String: Any] {
        let url = directory(dtype).appendingPathComponent("goldens-manifest.json")
        guard let root = try JSONSerialization
            .jsonObject(with: Data(contentsOf: url)) as? [String: Any] else {
            throw CocoaError(.fileReadCorruptFile)
        }
        return root
    }

    /// `(atol, rtol)` from `findings.tolerances.recommended_swift_gate_fp32`.
    static func tolerances(_ dtype: WeightDtype = .bf16) throws -> (atol: Float, rtol: Float) {
        let root = try manifest(dtype)
        guard let findings = root["findings"] as? [String: Any],
              let tolerances = findings["tolerances"] as? [String: Any],
              let gate = tolerances["recommended_swift_gate_fp32"] as? [String: Any],
              let atol = (gate["atol"] as? NSNumber)?.floatValue,
              let rtol = (gate["rtol"] as? NSNumber)?.floatValue else {
            throw CocoaError(.fileReadCorruptFile)
        }
        return (atol, rtol)
    }

    static func promptTokens(_ prompt: Prompt,
                             _ dtype: WeightDtype = .bf16) throws -> [Int] {
        let root = try manifest(dtype)
        guard let prompts = root["prompts"] as? [String: Any],
              let entry = prompts[prompt.rawValue] as? [String: Any],
              let ids = entry["token_ids"] as? [Int] else {
            throw CocoaError(.fileReadCorruptFile)
        }
        return ids
    }

    static func greedyRollout(_ prompt: Prompt,
                              _ dtype: WeightDtype = .bf16) throws -> [Int] {
        let root = try manifest(dtype)
        guard let prompts = root["prompts"] as? [String: Any],
              let ids = prompts["greedy_rollout_\(prompt.rawValue)"] as? [Int] else {
            throw CocoaError(.fileReadCorruptFile)
        }
        return ids
    }

    init(prompt: Prompt, phase: Phase, dtype: WeightDtype = .bf16) throws {
        self.prompt = prompt
        self.phase = phase
        self.dtype = dtype
        let base = Self.directory(dtype)
        tensors = try Self.readSafetensors(
            base.appendingPathComponent("\(phase.rawValue)_\(prompt.rawValue).safetensors"))
        let json = try Data(contentsOf: base.appendingPathComponent(
            "integers_\(phase.rawValue)_\(prompt.rawValue).json"))
        guard let root = try JSONSerialization.jsonObject(with: json)
            as? [String: Any] else {
            throw CocoaError(.fileReadCorruptFile)
        }
        integers = root
    }

    // MARK: - Accessors

    func tensor(_ key: String) throws -> Tensor {
        guard let value = tensors[key] else {
            throw ModelError.tensorNotFound(name: key)
        }
        return value
    }

    /// Per-query-position integer lists for a prefill file, or for one decode
    /// step of a decode file. Decode files carry one extra outer level.
    func intLists(_ key: String, step: Int? = nil) throws -> [[Int]] {
        guard let value = integers[key] else {
            throw ModelError.tensorNotFound(name: key)
        }
        if let step {
            guard let outer = value as? [[[Int]]] else {
                throw CocoaError(.fileReadCorruptFile)
            }
            return outer[step]
        }
        guard let lists = value as? [[Int]] else {
            throw CocoaError(.fileReadCorruptFile)
        }
        return lists
    }

    func ints(_ key: String) throws -> [Int] {
        guard let value = integers[key] as? [Int] else {
            throw ModelError.tensorNotFound(name: key)
        }
        return value
    }

    /// `layer{NN}` key prefix the goldens use.
    static func layerKey(_ L: Int) -> String { String(format: "layer%02d", L) }

    // MARK: - safetensors

    private static func readSafetensors(_ url: URL) throws -> [String: Tensor] {
        let data = try Data(contentsOf: url, options: .mappedIfSafe)
        guard data.count >= 8 else { throw CocoaError(.fileReadCorruptFile) }
        var headerLength: UInt64 = 0
        _ = withUnsafeMutableBytes(of: &headerLength) { dst in
            data.copyBytes(to: dst, from: 0..<8)
        }
        let headerEnd = 8 + Int(headerLength)
        guard let root = try JSONSerialization
            .jsonObject(with: data.subdata(in: 8..<headerEnd)) as? [String: Any] else {
            throw CocoaError(.fileReadCorruptFile)
        }
        var out: [String: Tensor] = [:]
        for (name, value) in root where name != "__metadata__" {
            guard let object = value as? [String: Any],
                  let dtype = object["dtype"] as? String,
                  let shape = object["shape"] as? [Int],
                  let offsets = object["data_offsets"] as? [Int],
                  offsets.count == 2 else { continue }
            precondition(dtype == "F32", "\(name) is \(dtype); goldens are float32")
            let bytes = [UInt8](data[(headerEnd + offsets[0])..<(headerEnd + offsets[1])])
            let values = stride(from: 0, to: bytes.count, by: 4).map { i in
                Float(bitPattern: UInt32(bytes[i]) | UInt32(bytes[i + 1]) << 8
                        | UInt32(bytes[i + 2]) << 16 | UInt32(bytes[i + 3]) << 24)
            }
            out[name] = Tensor(shape: shape, values: values)
        }
        return out
    }
}

/// Elementwise comparison against the manifest's fp32 gate, reporting the
/// worst offender rather than just failing.
struct FlashNextDelta {
    var maxAbs: Float = 0
    var maxRel: Float = 0
    var worstIndex: Int = -1
    var count: Int = 0
    /// `max over i of |a-b| - (atol + rtol*|b|)`. Non-positive means every
    /// element satisfies `numpy.allclose`.
    var worstSlack: Float = -.infinity
    var mismatched: Int = 0

    /// Element-wise `numpy.allclose`: `|a - b| <= atol + rtol * |b|`.
    static func compare(_ actual: [Float], _ expected: [Float],
                        atol: Float, rtol: Float) -> FlashNextDelta {
        var delta = FlashNextDelta()
        precondition(actual.count == expected.count,
                     "shape mismatch: \(actual.count) vs \(expected.count)")
        delta.count = expected.count
        for i in 0..<delta.count {
            let absolute = abs(actual[i] - expected[i])
            let relative = absolute / max(abs(expected[i]), 1e-30)
            let slack = absolute - (atol + rtol * abs(expected[i]))
            if absolute > delta.maxAbs { delta.maxAbs = absolute }
            delta.maxRel = max(delta.maxRel, relative)
            if slack > delta.worstSlack {
                delta.worstSlack = slack
                delta.worstIndex = i
            }
            if slack > 0 { delta.mismatched += 1 }
        }
        return delta
    }

    var passes: Bool { mismatched == 0 }

    var description: String {
        String(format: "maxAbs=%.3e maxRel=%.3e (%d/%d outside the gate, worst at %d)",
               maxAbs, maxRel, mismatched, count, worstIndex)
    }
}

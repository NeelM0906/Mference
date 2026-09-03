import Foundation

/// Which weight width each tensor of an **original-repo** install is quantized
/// to. One base width plus a small table of name-suffix overrides.
///
/// # Why this exists
///
/// Workstream 2 shipped quantize-in-flight with exactly one target — INT4
/// affine group-64 — as a deliberate scope cut. That was invisible while
/// Flash-Next was the only original-repo family, because its runner drives the
/// router through the generic INT4 matvec. Qwen 3.6 made it visible: the
/// independent mlx-community conversion of the same checkpoint keeps every
/// layer's `mlp.gate` (the MoE router) and `mlp.shared_expert_gate` at **INT8**
/// group-64, its manifest therefore records `quant.router.weightBits = 8`, and
/// `ManifestReader.validateQuant` refuses anything else for that family.
///
/// The extra bits are load-bearing, not ceremonial. §6 of
/// `docs/QUANTIZER_QUALITY.md` measures an INT4 router changing roughly 13-18 %
/// of the selected top-8 expert set and about a quarter of the top-1 expert,
/// against ~1.5 % and ~2 % for INT8. A uniform-INT4 install is therefore not
/// merely refused, it would be *wrong* — and it would make a quantizer-quality
/// comparison against that control measure routing divergence rather than
/// weight fidelity.
///
/// # Why it is a general mechanism rather than a Qwen 3.6 branch
///
/// A vendor ships one BF16 checkpoint; the community converts it, and the
/// conversion decides per tensor how many bits each one gets. Mixed width is
/// the common case, not the exotic one — the first two original-repo families
/// already disagree about it. So the *mechanism* is a data table the planner
/// consults, and adding a family means adding rows, not branches. What is
/// necessarily family-specific is the table's contents, because it mirrors a
/// particular community conversion's `config.json` overrides; that provenance
/// is recorded on each policy below.
///
/// Rules are matched by name suffix, longest first, so a more specific rule
/// always wins over a more general one regardless of table order.
public struct QuantBitPolicy: Sendable, Equatable {

    public struct Rule: Sendable, Equatable {
        /// Tensor-name suffix, including the `.weight`.
        public let suffix: String
        public let bits: Int

        public init(suffix: String, bits: Int) {
            self.suffix = suffix
            self.bits = bits
        }
    }

    /// Width for any tensor no rule matches.
    public let defaultBits: Int
    /// Overrides, held sorted by descending suffix length.
    public let rules: [Rule]

    public init(defaultBits: Int, rules: [Rule]) {
        self.defaultBits = defaultBits
        self.rules = rules.sorted {
            $0.suffix.count == $1.suffix.count
                ? $0.suffix < $1.suffix
                : $0.suffix.count > $1.suffix.count
        }
    }

    /// Every width this policy can emit. The repacker has a streaming
    /// quantizer for 4 and 8 only, so anything else is rejected at
    /// construction time by `validated(for:)`.
    public static let supportedBits: Set<Int> = [4, 8]

    public static let uniformInt4 = QuantBitPolicy(defaultBits: 4, rules: [])

    /// The MoE-router policy: the two gating tensors of a Qwen-style MoE block
    /// keep INT8, everything else takes the INT4 base.
    ///
    /// Derived from mlx-community's `Qwen3.6-35B-A3B-4bit` conversion
    /// (rev `38740b84`), whose `config.json` carries exactly these two suffixes
    /// as per-tensor `bits: 8, group_size: 64` overrides, one pair per layer.
    /// `mlp.gate` is the top-8-of-256 router; `mlp.shared_expert_gate` is the
    /// scalar gate on the always-on shared expert. Both are gating tensors
    /// whose output is fed through a softmax/sigmoid and then *compared*, which
    /// is why quantization noise there costs far more than the same noise in a
    /// projection.
    public static let moeRouterInt8 = QuantBitPolicy(defaultBits: 4, rules: [
        Rule(suffix: ".mlp.gate.weight", bits: 8),
        Rule(suffix: ".mlp.shared_expert_gate.weight", bits: 8),
    ])

    /// The policy for a family read from its vendor's original BF16 repo.
    ///
    /// Enumerated rather than defaulted: a family reaching this path without a
    /// considered answer should be a compile error at the next `case`, not a
    /// silent uniform-INT4 install that a runner may refuse (or, worse,
    /// accept and mis-route).
    // Internal rather than public: `RepackModelFamily` is internal.
    static func originalRepo(family: RepackModelFamily) -> QuantBitPolicy {
        switch family {
        case .qwen38flashnext:
            // Uniform INT4 on purpose. Flash-Next has no independent community
            // conversion to mirror, and its runner drives the router through
            // the generic INT4 matvec, so nothing here is INT8. Changing this
            // would change every byte that family has ever installed.
            return .uniformInt4
        case .qwen36:
            return .moeRouterInt8
        case .gemma4, .qwen38, .deepseekV4Flash, .inklingSmall, .maple:
            // None of these has an original-repo installer entry today, so no
            // conversion has been examined and no table can be honest. Uniform
            // INT4 is the base the quantize-in-flight path was built for; a
            // family arriving here should confirm its conversion's overrides
            // before trusting it.
            return .uniformInt4
        }
    }

    // MARK: - Application

    /// Width for `name`, or `defaultBits` when nothing matches.
    public func bits(forTensorNamed name: String) -> Int {
        for rule in rules where name.hasSuffix(rule.suffix) {
            return rule.bits
        }
        return defaultBits
    }

    /// `true` when `name` is quantized at something other than the base width.
    /// Counting these is what the manifest's `bitWidthOverridesHonored` audit
    /// records, so it can be compared against the control conversion's own
    /// override count.
    public func overrides(_ name: String) -> Bool {
        bits(forTensorNamed: name) != defaultBits
    }

    /// Fail loudly at plan time rather than emitting an install whose bytes no
    /// streaming quantizer knows how to produce.
    func validated(for family: RepackModelFamily) throws -> QuantBitPolicy {
        var widths = Set(rules.map(\.bits))
        widths.insert(defaultBits)
        guard widths.isSubset(of: Self.supportedBits) else {
            throw RepackError.configurationInvalid(
                detail: "quantization bit policy for \(family.rawValue) asks for "
                    + "\(widths.sorted()) bits; the repacker streams only "
                    + "\(Self.supportedBits.sorted())")
        }
        return self
    }
}

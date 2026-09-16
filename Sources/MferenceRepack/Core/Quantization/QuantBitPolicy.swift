import Foundation

/// Which weight width each tensor of an **original-repo** install is quantized
/// to. One base width plus a small table of name-suffix overrides.
///
/// # Why this exists
///
/// Workstream 2 shipped quantize-in-flight with exactly one target — INT4
/// affine group-64 — as a deliberate scope cut. Qwen 3.6 made that visible: the
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
/// Flash-Next looked like the counter-example for a while: it has no community
/// conversion to disagree with, its runner accepted a uniform-INT4 router, and
/// it shipped one. Measuring it settled the matter the other way — the deficit
/// is the same size on the real install, so the "no conversion to mirror"
/// argument turned out to be an absence of evidence rather than evidence of
/// absence. Both original-repo families now carry the same two overrides.
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
    ///
    /// It is now the policy for **both** original-repo families. Qwen 3.6
    /// adopted it because a trusted conversion had already made that choice;
    /// Flash-Next adopted it because the same measurement was made directly on
    /// its own shipped install and came out worse
    /// (`docs/experiments/2026-09-10-flashnext-router-int4-check.md`): its
    /// INT4 top-10-of-512 routers change ~14 % of the selected expert set per
    /// token against BF16, and agree exactly with the BF16 selection on only
    /// 13.2 % of tokens, where mlx-community's INT8 routers manage ~1.5 % and
    /// 85.5 %. The suffixes are the same because Flash-Next is the same MoE
    /// block shape, one vendor generation on — including the MTP draft layer,
    /// whose `mtp.layers.0.mlp.gate.weight` and
    /// `mtp.layers.0.mlp.shared_expert_gate.weight` are matched by the same two
    /// rules and get the same width. That is deliberate: the draft layer routes
    /// over the same 512 experts and a draft that routes differently from the
    /// target wastes verification, so it is not a tensor to economize on.
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
        case .qwen38flashnext, .qwen36:
            // Qwen 3.6 mirrors mlx-community's conversion of the same
            // checkpoint. Flash-Next has no community conversion to mirror, so
            // the same question was answered by measuring its own shipped
            // INT4 install against the BF16 source: ~14 % of the selected
            // top-10-of-512 experts change per token, and exact top-10
            // agreement is 0.132. Both gating tensors therefore keep INT8, on
            // the text layers and on the MTP draft layer alike.
            //
            // This changes the bytes of a `qwen38flashnext` install relative to
            // the uniform-INT4 one already on disk. That is the intent, and it
            // is why the old install stays readable rather than reproducible:
            // `ManifestReader.validateQuant` accepts 4 or 8 on the router slot,
            // and `FlashNextWeightMatrix` reads each tensor's width from its own
            // index entry rather than from a family-wide assumption.
            return .moeRouterInt8
        case .qwen38:
            // Dense Qwen3_5 text model: MLX's quant_predicate returns no
            // per-projection overrides when num_experts <= 0. Norms and
            // convolution kernels remain unquantized in the planner.
            return .uniformInt4
        case .minicpm5:
            // Uniform INT4, mirroring the vendor's own MLX conversion
            // (`openbmb/MiniCPM5-2B-MLX` rev `35ac38ee`): its config carries
            // the base `{bits 4, group_size 64, mode affine}` block and no
            // per-tensor overrides — every projection including
            // `embed_tokens` and `lm_head` is INT4 g64, and a dense llama has
            // no router to keep wider.
            return .uniformInt4
        case .gemma4, .deepseekV4Flash, .inklingSmall, .maple, .glm53Flash:
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

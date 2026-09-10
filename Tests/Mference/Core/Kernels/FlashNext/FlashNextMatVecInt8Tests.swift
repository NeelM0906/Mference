import Foundation
import Metal
import Testing
@testable import Mference
import MferenceValidationSupport

/// `FlashNextMatVec`'s INT8 branch, at the two shapes a `qwen38flashnext`
/// install can actually reach it with.
///
/// # What this does and does not gate
///
/// The *kernel* is not on trial here. The INT8 branch dispatches
/// `router_gemv_gemma4_r4` — the shipped Gemma/Qwen router GEMV — with no new
/// Metal at all, and `RouterWideTopK10Tests` already gates that kernel against
/// a CPU reference at this family's exact production geometry (512 x 2560,
/// INT8 affine group-64, top-10), through `MoE.encodeRouterGemma4`'s identical
/// dispatch. Re-deriving that here would be duplicating a passing gate.
///
/// What is on trial is the *wrapper*: that `FlashNextMatVec` binds the same
/// kernel correctly from its own call site. The failure modes it protects
/// against are the ones a shared-kernel reuse actually has — a buffer index
/// off by one, the ones activation-scale vector missing or the wrong dtype
/// (a BF16 `1.0` is `0x3F80`, and an FP16 `1.0` written into the same slot
/// would be read as ~`1.5e-5`, which scales every logit to zero rather than
/// crashing), the `_r4` threadgroup width taken from the BF16 path's 8 rows,
/// or `rows`/`cols` swapped in the constant slots.
///
/// The second shape is `[1, hidden]`, the shared expert's scalar gate. One row
/// is the degenerate case of a kernel that assigns four rows to a threadgroup,
/// and it is a real production call site, so it gets its own case.
@Suite struct FlashNextMatVecInt8Tests {

    private static let hidden = 2560
    private static let experts = 512

    private struct Fixture {
        let rows: [Quantization.Int8AffineRow]
        let x: [Float16]
        let expected: [Float]
    }

    /// Random rows are fine here — unlike `RouterWideTopK10Tests`, nothing
    /// downstream ranks these values, so float summation order only has to
    /// agree to a tolerance rather than to decide a selection.
    private static func fixture(rows m: Int, seed: UInt64) -> Fixture {
        var rng = SplitMix64(seed: seed)
        let xF32 = (0..<hidden).map { _ in rng.uniform(-1.0, 1.0) }
        let xF16 = xF32.map { Float16($0) }
        // The kernel reads FP16 activations, so the reference must dot against
        // the FP16-rounded values or it would measure the cast, not the GEMV.
        let xRef = xF16.map { Float($0) }
        var rows: [Quantization.Int8AffineRow] = []
        rows.reserveCapacity(m)
        for _ in 0..<m {
            rows.append(Quantization.quantizeInt8Affine(
                (0..<hidden).map { _ in rng.uniform(-1.0, 1.0) }))
        }
        return Fixture(rows: rows, x: xF16,
                       expected: DequantInt8GemvRef.apply(weightRows: rows,
                                                          x: xRef, n: hidden))
    }

    private static func run(_ fixture: Fixture, rows m: Int) throws -> [Float] {
        let context = try MetalContext()
        let matVec = try FlashNextMatVec(
            context: context,
            int4: try DequantInt4GEMV(context: context),
            int8Columns: hidden)

        let groups = hidden / Quantization.groupSize
        var packed = [UInt8](); packed.reserveCapacity(m * hidden)
        var scales = [UInt16](); scales.reserveCapacity(m * groups)
        var biases = [UInt16](); biases.reserveCapacity(m * groups)
        for row in fixture.rows {
            packed.append(contentsOf: row.packed)
            scales.append(contentsOf: row.scales)
            biases.append(contentsOf: row.biases)
        }

        guard let weightBuffer = context.device.makeBuffer(
                  bytes: packed, length: packed.count, options: .storageModeShared),
              let scaleBuffer = context.device.makeBuffer(
                  bytes: scales, length: scales.count * MemoryLayout<UInt16>.stride,
                  options: .storageModeShared),
              let biasBuffer = context.device.makeBuffer(
                  bytes: biases, length: biases.count * MemoryLayout<UInt16>.stride,
                  options: .storageModeShared),
              let xBuffer = Fp16Buffer.make(context.device, halves: fixture.x),
              let yBuffer = context.device.makeBuffer(
                  length: m * MemoryLayout<Float>.stride, options: .storageModeShared),
              let commandBuffer = context.queue.makeCommandBuffer() else {
            throw CocoaError(.fileReadUnknown)
        }
        // A poison value: a kernel that never ran, or that wrote fewer rows
        // than it claimed, must not pass by leaving zeros behind.
        yBuffer.contents().bindMemory(to: Float.self, capacity: m)
            .update(repeating: .nan, count: m)

        matVec.encode(commandBuffer: commandBuffer,
                      matrix: .int8(weights: weightBuffer, weightsOffset: 0,
                                    scales: scaleBuffer, scalesOffset: 0,
                                    biases: biasBuffer, biasesOffset: 0),
                      x: xBuffer, y: yBuffer,
                      rows: m, cols: hidden, outputFloat32: true)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.error == nil)
        let out = yBuffer.contents().bindMemory(to: Float.self, capacity: m)
        return (0..<m).map { out[$0] }
    }

    /// The router: `[512, 2560]`, the shape every text layer and the MTP draft
    /// layer carry.
    @Test func routerShapeMatchesTheCPUReference() throws {
        let fixture = Self.fixture(rows: Self.experts, seed: 0x5EED_1008)
        let actual = try Self.run(fixture, rows: Self.experts)
        let scale = fixture.expected.map { abs($0) }.max() ?? 1
        var worst: Float = 0
        for (a, b) in zip(actual, fixture.expected) {
            #expect(!a.isNaN, "the kernel left a row unwritten")
            worst = max(worst, abs(a - b))
        }
        // 2560 FP32 products reduced through a 32-lane tree against a
        // sequential vDSP dot: agreement is to float rounding, not bit for bit.
        #expect(worst < 2e-3 * max(scale, 1),
                "512x2560 INT8 router logits differ by \(worst) (scale \(scale))")
    }

    /// The shared expert's scalar gate: `[1, 2560]`. One row against a kernel
    /// that hands four rows to a threadgroup, so the three inactive SIMD groups
    /// must return without writing.
    @Test func scalarGateShapeMatchesTheCPUReference() throws {
        let fixture = Self.fixture(rows: 1, seed: 0x5EED_1009)
        let actual = try Self.run(fixture, rows: 1)
        #expect(!actual[0].isNaN, "the kernel left the single row unwritten")
        let expected = fixture.expected[0]
        #expect(abs(actual[0] - expected) < 2e-3 * max(abs(expected), 1),
                "1x2560 INT8 gate logit \(actual[0]) vs \(expected)")
    }

    // MARK: - Width derivation

    /// The resident index writes dtype 0 at both INT4 and INT8, so
    /// `FlashNextWeightMatrix` derives the width from `sizeBytes` and the
    /// shape. This is what keeps the uniform-INT4 install — where the same two
    /// tensors are dtype 0 with half the bytes — loading through the INT4 path
    /// unchanged.
    @Test("the stored width is derived from bytes and shape, not from the family",
          arguments: [(4, 512 * 2560 / 2), (8, 512 * 2560),
                      (4, 1 * 2560 / 2), (8, 1 * 2560)])
    func packedWidthIsDerivedFromTheEntry(bits: Int, bytes: Int) throws {
        let context = try MetalContext()
        let rows = UInt32(bytes * 8 / bits / 2560)
        guard let buffer = context.device.makeBuffer(
                length: max(bytes, 1), options: .storageModeShared) else {
            throw CocoaError(.fileReadUnknown)
        }
        let view = TensorView(buffer: buffer,
                              offset: 0, length: UInt64(bytes),
                              scaleOffset: 0, scaleLength: 0,
                              biasOffset: 0, biasLength: 0,
                              shape: (rows, 2560, 0, 0), dtype: 0)
        #expect(FlashNextWeightMatrix.packedWeightBits(view) == bits)
        switch FlashNextWeightMatrix.from(view) {
        case .int4: #expect(bits == 4)
        case .int8: #expect(bits == 8)
        case .bf16: Issue.record("dtype 0 must not resolve to BF16")
        }
    }
}

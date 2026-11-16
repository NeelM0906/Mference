import Foundation
import Metal

/// Internal native-draft alignment, not a speculative decoder. Attach `consume`
/// to the target's all-row consumer, then explicitly finish the pending tail
/// with the next target-selected token. No target row is replayed or invented.
/// The owner must checkpoint/restore BOTH target and primer around a branch.
final class FlashNextMTPPrimer {
    struct Checkpoint {
        fileprivate let owner: UUID
        fileprivate let epoch: UInt64
        fileprivate let targetPosition: Int
        fileprivate let pending: MTLBuffer?
        fileprivate let nextInput: Int32?
        fileprivate let draft: FlashNextMTPDraftRunner.Checkpoint
    }

    private let context: MetalContext
    private let draft: FlashNextMTPDraftRunner
    private let maxContext: Int
    private let vocabSize: Int
    private let rowBytes: Int
    private let rowScratch: MTLBuffer
    private let scratchLogits: MTLBuffer
    private let owner = UUID()
    private var epoch: UInt64 = 0
    private var dirty = false
    private var pending: MTLBuffer?
    private var nextInput: Int32?
    private(set) var targetPosition = 0
    var draftPosition: Int { draft.position }
    /// Correctness-only cancellation seam after a committed draft row.
    var didPrimeRow: ((Int) throws -> Void)?

    init(model: Model, context: MetalContext, maxContext: Int,
         policy: FlashNextMTPDraftRunner.ExpertPolicy) throws {
        self.context = context
        self.maxContext = maxContext
        vocabSize = model.config.vocabSize
        rowBytes = model.config.residualStreamWidth * 2
        draft = try FlashNextMTPDraftRunner(model: model, context: context, maxContext: maxContext, policy: policy)
        guard let row = context.device.makeBuffer(length: rowBytes, options: .storageModePrivate),
              let logits = context.device.makeBuffer(length: vocabSize * 2, options: .storageModePrivate) else {
            throw FlashNextForwardRunnerError.invalidInput("cannot allocate MTP priming buffers")
        }
        rowScratch = row
        scratchLogits = logits
    }

    func reset() {
        draft.reset()
        targetPosition = 0
        pending = nil
        nextInput = nil
        dirty = false
        epoch &+= 1
    }

    func checkpoint() throws -> Checkpoint {
        guard !dirty else { throw failure("cannot checkpoint dirty MTP priming") }
        return Checkpoint(owner: owner, epoch: epoch, targetPosition: targetPosition,
            pending: pending, nextInput: nextInput, draft: try draft.checkpoint())
    }

    func restore(_ checkpoint: Checkpoint) throws {
        guard checkpoint.owner == owner, checkpoint.epoch == epoch,
              checkpoint.targetPosition <= targetPosition else { throw failure("foreign or stale MTP priming checkpoint") }
        try draft.restore(checkpoint.draft)
        targetPosition = checkpoint.targetPosition
        pending = checkpoint.pending
        nextInput = checkpoint.nextInput
        dirty = false
        epoch &+= 1
    }

    /// Preserve one owned tail row until its successor token is known. Chunk
    /// boundaries therefore cannot silently pair a tail with its own token.
    func consume(_ rows: FlashNextForwardRunner.TargetHiddenRows) throws {
        guard !dirty, rows.startPosition == targetPosition, !rows.tokens.isEmpty,
              rows.tokens.count <= maxContext - targetPosition,
              rows.buffer.device.registryID == context.device.registryID,
              rows.buffer.length >= rows.tokens.count * rowBytes,
              rows.tokens.allSatisfy({ $0 >= 0 && Int($0) < vocabSize }),
              nextInput == nil || nextInput == rows.tokens.first else {
            throw failure("invalid target rows, position or shifted-tail token")
        }
        try Task.checkCancellation()
        dirty = true
        if let pending {
            _ = try draft.append(token: rows.tokens[0], targetHidden: pending,
                at: targetPosition - 1, into: scratchLogits)
            try didPrimeRow?(draft.position - 1)
        }
        for row in 0..<(rows.tokens.count - 1) {
            try copyRow(rows.buffer, offset: row * rowBytes, to: rowScratch)
            _ = try draft.append(token: rows.tokens[row + 1], targetHidden: rowScratch,
                at: targetPosition + row, into: scratchLogits)
            try didPrimeRow?(draft.position - 1)
        }
        guard let tail = context.device.makeBuffer(length: rowBytes, options: .storageModePrivate) else {
            throw failure("cannot preserve MTP priming tail")
        }
        try copyRow(rows.buffer, offset: (rows.tokens.count - 1) * rowBytes, to: tail)
        pending = tail
        nextInput = nil
        targetPosition += rows.tokens.count
        dirty = false
    }

    /// Complete the last pair using a target-selected next token (or the known
    /// next prompt token). A later target append must begin with that token.
    /// Output predicts the token AFTER nextToken; no token is emitted here.
    func finish(nextToken: Int32, into logits: MTLBuffer) throws -> FlashNextMTPDraftRunner.Output {
        guard !dirty, let pending, nextToken >= 0, Int(nextToken) < vocabSize,
              logits.length >= vocabSize * 2 else { throw failure("missing priming tail or invalid next token/output") }
        try Task.checkCancellation()
        dirty = true
        let result = try draft.append(token: nextToken, targetHidden: pending,
            at: targetPosition - 1, into: logits)
        try didPrimeRow?(draft.position - 1)
        self.pending = nil
        nextInput = nextToken
        dirty = false
        return result
    }

    /// Branch from the real target tail, then feed each draft's own full HC
    /// output into the next draft. Always restore priming state before returning;
    /// only target-verified rows may later enter the committed draft cache.
    func proposals(after nextToken: Int32, count: Int, into logits: MTLBuffer,
                   sample: (MTLBuffer) throws -> Int32) throws -> [Int32] {
        guard count > 0, count < maxContext - targetPosition else {
            throw failure("draft proposal count exceeds remaining target capacity")
        }
        let before = try checkpoint()
        var tokens: [Int32] = []
        do {
            var output = try finish(nextToken: nextToken, into: logits)
            for index in 0..<count {
                try Task.checkCancellation()
                let token = try sample(logits)
                guard token >= 0, Int(token) < vocabSize else { throw failure("invalid native draft sample") }
                tokens.append(token)
                if index + 1 < count {
                    output = try draft.append(token: token, targetHidden: output.hidden,
                        at: draft.position, into: logits)
                    try didPrimeRow?(draft.position - 1)
                }
            }
        } catch {
            try restore(before)
            throw error
        }
        try restore(before)
        return tokens
    }

    private func copyRow(_ source: MTLBuffer, offset: Int, to destination: MTLBuffer) throws {
        guard let cb = context.queue.makeCommandBuffer(), let blit = cb.makeBlitCommandEncoder() else {
            throw failure("cannot copy MTP priming row")
        }
        blit.copy(from: source, sourceOffset: offset, to: destination, destinationOffset: 0, size: rowBytes)
        blit.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        if let error = cb.error { throw failure("MTP priming copy failed: \(error)") }
    }

    private func failure(_ detail: String) -> FlashNextForwardRunnerError { .invalidInput(detail) }
}

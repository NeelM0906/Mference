import Metal

/// Optional approximate Maple head for singleton decode. It scores all
/// quantized centroids, then evaluates exact original LM-head rows only for
/// tokens in the selected clusters and the configured control-token set.
final class MapleFlashHead {
    private static let hiddenSize = 2_048
    private static let vocabularySize = 151_936

    private let context: MetalContext
    private let ternary: MapleTernaryGEMV
    private let metadata: ManifestMapleFlashHead
    private let centroids: TensorView
    private let tokenMap: [UInt32]
    private let centroidScores: MTLBuffer
    private let candidateTokens: MTLBuffer
    private let fillPipeline: MTLComputePipelineState
    private let gatherPipeline: MTLComputePipelineState

    init?(context: MetalContext, model: Model, ternary: MapleTernaryGEMV) throws {
        guard let metadata = model.manifest.flashHead else { return nil }
        let centroids = try model.resident(name: "lm_head_flash.centroids.weight")
        let map = try model.resident(name: "lm_head_flash.token_map")
        try Self.validate(metadata: metadata, centroids: centroids, tokenMap: map)

        let count = metadata.nClusters * metadata.clusterSize
        let mapPointer = map.buffer.contents().advanced(by: Int(map.offset))
            .bindMemory(to: Int32.self, capacity: count)
        var tokenMap: [UInt32] = []
        tokenMap.reserveCapacity(count)
        for index in 0..<count {
            let token = mapPointer[index]
            guard token >= 0 && token < Int32(Self.vocabularySize) else {
                throw MapleForwardRunnerError.invalidConfiguration(
                    "Maple FlashHead token map contains an out-of-range token")
            }
            tokenMap.append(UInt32(token))
        }

        guard let centroidScores = context.device.makeBuffer(
            length: metadata.nClusters * MemoryLayout<Float16>.stride,
            options: .storageModeShared),
              let candidateTokens = context.device.makeBuffer(
                length: (metadata.nProbes * metadata.clusterSize + metadata.forceTokens.count)
                    * MemoryLayout<UInt32>.stride,
                options: .storageModeShared) else {
            throw MapleForwardRunnerError.invalidConfiguration(
                "unable to allocate Maple FlashHead scratch")
        }
        let library = try MetalContext.moduleLibrary(device: context.device,
                                                     module: "maple_flash_head",
                                                     safeMath: true)
        guard let fill = library.makeFunction(name: "maple_flash_head_fill_negative_infinity"),
              let gather = library.makeFunction(name: "maple_flash_head_gather_int4_qmv_d2048") else {
            throw MetalError.missingFunction("maple FlashHead kernel")
        }

        self.context = context
        self.ternary = ternary
        self.metadata = metadata
        self.centroids = centroids
        self.tokenMap = tokenMap
        self.centroidScores = centroidScores
        self.candidateTokens = candidateTokens
        self.fillPipeline = try context.device.makeComputePipelineState(function: fill)
        self.gatherPipeline = try context.device.makeComputePipelineState(function: gather)
    }

    func encode(hidden: MTLBuffer, lmHead: TensorView, into logits: MTLBuffer) throws {
        let centroidPass = try commandBuffer()
        ternary.encodeInt4(commandBuffer: centroidPass,
                           weights: centroids.buffer, weightsOffset: Int(centroids.offset),
                           scales: centroids.buffer, scalesOffset: Int(centroids.scaleOffset),
                           biases: centroids.buffer, biasesOffset: Int(centroids.biasOffset),
                           x: hidden,
                           y: centroidScores,
                           rows: UInt32(metadata.nClusters))
        try finish(centroidPass)

        let count = writeCandidates()
        let headPass = try commandBuffer()
        encodeFill(commandBuffer: headPass, logits: logits)
        encodeGather(commandBuffer: headPass, hidden: hidden, lmHead: lmHead, logits: logits,
                     rows: count)
        try finish(headPass)
    }

    private func writeCandidates() -> UInt32 {
        let scores = centroidScores.contents().bindMemory(to: UInt16.self,
                                                           capacity: metadata.nClusters)
        let clusters = (0..<metadata.nClusters).sorted { left, right in
            let a = Self.rankingScore(scores[left])
            let b = Self.rankingScore(scores[right])
            return a == b ? left < right : a > b
        }.prefix(metadata.nProbes)
        var seen = Set<UInt32>()
        var candidates: [UInt32] = []
        candidates.reserveCapacity(metadata.nProbes * metadata.clusterSize + metadata.forceTokens.count)
        for cluster in clusters {
            let start = cluster * metadata.clusterSize
            for index in start..<(start + metadata.clusterSize) {
                let token = tokenMap[index]
                if seen.insert(token).inserted { candidates.append(token) }
            }
        }
        for token in metadata.forceTokens.map(UInt32.init) {
            if seen.insert(token).inserted { candidates.append(token) }
        }
        let destination = candidateTokens.contents().bindMemory(to: UInt32.self,
                                                                 capacity: candidates.count)
        for (index, token) in candidates.enumerated() { destination[index] = token }
        return UInt32(candidates.count)
    }

    private func encodeFill(commandBuffer: MTLCommandBuffer, logits: MTLBuffer) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(fillPipeline)
        encoder.setBuffer(logits, offset: 0, index: 0)
        var count = UInt32(Self.vocabularySize)
        encoder.setBytes(&count, length: MemoryLayout<UInt32>.stride, index: 1)
        encoder.dispatchThreads(MTLSize(width: Self.vocabularySize, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
    }

    private func encodeGather(commandBuffer: MTLCommandBuffer,
                              hidden: MTLBuffer,
                              lmHead: TensorView,
                              logits: MTLBuffer,
                              rows: UInt32) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(gatherPipeline)
        encoder.setBuffer(lmHead.buffer, offset: Int(lmHead.offset), index: 0)
        encoder.setBuffer(lmHead.buffer, offset: Int(lmHead.scaleOffset), index: 1)
        encoder.setBuffer(lmHead.buffer, offset: Int(lmHead.biasOffset), index: 2)
        encoder.setBuffer(hidden, offset: 0, index: 3)
        encoder.setBuffer(candidateTokens, offset: 0, index: 4)
        encoder.setBuffer(logits, offset: 0, index: 5)
        var rowCount = rows
        encoder.setBytes(&rowCount, length: MemoryLayout<UInt32>.stride, index: 6)
        encoder.dispatchThreadgroups(MTLSize(width: (Int(rows) + 7) / 8, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        encoder.endEncoding()
    }

    private func commandBuffer() throws -> MTLCommandBuffer {
        guard let commandBuffer = context.queue.makeCommandBuffer() else {
            throw MapleForwardRunnerError.commandFailed("unable to create Maple FlashHead command buffer")
        }
        return commandBuffer
    }

    private func finish(_ commandBuffer: MTLCommandBuffer) throws {
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        guard commandBuffer.status == .completed else {
            throw MapleForwardRunnerError.commandFailed(
                commandBuffer.error?.localizedDescription ?? "Maple FlashHead command buffer did not complete")
        }
    }

    private static func rankingScore(_ bits: UInt16) -> Float {
        let score = Float(Float16(bitPattern: bits))
        return score.isNaN ? -.infinity : score
    }

    private static func validate(metadata: ManifestMapleFlashHead,
                                 centroids: TensorView,
                                 tokenMap: TensorView) throws {
        guard metadata.nClusters > 0, metadata.clusterSize > 0,
              metadata.nClusters <= Int.max / metadata.clusterSize,
              metadata.nClusters * metadata.clusterSize == vocabularySize,
              metadata.nProbes > 0, metadata.nProbes <= metadata.nClusters,
              metadata.groupSize == Quantization.groupSize, metadata.bits == 4,
              metadata.headGroupSize == Quantization.groupSize, metadata.headBits == 4,
              metadata.scaledCentroids,
              Set(metadata.forceTokens).count == metadata.forceTokens.count,
              metadata.forceTokens.allSatisfy({ $0 >= 0 && $0 < vocabularySize }) else {
            throw MapleForwardRunnerError.invalidConfiguration("Maple FlashHead metadata is invalid")
        }
        let centroidWeights = UInt64(metadata.nClusters * hiddenSize / 2)
        let centroidParameters = UInt64(metadata.nClusters * (hiddenSize / Quantization.groupSize)
            * MemoryLayout<UInt16>.stride)
        guard centroids.dtype == 0,
              centroids.shape.0 == UInt32(metadata.nClusters),
              centroids.shape.1 == UInt32(hiddenSize),
              centroids.shape.2 == 0, centroids.shape.3 == 0,
              centroids.length == centroidWeights,
              centroids.scaleLength == centroidParameters,
              centroids.biasLength == centroidParameters,
              tokenMap.dtype == 5,
              tokenMap.shape.0 == UInt32(metadata.nClusters),
              tokenMap.shape.1 == UInt32(metadata.clusterSize),
              tokenMap.shape.2 == 0, tokenMap.shape.3 == 0,
              tokenMap.length == UInt64(metadata.nClusters * metadata.clusterSize
                                          * MemoryLayout<Int32>.stride),
              tokenMap.scaleLength == 0, tokenMap.biasLength == 0 else {
            throw MapleForwardRunnerError.invalidConfiguration("Maple FlashHead tensors are invalid")
        }
        try validateRange(centroids, companions: true, name: "centroids")
        try validateRange(tokenMap, companions: false, name: "token map")
    }

    private static func validateRange(_ view: TensorView,
                                      companions: Bool,
                                      name: String) throws {
        let bufferLength = UInt64(view.buffer.length)
        func contains(_ offset: UInt64, _ length: UInt64) -> Bool {
            offset <= bufferLength && length <= bufferLength - offset
        }
        guard contains(view.offset, view.length),
              (!companions || (contains(view.scaleOffset, view.scaleLength)
                               && contains(view.biasOffset, view.biasLength))),
              view.offset.isMultiple(of: UInt64(MemoryLayout<UInt16>.stride)),
              (!companions || (view.scaleOffset.isMultiple(of: UInt64(MemoryLayout<UInt16>.stride))
                               && view.biasOffset.isMultiple(of: UInt64(MemoryLayout<UInt16>.stride)))) else {
            throw MapleForwardRunnerError.invalidConfiguration(
                "Maple FlashHead \(name) range is invalid")
        }
    }
}

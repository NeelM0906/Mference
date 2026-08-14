import Metal

/// Fixed-shape Maple decode MoE primitives. Inputs and outputs marked BF16
/// occupy two-byte Metal buffers containing native BF16 bit patterns. An
/// instance is serial: its router logits and routed argument table must not
/// be rebound until their prior command buffers complete.
final class MapleMoE {
    /// One routed expert's blob as a byte range inside a cache buffer. Slot
    /// caches pack many experts into a single slab buffer, so consumers must
    /// carry the offset; a whole dedicated buffer is (buffer, 0, length).
    typealias RoutedBlob = (buffer: MTLBuffer, offset: Int, length: Int)

    static let dimension = 2_048
    static let intermediate = 512
    static let expertCount = 256
    static let topK = 8

    private static let routerWeightBytes = expertCount * dimension * MemoryLayout<UInt16>.stride
    private static let vectorBytes = dimension * MemoryLayout<UInt16>.stride
    private static let actsBytes = topK * intermediate * MemoryLayout<UInt16>.stride
    private static let routeIndexBytes = topK * MemoryLayout<UInt32>.stride
    private static let routeWeightBytes = topK * MemoryLayout<Float>.stride
    private static let projectionWeightBytes = intermediate * dimension / 4
    private static let projectionCompanionBytes = intermediate * (dimension / 64)
        * MemoryLayout<UInt16>.stride
    private static let downWeightBytes = dimension * intermediate / 4
    private static let downCompanionBytes = dimension * (intermediate / 64)
        * MemoryLayout<UInt16>.stride

    private let router: MTLComputePipelineState
    private let select: MTLComputePipelineState
    private let phase1: MTLComputePipelineState
    private let phase1Subset: MTLComputePipelineState
    private let phase2: MTLComputePipelineState
    private let logits: MTLBuffer
    private let routedArgumentEncoder: MTLArgumentEncoder
    private let routedArgumentBuffer: MTLBuffer
    private let routedArgumentBytes: Int
    private var boundBlobs: [RoutedBlob]?
    private var boundOffsets: MoEExpertOffsets?
    private var routedArgumentUsers: [MTLCommandBuffer] = []
    private var routerLogitsUser: MTLCommandBuffer?

    init(context: MetalContext) throws {
        let library = try MetalContext.moduleLibrary(
            device: context.device, module: "maple_moe", safeMath: true)
        guard let routerFunction = library.makeFunction(name: "maple_router_bf16_gemv"),
              let selectFunction = library.makeFunction(name: "maple_router_top8_full_softmax"),
              let phase1Function = library.makeFunction(name: "maple_moe_phase1"),
              let subsetFunction = library.makeFunction(name: "maple_moe_phase1_subset"),
              let phase2Function = library.makeFunction(name: "maple_moe_phase2")
        else {
            throw MetalError.missingFunction("Maple MoE kernels")
        }
        let router = try Self.pipeline(routerFunction, device: context.device, threads: 256)
        guard router.threadExecutionWidth == 32 else {
            throw MetalError.libraryCompileFailed(
                "Maple router GEMV requires a 32-lane SIMD width")
        }
        self.router = router
        self.select = try Self.pipeline(selectFunction, device: context.device, threads: 256)
        self.phase1 = try Self.pipeline(phase1Function, device: context.device, threads: 64)
        self.phase1Subset = try Self.pipeline(subsetFunction, device: context.device, threads: 64)
        self.phase2 = try Self.pipeline(phase2Function, device: context.device, threads: 256)
        let argumentEncoder = phase1Function.makeArgumentEncoder(bufferIndex: 0)
        guard let logits = context.device.makeBuffer(
                  length: Self.expertCount * MemoryLayout<Float>.stride,
                  options: .storageModePrivate),
              let argumentBuffer = context.device.makeBuffer(
                  length: argumentEncoder.encodedLength,
                  options: .storageModeShared)
        else {
            throw MetalError.noDevice
        }
        self.logits = logits
        self.routedArgumentEncoder = argumentEncoder
        self.routedArgumentBuffer = argumentBuffer
        self.routedArgumentBytes = argumentEncoder.encodedLength
    }

    func encodeRouterTop8(commandBuffer: MTLCommandBuffer,
                          weights: MTLBuffer, weightsOffset: Int = 0,
                          hidden: MTLBuffer, hiddenOffset: Int = 0,
                          indices: MTLBuffer, indicesOffset: Int = 0,
                          routingWeights: MTLBuffer, routingWeightsOffset: Int = 0) {
        Self.requireRange(weights, offset: weightsOffset, bytes: Self.routerWeightBytes,
                          alignment: MemoryLayout<UInt16>.stride, named: "router weights")
        Self.requireRange(hidden, offset: hiddenOffset, bytes: Self.vectorBytes,
                          alignment: MemoryLayout<UInt16>.stride, named: "router hidden")
        Self.requireRange(indices, offset: indicesOffset, bytes: Self.routeIndexBytes,
                          alignment: MemoryLayout<UInt32>.stride, named: "router indices")
        Self.requireRange(routingWeights, offset: routingWeightsOffset,
                          bytes: Self.routeWeightBytes, alignment: MemoryLayout<Float>.stride,
                          named: "router weights output")
        Self.requireDisjoint(indices, indicesOffset, Self.routeIndexBytes,
                             routingWeights, routingWeightsOffset, Self.routeWeightBytes,
                             left: "router indices", right: "router weights output")
        Self.requireDisjoint(indices, indicesOffset, Self.routeIndexBytes,
                             weights, weightsOffset, Self.routerWeightBytes,
                             left: "router indices", right: "router weights")
        Self.requireDisjoint(routingWeights, routingWeightsOffset, Self.routeWeightBytes,
                             weights, weightsOffset, Self.routerWeightBytes,
                             left: "router weights output", right: "router weights")
        Self.requireDisjoint(indices, indicesOffset, Self.routeIndexBytes,
                             hidden, hiddenOffset, Self.vectorBytes,
                             left: "router indices", right: "router hidden")
        Self.requireDisjoint(routingWeights, routingWeightsOffset, Self.routeWeightBytes,
                             hidden, hiddenOffset, Self.vectorBytes,
                             left: "router weights output", right: "router hidden")

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        reserveRouterLogits(for: commandBuffer)
        encoder.setComputePipelineState(router)
        encoder.setBuffer(weights, offset: weightsOffset, index: 0)
        encoder.setBuffer(hidden, offset: hiddenOffset, index: 1)
        encoder.setBuffer(logits, offset: 0, index: 2)
        encoder.dispatchThreadgroups(
            MTLSize(width: Self.expertCount / 32, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()

        guard let selectEncoder = commandBuffer.makeComputeCommandEncoder() else { return }
        selectEncoder.setComputePipelineState(select)
        selectEncoder.setBuffer(logits, offset: 0, index: 0)
        selectEncoder.setBuffer(indices, offset: indicesOffset, index: 1)
        selectEncoder.setBuffer(routingWeights, offset: routingWeightsOffset, index: 2)
        selectEncoder.dispatchThreadgroups(
            MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        selectEncoder.endEncoding()
    }

    /// Binds exactly eight expert blobs in descending router-rank order.
    func makeRoutedArgumentBuffer(routedBlobs: [RoutedBlob], offsets: MoEExpertOffsets) -> MTLBuffer {
        requireArgumentTableAvailableForRebind()
        Self.validateBlobs(routedBlobs, offsets: offsets)
        routedArgumentEncoder.setArgumentBuffer(routedArgumentBuffer, offset: 0)
        for (rank, blob) in routedBlobs.enumerated() {
            routedArgumentEncoder.setBuffer(blob.buffer, offset: blob.offset, index: rank)
        }
        boundBlobs = routedBlobs
        boundOffsets = offsets
        return routedArgumentBuffer
    }

    func encodePhase1(commandBuffer: MTLCommandBuffer,
                      routedArgumentBuffer: MTLBuffer,
                      routedBlobs: [RoutedBlob], offsets: MoEExpertOffsets,
                      x: MTLBuffer, xOffset: Int = 0,
                      acts: MTLBuffer, actsOffset: Int = 0) {
        requireOwnedArgumentBuffer(routedArgumentBuffer)
        requireCurrentBinding(routedBlobs, offsets: offsets)
        Self.validatePhase1Inputs(routedArgumentBuffer: routedArgumentBuffer,
                                  routedArgumentBytes: routedArgumentBytes,
                                  routedBlobs: routedBlobs, offsets: offsets,
                                  x: x, xOffset: xOffset, acts: acts, actsOffset: actsOffset)
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        markArgumentTableRead(by: commandBuffer)
        encoder.setComputePipelineState(phase1)
        encoder.setBuffer(routedArgumentBuffer, offset: 0, index: 0)
        for blob in routedBlobs { encoder.useResource(blob.buffer, usage: .read) }
        var offsets = offsets
        encoder.setBytes(&offsets, length: MemoryLayout<MoEExpertOffsets>.stride, index: 1)
        encoder.setBuffer(x, offset: xOffset, index: 2)
        encoder.setBuffer(acts, offset: actsOffset, index: 3)
        encoder.dispatchThreadgroups(
            MTLSize(width: Self.topK * Self.intermediate / 8, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        encoder.endEncoding()
    }

    /// Writes each subset result into its original router-rank act slot.
    func encodePhase1Subset(commandBuffer: MTLCommandBuffer,
                            routedArgumentBuffer: MTLBuffer,
                            routedBlobs: [RoutedBlob], offsets: MoEExpertOffsets,
                            x: MTLBuffer, xOffset: Int = 0,
                            acts: MTLBuffer, actsOffset: Int = 0,
                            activeSlotIndices: [UInt32]) {
        guard !activeSlotIndices.isEmpty else { return }
        precondition(activeSlotIndices.count <= Self.topK,
                     "Maple MoE accepts at most eight active ranks")
        precondition(activeSlotIndices.allSatisfy { $0 < UInt32(Self.topK) },
                     "Maple MoE active rank is out of range")
        precondition(Set(activeSlotIndices).count == activeSlotIndices.count,
                     "Maple MoE active ranks must be unique")
        requireOwnedArgumentBuffer(routedArgumentBuffer)
        requireCurrentBinding(routedBlobs, offsets: offsets)
        Self.validatePhase1Inputs(routedArgumentBuffer: routedArgumentBuffer,
                                  routedArgumentBytes: routedArgumentBytes,
                                  routedBlobs: routedBlobs, offsets: offsets,
                                  x: x, xOffset: xOffset, acts: acts, actsOffset: actsOffset)

        var activeCount = UInt32(activeSlotIndices.count)
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        markArgumentTableRead(by: commandBuffer)
        encoder.setComputePipelineState(phase1Subset)
        encoder.setBuffer(routedArgumentBuffer, offset: 0, index: 0)
        for rank in activeSlotIndices { encoder.useResource(routedBlobs[Int(rank)].buffer, usage: .read) }
        var offsets = offsets
        encoder.setBytes(&offsets, length: MemoryLayout<MoEExpertOffsets>.stride, index: 1)
        encoder.setBuffer(x, offset: xOffset, index: 2)
        encoder.setBuffer(acts, offset: actsOffset, index: 3)
        activeSlotIndices.withUnsafeBytes { bytes in
            encoder.setBytes(bytes.baseAddress!, length: bytes.count, index: 4)
        }
        encoder.setBytes(&activeCount, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.dispatchThreadgroups(
            MTLSize(width: activeSlotIndices.count * Self.intermediate / 8, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        encoder.endEncoding()
    }

    /// Emits the routed expert sum only. The outer Maple layer owns residual addition.
    func encodePhase2(commandBuffer: MTLCommandBuffer,
                      routedArgumentBuffer: MTLBuffer,
                      routedBlobs: [RoutedBlob], offsets: MoEExpertOffsets,
                      acts: MTLBuffer, actsOffset: Int = 0,
                      routingWeights: MTLBuffer, routingWeightsOffset: Int = 0,
                      output: MTLBuffer, outputOffset: Int = 0) {
        requireOwnedArgumentBuffer(routedArgumentBuffer)
        requireCurrentBinding(routedBlobs, offsets: offsets)
        Self.validateBlobs(routedBlobs, offsets: offsets)
        Self.requireRange(routedArgumentBuffer, offset: 0, bytes: routedArgumentBytes,
                          alignment: 1, named: "routed argument table")
        Self.requireRange(acts, offset: actsOffset, bytes: Self.actsBytes,
                          alignment: MemoryLayout<UInt16>.stride, named: "acts")
        Self.requireRange(routingWeights, offset: routingWeightsOffset,
                          bytes: Self.routeWeightBytes, alignment: MemoryLayout<Float>.stride,
                          named: "routing weights")
        Self.requireRange(output, offset: outputOffset, bytes: Self.vectorBytes,
                          alignment: MemoryLayout<UInt16>.stride, named: "output")
        Self.requirePhase2Disjoint(output, outputOffset, routedArgumentBuffer, routedArgumentBytes,
                                   routedBlobs, acts, actsOffset,
                                   routingWeights, routingWeightsOffset)

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        markArgumentTableRead(by: commandBuffer)
        encoder.setComputePipelineState(phase2)
        encoder.setBuffer(routedArgumentBuffer, offset: 0, index: 0)
        for blob in routedBlobs { encoder.useResource(blob.buffer, usage: .read) }
        var offsets = offsets
        encoder.setBytes(&offsets, length: MemoryLayout<MoEExpertOffsets>.stride, index: 1)
        encoder.setBuffer(acts, offset: actsOffset, index: 2)
        encoder.setBuffer(routingWeights, offset: routingWeightsOffset, index: 3)
        encoder.setBuffer(output, offset: outputOffset, index: 4)
        encoder.dispatchThreadgroups(
            MTLSize(width: Self.dimension, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
    }

    private static func pipeline(_ function: MTLFunction, device: MTLDevice,
                                 threads: Int) throws -> MTLComputePipelineState {
        let descriptor = MTLComputePipelineDescriptor()
        descriptor.computeFunction = function
        descriptor.maxTotalThreadsPerThreadgroup = threads
        descriptor.threadGroupSizeIsMultipleOfThreadExecutionWidth = true
        return try device.makeComputePipelineState(descriptor: descriptor,
                                                   options: [], reflection: nil)
    }

    private func requireOwnedArgumentBuffer(_ buffer: MTLBuffer) {
        precondition(buffer === routedArgumentBuffer,
                     "Maple MoE argument table must be created by this instance")
    }

    private func requireArgumentTableAvailableForRebind() {
        pruneTerminalArgumentUsers()
        precondition(routedArgumentUsers.isEmpty,
                     "Maple MoE argument table is still in use by an encoded command buffer")
    }

    private func markArgumentTableRead(by commandBuffer: MTLCommandBuffer) {
        pruneTerminalArgumentUsers()
        if !routedArgumentUsers.contains(where: { $0 === commandBuffer }) {
            routedArgumentUsers.append(commandBuffer)
        }
    }

    private func pruneTerminalArgumentUsers() {
        routedArgumentUsers.removeAll { Self.isTerminal($0) }
    }

    private func reserveRouterLogits(for commandBuffer: MTLCommandBuffer) {
        if let routerLogitsUser, Self.isTerminal(routerLogitsUser) {
            self.routerLogitsUser = nil
        }
        if let routerLogitsUser {
            precondition(routerLogitsUser === commandBuffer,
                         "Maple MoE router logits are still in use by another command buffer")
        } else {
            routerLogitsUser = commandBuffer
        }
    }

    private static func isTerminal(_ commandBuffer: MTLCommandBuffer) -> Bool {
        commandBuffer.status == .completed || commandBuffer.status == .error
    }

    private func requireCurrentBinding(_ blobs: [RoutedBlob], offsets: MoEExpertOffsets) {
        guard let boundBlobs, let boundOffsets else {
            preconditionFailure("Maple MoE argument table has not been bound")
        }
        precondition(boundBlobs.count == blobs.count &&
                     zip(boundBlobs, blobs).allSatisfy {
                         $0.buffer === $1.buffer && $0.offset == $1.offset && $0.length == $1.length
                     },
                     "Maple MoE blobs differ from the current argument table")
        precondition(Self.offsetsEqual(boundOffsets, offsets),
                     "Maple MoE offsets differ from the current argument table")
    }

    private static func offsetsEqual(_ lhs: MoEExpertOffsets, _ rhs: MoEExpertOffsets) -> Bool {
        lhs.gateWOff == rhs.gateWOff && lhs.gateSOff == rhs.gateSOff &&
        lhs.gateBOff == rhs.gateBOff && lhs.upWOff == rhs.upWOff &&
        lhs.upSOff == rhs.upSOff && lhs.upBOff == rhs.upBOff &&
        lhs.downWOff == rhs.downWOff && lhs.downSOff == rhs.downSOff &&
        lhs.downBOff == rhs.downBOff
    }

    private static func validatePhase1Inputs(routedArgumentBuffer: MTLBuffer,
                                             routedArgumentBytes: Int,
                                             routedBlobs: [RoutedBlob], offsets: MoEExpertOffsets,
                                             x: MTLBuffer, xOffset: Int,
                                             acts: MTLBuffer, actsOffset: Int) {
        validateBlobs(routedBlobs, offsets: offsets)
        requireRange(routedArgumentBuffer, offset: 0, bytes: routedArgumentBytes,
                     alignment: 1, named: "routed argument table")
        requireRange(x, offset: xOffset, bytes: vectorBytes,
                     alignment: MemoryLayout<UInt16>.stride, named: "expert input")
        requireRange(acts, offset: actsOffset, bytes: actsBytes,
                     alignment: MemoryLayout<UInt16>.stride, named: "acts")
        requireDisjoint(acts, actsOffset, actsBytes, x, xOffset, vectorBytes,
                        left: "acts", right: "expert input")
        requireDisjoint(acts, actsOffset, actsBytes,
                        routedArgumentBuffer, 0, routedArgumentBytes,
                        left: "acts", right: "routed argument table")
        for blob in routedBlobs {
            requireDisjoint(acts, actsOffset, actsBytes, blob.buffer, blob.offset, blob.length,
                            left: "acts", right: "routed expert")
        }
    }

    private static func validateBlobs(_ routedBlobs: [RoutedBlob], offsets: MoEExpertOffsets) {
        precondition(routedBlobs.count == topK, "Maple MoE requires eight routed blobs")
        for left in routedBlobs.indices {
            for right in routedBlobs.indices where right > left {
                precondition(routedBlobs[left].buffer !== routedBlobs[right].buffer ||
                             routedBlobs[left].offset != routedBlobs[right].offset,
                             "Maple MoE routed blobs must be distinct rank slots")
            }
        }
        let ranges: [(Int, Int, String)] = [
            (Int(offsets.gateWOff), projectionWeightBytes, "gate weights"),
            (Int(offsets.gateSOff), projectionCompanionBytes, "gate scales"),
            (Int(offsets.gateBOff), projectionCompanionBytes, "gate biases"),
            (Int(offsets.upWOff), projectionWeightBytes, "up weights"),
            (Int(offsets.upSOff), projectionCompanionBytes, "up scales"),
            (Int(offsets.upBOff), projectionCompanionBytes, "up biases"),
            (Int(offsets.downWOff), downWeightBytes, "down weights"),
            (Int(offsets.downSOff), downCompanionBytes, "down scales"),
            (Int(offsets.downBOff), downCompanionBytes, "down biases"),
        ]
        for blob in routedBlobs {
            precondition(blob.offset >= 0 && blob.length >= 0 &&
                         blob.offset <= blob.buffer.length &&
                         blob.length <= blob.buffer.length - blob.offset,
                         "Maple MoE routed blob slice exceeds its buffer")
            for (offset, bytes, name) in ranges {
                requireSliceRange(blob, offset: offset, bytes: bytes,
                                  alignment: name.hasSuffix("weights") ? 1 : MemoryLayout<UInt16>.stride,
                                  named: name)
            }
            for left in ranges.indices {
                for right in ranges.indices where right > left {
                    let a = ranges[left]
                    let b = ranges[right]
                    precondition(a.0 + a.1 <= b.0 || b.0 + b.1 <= a.0,
                                 "Maple MoE expert tensor ranges overlap")
                }
            }
        }
    }

    private static func requirePhase2Disjoint(_ output: MTLBuffer, _ outputOffset: Int,
                                              _ argument: MTLBuffer, _ argumentBytes: Int,
                                              _ blobs: [RoutedBlob],
                                              _ acts: MTLBuffer, _ actsOffset: Int,
                                              _ weights: MTLBuffer, _ weightsOffset: Int) {
        requireDisjoint(output, outputOffset, vectorBytes, acts, actsOffset, actsBytes,
                        left: "output", right: "acts")
        requireDisjoint(output, outputOffset, vectorBytes, weights, weightsOffset, routeWeightBytes,
                        left: "output", right: "routing weights")
        requireDisjoint(output, outputOffset, vectorBytes, argument, 0, argumentBytes,
                        left: "output", right: "routed argument table")
        for blob in blobs {
            requireDisjoint(output, outputOffset, vectorBytes, blob.buffer, blob.offset, blob.length,
                            left: "output", right: "routed expert")
        }
    }

    /// Validates a tensor range relative to a blob slice. Alignment applies to
    /// the absolute buffer offset the kernel dereferences.
    private static func requireSliceRange(_ blob: RoutedBlob, offset: Int, bytes: Int,
                                          alignment: Int, named: String) {
        precondition(offset >= 0 && offset <= blob.length && bytes <= blob.length - offset,
                     "Maple \(named) buffer is too small")
        precondition((blob.offset + offset).isMultiple(of: alignment),
                     "Maple \(named) offset is misaligned")
    }

    private static func requireRange(_ buffer: MTLBuffer, offset: Int, bytes: Int,
                                     alignment: Int, named: String) {
        precondition(offset >= 0 && offset <= buffer.length && bytes <= buffer.length - offset,
                     "Maple \(named) buffer is too small")
        precondition(offset.isMultiple(of: alignment), "Maple \(named) offset is misaligned")
    }

    private static func requireDisjoint(_ leftBuffer: MTLBuffer, _ leftOffset: Int, _ leftBytes: Int,
                                        _ rightBuffer: MTLBuffer, _ rightOffset: Int, _ rightBytes: Int,
                                        left: String, right: String) {
        guard leftBuffer === rightBuffer else { return }
        precondition(leftOffset + leftBytes <= rightOffset || rightOffset + rightBytes <= leftOffset,
                     "Maple \(left) and \(right) ranges must not overlap")
    }
}

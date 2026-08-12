import Foundation

public enum RangeCopyTransform: Sendable, Equatable {
    case identity
    case unpackInt2ToInt4
    case repeatBF16(count: Int, negated: Bool)

    var inputUnitBytes: UInt64 {
        switch self {
        case .identity: 1
        case .unpackInt2ToInt4: 4
        case .repeatBF16: 2
        }
    }

    func destinationByteCount(for sourceByteCount: UInt64) throws -> UInt64 {
        let multiplier: UInt64
        switch self {
        case .identity:
            multiplier = 1
        case .unpackInt2ToInt4:
            multiplier = 2
        case .repeatBF16(let count, _):
            guard count > 0, let value = UInt64(exactly: count) else {
                throw RepackError.configurationInvalid(
                    detail: "repeat-bf16 count must be positive")
            }
            multiplier = value
        }
        let (result, overflow) = sourceByteCount.multipliedReportingOverflow(by: multiplier)
        guard !overflow else {
            throw RepackError.configurationInvalid(
                detail: "range transform output size overflows UInt64")
        }
        return result
    }

    var fingerprintDescription: String {
        switch self {
        case .identity: "identity"
        case .unpackInt2ToInt4: "unpack-int2-to-int4"
        case .repeatBF16(let count, let negated):
            "repeat-bf16:\(count):\(negated ? 1 : 0)"
        }
    }
}

public struct RangeCopy: Sendable, Equatable {
    public let shardID: String
    public let sourceOffset: UInt64
    public let size: UInt64
    public let destinationPath: String
    public let destinationOffset: UInt64
    public let transform: RangeCopyTransform

    public init(shardID: String,
                sourceOffset: UInt64,
                size: UInt64,
                destinationPath: String,
                destinationOffset: UInt64,
                transform: RangeCopyTransform = .identity) {
        self.shardID = shardID
        self.sourceOffset = sourceOffset
        self.size = size
        self.destinationPath = destinationPath
        self.destinationOffset = destinationOffset
        self.transform = transform
    }

    func destinationByteCount() throws -> UInt64 {
        try transform.destinationByteCount(for: size)
    }
}

public struct CoalescedRangeCopy: Sendable, Equatable {
    public let id: String
    public let shardID: String
    public let sourceOffset: UInt64
    public let size: UInt64
    public let destinations: [RangeCopy]

    func destinationByteCount() throws -> UInt64 {
        try destinations.reduce(UInt64(0)) { total, destination in
            let (next, overflow) = total.addingReportingOverflow(
                try destination.destinationByteCount())
            guard !overflow else {
                throw RepackError.configurationInvalid(
                    detail: "coalesced destination size overflows UInt64")
            }
            return next
        }
    }
}

public struct RemoteExpectedOutput: Codable, Sendable, Equatable {
    public let relativePath: String
    public let size: UInt64
}

public struct RangeCopyPlan: Sendable {
    public let scalarCopies: [RangeCopy]
    public let coalescedCopies: [CoalescedRangeCopy]
    public let remoteBytesToDownload: UInt64
    public let remoteGapBytesDownloaded: UInt64
    public let canonicalFingerprint: String
    public let residentIndexSha256: String
    public let expectedOutputs: [RemoteExpectedOutput]
}

public enum RangeCopyPlanner {
    static func plan(repackPlan: RepackPlan,
                     rangeChunkBytes: Int,
                     layoutMode: String = "identity",
                     layoutOrderSha256: String? = nil) throws -> RangeCopyPlan {
        var copies: [RangeCopy] = []
        copies.reserveCapacity(repackPlan.resident.entries.count * 3)

        func appendCopy(shardID: String,
                        sourceOffset: UInt64,
                        sourceSize: UInt64,
                        destinationPath: String,
                        destinationOffset: UInt64,
                        destinationSize: UInt64,
                        destinationLimit: UInt64,
                        transform: RangeCopyTransform) throws {
            let copy = RangeCopy(
                shardID: shardID,
                sourceOffset: sourceOffset,
                size: sourceSize,
                destinationPath: destinationPath,
                destinationOffset: destinationOffset,
                transform: transform)
            guard try copy.destinationByteCount() == destinationSize else {
                throw RepackError.configurationInvalid(
                    detail: "range transform output size does not match its planned destination")
            }
            let destinationEnd = try checkedEnd(
                offset: destinationOffset,
                size: destinationSize,
                detail: "planned destination range overflows UInt64")
            guard destinationEnd <= destinationLimit else {
                throw RepackError.configurationInvalid(
                    detail: "range transform output exceeds its planned file")
            }
            copies.append(copy)
        }

        for entry in repackPlan.resident.entries {
            try appendCopy(
                shardID: entry.sourceWeight.shardPath,
                sourceOffset: entry.sourceWeight.absoluteOffset,
                sourceSize: entry.sourceWeight.sizeBytes,
                destinationPath: entry.fileOffsetPath(in: repackPlan.resident),
                destinationOffset: entry.fileOffset,
                destinationSize: entry.sizeBytes,
                destinationLimit: repackPlan.resident.totalSize,
                transform: entry.weightTransform)
            if let scales = entry.sourceScales {
                try appendCopy(
                    shardID: scales.shardPath,
                    sourceOffset: scales.absoluteOffset,
                    sourceSize: scales.sizeBytes,
                    destinationPath: repackPlan.resident.path,
                    destinationOffset: entry.scaleOffset,
                    destinationSize: entry.scaleSize,
                    destinationLimit: repackPlan.resident.totalSize,
                    transform: entry.scaleTransform)
            }
            if let biases = entry.sourceBiases {
                try appendCopy(
                    shardID: biases.shardPath,
                    sourceOffset: biases.absoluteOffset,
                    sourceSize: biases.sizeBytes,
                    destinationPath: repackPlan.resident.path,
                    destinationOffset: entry.biasOffset,
                    destinationSize: entry.biasSize,
                    destinationLimit: repackPlan.resident.totalSize,
                    transform: entry.biasTransform)
            }
        }

        for layer in repackPlan.layers where layer.expertsPerLayer > 0 {
            for expert in 0..<layer.expertsPerLayer {
                let blobBase = try checkedProduct(
                    UInt64(layer.physicalRank(for: expert)),
                    layer.expertStride,
                    detail: "expert destination offset overflows UInt64")
                for slice in layer.subTensors {
                    let sourceStride = try checkedProduct(
                        UInt64(expert),
                        slice.sourceOffsetPerExpert,
                        detail: "expert source offset overflows UInt64")
                    try appendCopy(
                        shardID: slice.sourceTensor.shardPath,
                        sourceOffset: checkedEnd(
                            offset: slice.sourceTensor.absoluteOffset,
                            size: sourceStride,
                            detail: "expert source offset overflows UInt64"),
                        sourceSize: slice.sourceOffsetPerExpert,
                        destinationPath: layer.path,
                        destinationOffset: checkedEnd(
                            offset: blobBase,
                            size: slice.offsetInExpertBlob,
                            detail: "expert destination offset overflows UInt64"),
                        destinationSize: slice.sizeInExpertBlob,
                        destinationLimit: layer.fileSize,
                        transform: slice.transform)
                }
            }
        }

        try validateDestinationIntervals(copies, outputRoot: outputRoot(for: repackPlan))
        let coalesced = try coalesce(copies: copies, rangeChunkBytes: rangeChunkBytes)
        let indexData = try ResidentWriter.encodeIndex(plan: repackPlan.resident)
        let indexSha = hashData(indexData)
        let expectedOutputs = expectedOutputList(for: repackPlan)
        let fingerprint = try canonicalFingerprint(
            copies: coalesced,
            outputRoot: outputRoot(for: repackPlan),
            rangeChunkBytes: rangeChunkBytes,
            layoutMode: layoutMode,
            layoutOrderSha256: layoutOrderSha256,
            residentIndexSha256: indexSha,
            expectedOutputs: expectedOutputs)
        let downloaded = try checkedSum(
            coalesced.map(\.size),
            detail: "remote range byte total overflows UInt64")
        let copied = try sourceUnionBytes(copies)
        guard downloaded >= copied else {
            throw RepackError.configurationInvalid(
                detail: "coalesced source ranges are smaller than their source union")
        }
        return RangeCopyPlan(scalarCopies: copies,
                             coalescedCopies: coalesced,
                             remoteBytesToDownload: downloaded,
                             remoteGapBytesDownloaded: downloaded - copied,
                             canonicalFingerprint: fingerprint,
                             residentIndexSha256: indexSha,
                             expectedOutputs: expectedOutputs)
    }

    public static func coalesce(copies: [RangeCopy],
                                rangeChunkBytes: Int) throws -> [CoalescedRangeCopy] {
        guard rangeChunkBytes > 0 else {
            throw RepackError.configurationInvalid(detail: "rangeChunkBytes must be positive")
        }
        let sorted = try splitLargeCopies(copies, rangeChunkBytes: rangeChunkBytes).sorted {
            if $0.shardID != $1.shardID { return $0.shardID < $1.shardID }
            return $0.sourceOffset < $1.sourceOffset
        }
        var out: [CoalescedRangeCopy] = []
        var currentShard: String?
        var currentStart: UInt64 = 0
        var currentEnd: UInt64 = 0
        var currentDestinations: [RangeCopy] = []

        func flush() {
            guard let shard = currentShard else { return }
            out.append(CoalescedRangeCopy(id: "",
                                          shardID: shard,
                                          sourceOffset: currentStart,
                                          size: currentEnd - currentStart,
                                          destinations: currentDestinations))
        }

        for copy in sorted where copy.size > 0 {
            let copyEnd = try checkedEnd(
                offset: copy.sourceOffset,
                size: copy.size,
                detail: "source range overflows UInt64")
            if currentShard == nil {
                currentShard = copy.shardID
                currentStart = copy.sourceOffset
                currentEnd = copyEnd
                currentDestinations = [copy]
                continue
            }
            let proposedStart = currentStart
            let proposedEnd = max(currentEnd, copyEnd)
            let canMerge = currentShard == copy.shardID
                && proposedEnd >= proposedStart
                && proposedEnd - proposedStart <= UInt64(rangeChunkBytes)
            if canMerge {
                currentEnd = proposedEnd
                currentDestinations.append(copy)
            } else {
                flush()
                currentShard = copy.shardID
                currentStart = copy.sourceOffset
                currentEnd = copyEnd
                currentDestinations = [copy]
            }
        }
        flush()
        return out.enumerated().map { index, copy in
            CoalescedRangeCopy(
                id: String(format: "range-%08d", index),
                shardID: copy.shardID,
                sourceOffset: copy.sourceOffset,
                size: copy.size,
                destinations: copy.destinations.sorted(by: destinationOrder))
        }
    }

    private static func splitLargeCopies(_ copies: [RangeCopy],
                                         rangeChunkBytes: Int) throws -> [RangeCopy] {
        let limit = UInt64(rangeChunkBytes)
        var out: [RangeCopy] = []
        for copy in copies {
            let unit = copy.transform.inputUnitBytes
            guard copy.size > 0, copy.size % unit == 0, limit >= unit else {
                throw RepackError.configurationInvalid(
                    detail: "range transform \(copy.transform.fingerprintDescription) "
                        + "requires nonempty \(unit)-byte aligned source chunks")
            }
            _ = try copy.destinationByteCount()
            let alignedLimit = limit - limit % unit
            var remaining = copy.size
            var src = copy.sourceOffset
            var dst = copy.destinationOffset
            while remaining > 0 {
                let n = min(remaining, alignedLimit)
                out.append(RangeCopy(shardID: copy.shardID,
                                     sourceOffset: src,
                                     size: n,
                                     destinationPath: copy.destinationPath,
                                     destinationOffset: dst,
                                     transform: copy.transform))
                remaining -= n
                src = try checkedEnd(
                    offset: src,
                    size: n,
                    detail: "split source range overflows UInt64")
                dst = try checkedEnd(
                    offset: dst,
                    size: copy.transform.destinationByteCount(for: n),
                    detail: "split destination range overflows UInt64")
            }
        }
        return out
    }

    private static func outputRoot(for plan: RepackPlan) -> String {
        (plan.resident.path as NSString).deletingLastPathComponent
    }

    private static func checkedEnd(offset: UInt64,
                                   size: UInt64,
                                   detail: String) throws -> UInt64 {
        let (end, overflow) = offset.addingReportingOverflow(size)
        guard !overflow else {
            throw RepackError.configurationInvalid(detail: detail)
        }
        return end
    }

    private static func checkedProduct(_ lhs: UInt64,
                                       _ rhs: UInt64,
                                       detail: String) throws -> UInt64 {
        let (product, overflow) = lhs.multipliedReportingOverflow(by: rhs)
        guard !overflow else {
            throw RepackError.configurationInvalid(detail: detail)
        }
        return product
    }

    private static func checkedSum(_ values: [UInt64],
                                   detail: String) throws -> UInt64 {
        try values.reduce(UInt64(0)) { total, value in
            try checkedEnd(offset: total, size: value, detail: detail)
        }
    }

    private static func sourceUnionBytes(_ copies: [RangeCopy]) throws -> UInt64 {
        let sorted = copies.filter { $0.size > 0 }.sorted {
            if $0.shardID != $1.shardID { return $0.shardID < $1.shardID }
            return $0.sourceOffset < $1.sourceOffset
        }
        var total: UInt64 = 0
        var shard: String?
        var start: UInt64 = 0
        var end: UInt64 = 0
        for copy in sorted {
            let copyEnd = try checkedEnd(
                offset: copy.sourceOffset,
                size: copy.size,
                detail: "source range overflows UInt64")
            if shard != copy.shardID || copy.sourceOffset > end {
                if shard != nil {
                    total = try checkedEnd(
                        offset: total,
                        size: end - start,
                        detail: "source union byte total overflows UInt64")
                }
                shard = copy.shardID
                start = copy.sourceOffset
                end = copyEnd
            } else {
                end = max(end, copyEnd)
            }
        }
        if shard != nil {
            total = try checkedEnd(
                offset: total,
                size: end - start,
                detail: "source union byte total overflows UInt64")
        }
        return total
    }

    private static func expectedOutputList(for plan: RepackPlan) -> [RemoteExpectedOutput] {
        var outputs = [
            RemoteExpectedOutput(relativePath: "model_weights.bin",
                                 size: plan.resident.totalSize)
        ]
        outputs.append(contentsOf: plan.layers
            .filter { $0.expertsPerLayer > 0 }
            .map {
                RemoteExpectedOutput(
                    relativePath: "packed_experts/" + ($0.path as NSString).lastPathComponent,
                    size: $0.fileSize)
            })
        return outputs.sorted { $0.relativePath < $1.relativePath }
    }

    static func validateDestinationIntervals(_ copies: [RangeCopy],
                                             outputRoot: String) throws {
        let sorted = try copies.map { copy in
            (try normalizedRelativePath(copy.destinationPath, root: outputRoot), copy)
        }.sorted {
            if $0.0 != $1.0 { return $0.0 < $1.0 }
            return $0.1.destinationOffset < $1.1.destinationOffset
        }
        var previousPath: String?
        var previousEnd: UInt64 = 0
        for (path, copy) in sorted {
            let destinationEnd = try checkedEnd(
                offset: copy.destinationOffset,
                size: copy.destinationByteCount(),
                detail: "destination range overflows \(path)")
            if previousPath == path, copy.destinationOffset < previousEnd {
                throw RepackError.configurationInvalid(
                    detail: "overlapping destination ranges in \(path)")
            }
            previousPath = path
            previousEnd = destinationEnd
        }
    }

    static func canonicalFingerprint(
        copies: [CoalescedRangeCopy],
        outputRoot: String,
        rangeChunkBytes: Int,
        layoutMode: String,
        layoutOrderSha256: String?,
        residentIndexSha256: String,
        expectedOutputs: [RemoteExpectedOutput]
    ) throws -> String {
        let usesTransforms = copies.contains { copy in
            copy.destinations.contains { $0.transform != .identity }
        }
        var writer = FingerprintWriter(domain: usesTransforms
            ? "Mference.RemoteInstallPlan.v2"
            : "Mference.RemoteInstallPlan.v1")
        writer.append(UInt64(GTurboJSON.versionMajor))
        writer.append(UInt64(GTurboJSON.versionMinor))
        writer.append(UInt64(rangeChunkBytes))
        writer.append(layoutMode)
        writer.append(layoutOrderSha256 ?? "")
        writer.append(residentIndexSha256)
        writer.append(UInt64(expectedOutputs.count))
        for output in expectedOutputs {
            writer.append(output.relativePath)
            writer.append(output.size)
        }
        writer.append(UInt64(copies.count))
        for copy in copies {
            writer.append(copy.id)
            writer.append(copy.shardID.precomposedStringWithCanonicalMapping)
            writer.append(copy.sourceOffset)
            writer.append(copy.size)
            writer.append(UInt64(copy.destinations.count))
            for destination in copy.destinations {
                writer.append(try normalizedRelativePath(
                    destination.destinationPath,
                    root: outputRoot))
                writer.append(destination.destinationOffset)
                guard destination.sourceOffset >= copy.sourceOffset else {
                    throw RepackError.configurationInvalid(
                        detail: "destination source offset precedes its coalesced range")
                }
                writer.append(destination.sourceOffset - copy.sourceOffset)
                writer.append(destination.size)
                if usesTransforms {
                    writer.append(try destination.destinationByteCount())
                    writer.append(destination.transform.fingerprintDescription)
                }
            }
        }
        return writer.finalize()
    }

    static func normalizedRelativePath(_ path: String, root: String) throws -> String {
        let normalizedRoot = root.hasSuffix("/") ? String(root.dropLast()) : root
        let normalizedPath = path
        let prefix = normalizedRoot.hasSuffix("/") ? normalizedRoot : normalizedRoot + "/"
        guard normalizedPath.hasPrefix(prefix) else {
            throw RepackError.configurationInvalid(
                detail: "destination \(normalizedPath) is outside partial output \(normalizedRoot)")
        }
        let relative = String(normalizedPath.dropFirst(prefix.count))
            .precomposedStringWithCanonicalMapping
        let components = relative.split(separator: "/", omittingEmptySubsequences: false)
        guard !relative.isEmpty,
              !relative.hasPrefix("/"),
              components.allSatisfy({ !$0.isEmpty && $0 != "." && $0 != ".." }) else {
            throw RepackError.configurationInvalid(
                detail: "invalid relative destination path \(relative)")
        }
        return relative
    }

    private static func destinationOrder(_ lhs: RangeCopy, _ rhs: RangeCopy) -> Bool {
        if lhs.destinationPath != rhs.destinationPath {
            return lhs.destinationPath < rhs.destinationPath
        }
        if lhs.destinationOffset != rhs.destinationOffset {
            return lhs.destinationOffset < rhs.destinationOffset
        }
        return lhs.sourceOffset < rhs.sourceOffset
    }
}

private extension ResidentEntry {
    func fileOffsetPath(in plan: ResidentFilePlan) -> String {
        plan.path
    }
}

private struct FingerprintWriter {
    private var stream = Sha256Stream()

    init(domain: String) {
        append(domain)
    }

    mutating func append(_ value: UInt64) {
        var littleEndian = value.littleEndian
        withUnsafeBytes(of: &littleEndian) { stream.update($0) }
    }

    mutating func append(_ value: String) {
        let data = Data(value.utf8)
        append(UInt64(data.count))
        data.withUnsafeBytes { stream.update($0) }
    }

    func finalize() -> String {
        stream.finalizeHexString()
    }
}

private func hashData(_ data: Data) -> String {
    var stream = Sha256Stream()
    data.withUnsafeBytes { stream.update($0) }
    return stream.finalizeHexString()
}

import Darwin
import Foundation

/// Stores a Gemma 4 QAT install's routed experts without their bias arrays.
///
/// The checkpoint is Q4_0 re-expressed as MLX affine INT4, so every group's
/// bias is exactly `-8 * scale` in BF16. The runtime rebuilds the arrays after
/// each read (`ExpertStorage` in the Mference target), which cuts every
/// routed-expert read from the SSD by about a tenth with identical results.
///
/// The input install is only read. The output is a new directory holding
/// copies (APFS clones) of the other files, rewritten layer files, a
/// `layout.json` with an `expertStorage` section, a manifest whose routed
/// bias type is `impliedNeg8Scale`, and a receipt bound to the output path.
/// The manifest is written last, so an interrupted run leaves an install the
/// runtime rejects as partial; a failed run removes its output. A group whose
/// bias is not exactly `-8 * scale`, or an input file whose bytes disagree
/// with the input manifest, aborts the conversion.
public enum QATImpliedBiasConverter {
    public static let modelID = "gemma-4-26b-a4b-it-qat-q4_0-mlx-aligned"
    public static let impliedBiasType = "impliedNeg8Scale"
    public static let storageFormat = "impliedNeg8ScaleBiases"

    public struct Result: Sendable {
        public let layers: Int
        public let expertStride: UInt64
        public let storedExpertStride: UInt64
        public let routedBytesBefore: UInt64
        public let routedBytesAfter: UInt64
        public let groupsChecked: UInt64
    }

    /// The BF16 bit pattern of `-8 * scale`; must equal the runtime's
    /// `ExpertStorage.neg8ScaleBits` for every input.
    @inline(__always)
    static func neg8ScaleBits(_ scale: UInt16) -> UInt16 {
        let product = Float(bitPattern: UInt32(scale) << 16) * -8
        return UInt16(truncatingIfNeeded: product.bitPattern >> 16)
    }

    struct Tensor {
        let name: String
        let offset: UInt64
        let size: UInt64
        let dtype: String
    }

    struct Plan {
        let segments: [(storedOffset: UInt64, memoryOffset: UInt64, size: UInt64)]
        let implied: [(biases: Tensor, scales: Tensor)]
        let storedExpertStride: UInt64
    }

    /// Stored segments are the expert's non-bias tensors, merged where they
    /// touch in memory and packed back to back; biases with a same-sized BF16
    /// `_scales` partner become implied.
    static func plan(tensors: [String: Tensor], expertStride: UInt64, pageSize: UInt64) throws -> Plan {
        var implied: [(biases: Tensor, scales: Tensor)] = []
        var stored: [Tensor] = []
        for tensor in tensors.values.sorted(by: { $0.offset < $1.offset }) {
            if tensor.name.hasSuffix("_biases") {
                let scalesName = String(tensor.name.dropLast("_biases".count)) + "_scales"
                guard let scales = tensors[scalesName], scales.size == tensor.size,
                      scales.dtype == "BF16", tensor.dtype == "BF16" else {
                    throw RepackError.configurationInvalid(
                        detail: "\(tensor.name) has no matching BF16 \(scalesName)")
                }
                implied.append((tensor, scales))
            } else {
                stored.append(tensor)
            }
        }
        guard !implied.isEmpty else {
            throw RepackError.configurationInvalid(detail: "routed experts carry no bias arrays")
        }
        var runs: [(memoryOffset: UInt64, size: UInt64)] = []
        for tensor in stored {
            if let last = runs.last, last.memoryOffset + last.size == tensor.offset {
                runs[runs.count - 1].size += tensor.size
            } else {
                runs.append((tensor.offset, tensor.size))
            }
        }
        var cursor: UInt64 = 0
        var segments: [(storedOffset: UInt64, memoryOffset: UInt64, size: UInt64)] = []
        for run in runs {
            guard run.memoryOffset + run.size <= expertStride else {
                throw RepackError.configurationInvalid(detail: "expert tensor exceeds expertStride")
            }
            segments.append((cursor, run.memoryOffset, run.size))
            cursor += run.size
        }
        for pair in implied {
            let bias = pair.biases
            guard !runs.contains(where: { bias.offset < $0.memoryOffset + $0.size
                                           && $0.memoryOffset < bias.offset + bias.size }) else {
                throw RepackError.configurationInvalid(detail: "\(bias.name) overlaps stored tensors")
            }
        }
        let stride = (cursor + pageSize - 1) / pageSize * pageSize
        return Plan(segments: segments, implied: implied, storedExpertStride: stride)
    }

    /// Tensors of the first routed expert in an encoded `layout.json`.
    static func referenceTensors(layout: [String: Any]) throws -> [String: Tensor] {
        guard let layers = layout["layers"] as? [[String: Any]],
              let first = layers.lazy.compactMap({ ($0["experts"] as? [[String: Any]])?.first }).first else {
            throw RepackError.configurationInvalid(detail: "layout.json has no routed experts")
        }
        return try tensors(first)
    }

    static func tensors(_ expert: [String: Any]) throws -> [String: Tensor] {
        guard let map = expert["tensors"] as? [String: [String: Any]] else {
            throw RepackError.configurationInvalid(detail: "layout.json expert without tensors")
        }
        var result: [String: Tensor] = [:]
        for (name, entry) in map {
            guard let offset = (entry["offset"] as? NSNumber)?.uint64Value,
                  let size = (entry["size"] as? NSNumber)?.uint64Value,
                  let dtype = entry["dtype"] as? String else {
                throw RepackError.configurationInvalid(detail: "layout.json malformed tensor \(name)")
            }
            result[name] = Tensor(name: name, offset: offset, size: size, dtype: dtype)
        }
        return result
    }

    /// Physical rank -> layout index for one layer, requiring every expert at
    /// `rank * expertStride` with the reference tensor layout.
    static func ranks(experts: [[String: Any]], expertsPerLayer: Int, expertStride: UInt64,
                      reference: [String: Tensor], relative: String) throws -> [Int] {
        let key = reference.mapValues { "\($0.offset)/\($0.size)/\($0.dtype)" }
        var byRank = [Int](repeating: -1, count: expertsPerLayer)
        guard experts.count == expertsPerLayer else {
            throw RepackError.configurationInvalid(detail: "\(relative) expert count mismatch")
        }
        for (index, expert) in experts.enumerated() {
            guard let offset = (expert["offset"] as? NSNumber)?.uint64Value,
                  (expert["size"] as? NSNumber)?.uint64Value == expertStride,
                  offset.isMultiple(of: expertStride), offset / expertStride < UInt64(expertsPerLayer),
                  byRank[Int(offset / expertStride)] == -1,
                  try tensors(expert).mapValues({ "\($0.offset)/\($0.size)/\($0.dtype)" }) == key else {
                throw RepackError.configurationInvalid(
                    detail: "\(relative) expert \(index) is misplaced or nonuniform")
            }
            byRank[Int(offset / expertStride)] = index
        }
        return byRank
    }

    /// Packs one layer file of experts stored at `rank * expertStride` into
    /// experts at `rank * storedExpertStride`, refusing any bias that is not
    /// `-8 * scale`. Returns the hashes of the bytes read and written.
    static func compactLayer(inputFD: Int32, inputPath: String, outputFD: Int32, outputPath: String,
                             expertsPerLayer: Int, expertStride: UInt64, plan: Plan,
                             describeExpert: (Int) -> String)
        throws -> (inputSha: String, outputSha: String, groups: UInt64) {
        let storedStride = plan.storedExpertStride
        let chunkExperts = 16
        let inputBuffer = UnsafeMutableRawBufferPointer.allocate(
            byteCount: chunkExperts * Int(expertStride), alignment: 16_384)
        let outputBuffer = UnsafeMutableRawBufferPointer.allocate(
            byteCount: chunkExperts * Int(storedStride), alignment: 16_384)
        let scratch = UnsafeMutableRawBufferPointer.allocate(
            byteCount: Int(plan.implied.map(\.biases.size).max() ?? 2), alignment: 16)
        defer {
            inputBuffer.deallocate()
            outputBuffer.deallocate()
            scratch.deallocate()
        }
        var inputHash = Sha256Stream()
        var outputHash = Sha256Stream()
        var groupsChecked: UInt64 = 0
        var rank = 0
        while rank < expertsPerLayer {
            let count = min(chunkExperts, expertsPerLayer - rank)
            let inputBytes = count * Int(expertStride)
            try Posix.preadAll(fd: inputFD, path: inputPath, buf: inputBuffer.baseAddress!,
                               count: inputBytes, offset: UInt64(rank) * expertStride)
            inputHash.update(UnsafeRawBufferPointer(rebasing: inputBuffer.prefix(inputBytes)))
            let outputBytes = count * Int(storedStride)
            memset(outputBuffer.baseAddress!, 0, outputBytes)
            for local in 0..<count {
                let source = UnsafeRawPointer(inputBuffer.baseAddress!.advanced(by: local * Int(expertStride)))
                for pair in plan.implied {
                    let groups = Int(pair.biases.size) / 2
                    for g in 0..<groups {
                        let scale = source.loadUnaligned(
                            fromByteOffset: Int(pair.scales.offset) + 2 * g, as: UInt16.self)
                        scratch.storeBytes(of: neg8ScaleBits(scale), toByteOffset: 2 * g, as: UInt16.self)
                    }
                    guard memcmp(scratch.baseAddress!, source.advanced(by: Int(pair.biases.offset)),
                                 Int(pair.biases.size)) == 0 else {
                        throw RepackError.configurationInvalid(detail:
                            "\(describeExpert(rank + local)) \(pair.biases.name): "
                            + "a bias is not exactly -8 * scale; the checkpoint cannot drop its biases")
                    }
                    groupsChecked += UInt64(groups)
                }
                let target = outputBuffer.baseAddress!.advanced(by: local * Int(storedStride))
                for segment in plan.segments {
                    memcpy(target.advanced(by: Int(segment.storedOffset)),
                           source.advanced(by: Int(segment.memoryOffset)), Int(segment.size))
                }
            }
            try Posix.pwriteAll(fd: outputFD, path: outputPath, buf: outputBuffer.baseAddress!,
                                count: outputBytes, offset: UInt64(rank) * storedStride)
            outputHash.update(UnsafeRawBufferPointer(rebasing: outputBuffer.prefix(outputBytes)))
            rank += count
        }
        try Posix.fsync(outputFD, path: outputPath)
        return (inputHash.finalizeHexString(), outputHash.finalizeHexString(), groupsChecked)
    }

    /// Replaces a layer file with its compact form through `<path>.compact`,
    /// so at most one extra layer's bytes exist on disk.
    static func compactLayerInPlace(path: String, relative: String, expertsPerLayer: Int,
                                    expertStride: UInt64, plan: Plan)
        throws -> (size: UInt64, sha: String, groups: UInt64) {
        let temporary = path + ".compact"
        if try Posix.entryKind(temporary) != .absent {
            try FileManager.default.removeItem(atPath: temporary)
        }
        let inputFD = try Posix.openRead(path)
        defer { close(inputFD) }
        guard try Posix.fileSize(fd: inputFD, path: path) == UInt64(expertsPerLayer) * expertStride else {
            throw RepackError.configurationInvalid(detail: "\(relative) has the wrong size to compact")
        }
        let outputFD = try Posix.openCreateRW(temporary)
        defer { close(outputFD) }
        let result = try compactLayer(inputFD: inputFD, inputPath: path, outputFD: outputFD,
                                      outputPath: temporary, expertsPerLayer: expertsPerLayer,
                                      expertStride: expertStride, plan: plan,
                                      describeExpert: { "\(relative) expert rank \($0)" })
        try Posix.rename(from: temporary, to: path)
        return (UInt64(expertsPerLayer) * plan.storedExpertStride, result.outputSha, result.groups)
    }

    /// `layout` with every expert moved to `rank * storedExpertStride` and an
    /// `expertStorage` section; tensor offsets keep describing the expanded
    /// expert the kernels read.
    static func compactLayout(_ layout: [String: Any], expertStride: UInt64, plan: Plan) throws -> [String: Any] {
        var result = layout
        guard var layers = layout["layers"] as? [[String: Any]] else {
            throw RepackError.configurationInvalid(detail: "layout.json has no layers")
        }
        for index in layers.indices {
            guard var experts = layers[index]["experts"] as? [[String: Any]] else { continue }
            for e in experts.indices {
                guard let offset = (experts[e]["offset"] as? NSNumber)?.uint64Value,
                      offset.isMultiple(of: expertStride) else {
                    throw RepackError.configurationInvalid(detail: "layout.json expert offset is misaligned")
                }
                experts[e]["offset"] = offset / expertStride * plan.storedExpertStride
                experts[e]["size"] = plan.storedExpertStride
            }
            layers[index]["experts"] = experts
        }
        result["layers"] = layers
        result["expertStorage"] = [
            "format": storageFormat,
            "storedExpertStride": plan.storedExpertStride,
            "segments": plan.segments.map {
                ["storedOffset": $0.storedOffset, "memoryOffset": $0.memoryOffset, "size": $0.size]
            },
            "impliedBiases": plan.implied.map { ["biases": $0.biases.name, "scales": $0.scales.name] },
        ] as [String: Any]
        return result
    }

    public static func run(inputGTurbo: String,
                           outputGTurbo: String,
                           pageSize: UInt64 = UInt64(getpagesize()),
                           progress: ((String) -> Void)? = nil) throws -> Result {
        let input = URL(fileURLWithPath: inputGTurbo).standardizedFileURL.path
        let output = URL(fileURLWithPath: outputGTurbo).standardizedFileURL.path
        guard try Posix.entryKind(output) == .absent else {
            throw RepackError.configurationInvalid(detail: "\(output) already exists")
        }
        let inputLock = try InstallLock.acquire(outputDirectory: input)
        let outputLock = try InstallLock.acquire(outputDirectory: output)
        defer { _ = (inputLock, outputLock) }

        func path(_ root: String, _ relative: String) -> String {
            (root as NSString).appendingPathComponent(relative)
        }
        func json(_ relative: String, limit: UInt64) throws -> [String: Any] {
            let data = try Posix.readBoundedData(path(input, relative), maximumBytes: limit)
            guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                throw RepackError.configurationInvalid(detail: "\(relative) is not a JSON object")
            }
            return object
        }

        // -- Input metadata.
        var manifest = try json("manifest.json", limit: 8 << 20)
        guard manifest["modelID"] as? String == modelID,
              var quant = manifest["quant"] as? [String: Any],
              var routedQuant = quant["routedExpert"] as? [String: Any],
              (routedQuant["biasType"] as? String)?.lowercased() == "bf16",
              var files = manifest["files"] as? [String: [String: Any]],
              let expertStride = (manifest["expertStride"] as? NSNumber)?.uint64Value else {
            throw RepackError.configurationInvalid(
                detail: "\(input) is not a Gemma 4 QAT install with explicit routed biases")
        }
        let layout = try json("packed_experts/layout.json", limit: VerifiedInstallTool.metadataMaxBytes)
        guard (layout["expertStride"] as? NSNumber)?.uint64Value == expertStride,
              let expertsPerLayer = layout["expertsPerLayer"] as? Int, expertsPerLayer > 0,
              let layers = layout["layers"] as? [[String: Any]], !layers.isEmpty,
              layout["expertStorage"] == nil else {
            throw RepackError.configurationInvalid(detail: "layout.json does not match the manifest")
        }
        let reference = try referenceTensors(layout: layout)
        let plan = try plan(tensors: reference, expertStride: expertStride, pageSize: pageSize)

        var provenance: (repoID: String, revision: String)?
        if let receipt = try? json(VerifiedInstallReceiptWriter.fileName, limit: 4 << 20),
           let repoID = receipt["sourceRepoID"] as? String,
           let revision = receipt["sourceRevision"] as? String,
           (receipt["manifestSha256"] as? String)?.lowercased()
               == (try Sha256Stream.hashFile(path: path(input, "manifest.json"))).lowercased() {
            provenance = (repoID, revision)
        }

        // -- Output: copies first, layers next, metadata last.
        try Posix.mkdirP(output)
        var succeeded = false
        defer {
            if !succeeded { try? FileManager.default.removeItem(atPath: output) }
        }
        let layerFiles = Set(layers.compactMap { $0["file"] as? String }.map { "packed_experts/" + $0 })
        for relative in files.keys.sorted()
            where !layerFiles.contains(relative) && relative != "packed_experts/layout.json" {
            let destination = path(output, relative)
            try Posix.mkdirP((destination as NSString).deletingLastPathComponent)
            try FileManager.default.copyItem(atPath: path(input, relative), toPath: destination)
        }
        try Posix.mkdirP(path(output, "packed_experts"))

        var groupsChecked: UInt64 = 0
        var routedBytesBefore: UInt64 = 0
        var routedBytesAfter: UInt64 = 0
        for (layerIndex, layer) in layers.enumerated() {
            guard let file = layer["file"] as? String,
                  let experts = layer["experts"] as? [[String: Any]] else {
                throw RepackError.configurationInvalid(detail: "layout.json malformed layer \(layerIndex)")
            }
            if experts.isEmpty { continue }
            let relative = "packed_experts/" + file
            guard let entry = files[relative],
                  let expectedSize = (entry["size"] as? NSNumber)?.uint64Value,
                  let expectedSha = entry["sha256"] as? String,
                  expectedSize == UInt64(expertsPerLayer) * expertStride else {
                throw RepackError.configurationInvalid(detail: "\(relative) does not match the layout")
            }
            let byRank = try ranks(experts: experts, expertsPerLayer: expertsPerLayer,
                                   expertStride: expertStride, reference: reference, relative: relative)
            let inputFD = try Posix.openRead(path(input, relative))
            defer { close(inputFD) }
            guard try Posix.fileSize(fd: inputFD, path: relative) == expectedSize else {
                throw RepackError.configurationInvalid(detail: "\(relative) has the wrong size")
            }
            let outputPath = path(output, relative)
            let outputFD = try Posix.openCreateRW(outputPath)
            defer { close(outputFD) }
            let result = try compactLayer(inputFD: inputFD, inputPath: relative, outputFD: outputFD,
                                          outputPath: outputPath, expertsPerLayer: expertsPerLayer,
                                          expertStride: expertStride, plan: plan,
                                          describeExpert: { "\(relative) expert \(byRank[$0])" })
            guard result.inputSha == expectedSha.lowercased() else {
                throw RepackError.configurationInvalid(
                    detail: "\(relative) does not match its SHA-256 in the input manifest")
            }
            let storedSize = UInt64(expertsPerLayer) * plan.storedExpertStride
            files[relative] = ["size": storedSize, "sha256": result.outputSha]
            groupsChecked += result.groups
            routedBytesBefore += expectedSize
            routedBytesAfter += storedSize
            progress?("\(relative): \(expectedSize) -> \(storedSize) bytes")
        }

        // -- layout.json, manifest.json, receipt.
        let layoutData = try JSONSerialization.data(
            withJSONObject: try compactLayout(layout, expertStride: expertStride, plan: plan),
            options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes])
        let layoutPath = path(output, "packed_experts/layout.json")
        try Posix.atomicWrite(layoutData, to: layoutPath, durableIn: path(output, "packed_experts"))
        files["packed_experts/layout.json"] = ["size": layoutData.count,
                                               "sha256": try Sha256Stream.hashFile(path: layoutPath)]

        routedQuant["biasType"] = impliedBiasType
        quant["routedExpert"] = routedQuant
        manifest["quant"] = quant
        manifest["files"] = files

        var receiptFiles: [RepackAudit.OutputFile] = []
        for (relative, entry) in files.sorted(by: { $0.key < $1.key }) {
            let actual = path(output, relative)
            let attributes = try FileManager.default.attributesOfItem(atPath: actual)
            let sha = try Sha256Stream.hashFile(path: actual, tileBytes: 8 << 20)
            guard let size = (attributes[.size] as? NSNumber)?.uint64Value,
                  size == (entry["size"] as? NSNumber)?.uint64Value,
                  sha == (entry["sha256"] as? String)?.lowercased() else {
                throw RepackError.configurationInvalid(detail: "\(relative) changed while converting")
            }
            receiptFiles.append(.init(relativePath: relative, size: size, sha256: sha))
        }
        let manifestData = try JSONSerialization.data(withJSONObject: manifest,
                                                      options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes])
        let manifestPath = path(output, "manifest.json")
        try Posix.atomicWrite(manifestData, to: manifestPath, durableIn: output)
        let manifestSha = try Sha256Stream.hashFile(path: manifestPath)
        let receipt = try VerifiedInstallReceiptWriter.encode(
            outputDir: output,
            manifestSha256: manifestSha,
            manifestSize: UInt64(manifestData.count),
            sourceRepoID: provenance?.repoID,
            sourceRevision: provenance?.revision,
            files: receiptFiles)
        try Posix.atomicWrite(receipt, to: path(output, VerifiedInstallReceiptWriter.fileName),
                              durableIn: output)
        succeeded = true
        return Result(layers: layers.count, expertStride: expertStride,
                      storedExpertStride: plan.storedExpertStride,
                      routedBytesBefore: routedBytesBefore, routedBytesAfter: routedBytesAfter,
                      groupsChecked: groupsChecked)
    }
}

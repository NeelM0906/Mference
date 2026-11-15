import Foundation

/// The install path for a pinned GGUF source (WS4). It shares the range-copy,
/// checkpoint, output-recording and promotion machinery with the safetensors
/// path, but nothing safetensors-specific: the pin is per shard, the
/// tokenizer sidecars are synthesized from GGUF metadata (nothing is fetched
/// from a vendor repo), and every tensor byte is an identity copy.
extension RemoteStreamingRepacker {
    func runPreparedGGUF(paths: RemoteInstallPaths,
                         pin: GGUFSourcePin,
                         progress: @escaping @Sendable (ModelInstallProgress) -> Void) async throws
        -> RemoteStreamingRepackResult {
        try Task.checkCancellation()
        let saved = options.resume ? try RemoteInstallCheckpoint.load(from: paths.checkpointFile) : nil
        if let saved {
            guard saved.repoID == options.repoID, saved.requestedRevision == options.revision else {
                throw RepackError.installStateIncompatible(detail: "saved download belongs to a different source")
            }
        }
        let remote = HuggingFaceRemoteSource(
            repoID: options.repoID, requestedRevision: options.revision, resolvedCommit: pin.commit,
            token: options.token, downloadSession: options.downloadSession, baseURL: options.baseURL,
            tempDirectory: paths.partialDirectory,
            retryPolicy: RemoteRetryPolicy(attempts: options.rangeRetryAttempts,
                                           baseDelayNs: options.retryBaseDelayNs))
        progress(.downloadingMetadata)
        let snapshot = try await RemoteGGUFSnapshotLoader.load(remote: remote, pin: pin, audit: audit)
        try Task.checkCancellation()

        let metadata = snapshot.headers[0].metadata
        let firstShard = pin.shards[0].path
        let arch = try ArchInfo.fromGGUFQwen4Exp(metadata: metadata,
                                                 tensors: snapshot.headers.flatMap(\.tensors),
                                                 path: firstShard)
        let sidecars = try GGUFTokenizer.synthesize(metadata: metadata, path: firstShard)
        let pleHash = try GTurboJSON.GGUFPleHash.from(metadata: metadata, path: firstShard)
        let plan = try FlashNextGGUFPlanner.plan(
            headers: snapshot.headers, shardPaths: pin.shards.map(\.path), arch: arch,
            outputDir: paths.partialDirectory,
            matchedModelID: SourceFingerprint.modelID(forIndexSha256: pin.fingerprint))
        if options.requireKnownSource, plan.matchedModelID == nil {
            throw RepackError.sourceFingerprintRejected(path: firstShard, sha256: pin.fingerprint)
        }
        let rangePlan = try RangeCopyPlanner.plan(repackPlan: plan, rangeChunkBytes: options.rangeChunkBytes,
                                                  layoutMode: "identity", layoutOrderSha256: nil)

        var checkpoint = saved ?? RemoteInstallCheckpoint(
            repoID: options.repoID, requestedRevision: options.revision, resolvedCommit: pin.commit,
            sourceIndexSHA256: pin.fingerprint, planFingerprint: rangePlan.canonicalFingerprint,
            totalSourceBytes: rangePlan.remoteBytesToDownload)
        if saved != nil {
            guard checkpoint.resolvedCommit == pin.commit,
                  checkpoint.totalSourceBytes == rangePlan.remoteBytesToDownload,
                  checkpoint.matches(repoID: options.repoID, requestedRevision: options.revision,
                                     sourceIndexSHA256: pin.fingerprint,
                                     planFingerprint: rangePlan.canonicalFingerprint) else {
                throw RepackError.installStateIncompatible(detail: "saved download source or copy plan changed")
            }
            if try outputFilesMatch(plan: plan, rangePlan: rangePlan) {
                checkpoint.completedRanges = try Self.validatedCompletedRanges(
                    checkpoint.completedRanges, copies: rangePlan.coalescedCopies,
                    partialDirectory: paths.partialDirectory)
            } else {
                checkpoint.completedRanges = []
                try FileManager.default.removeItem(atPath: paths.partialDirectory)
                try Posix.mkdirP(paths.partialDirectory)
                try createOutputFiles(plan: plan, paths: paths)
            }
            try checkpoint.write(to: paths.checkpointFile, parentDirectory: paths.parentDirectory)
        }

        let outputBytes = plan.resident.totalSize
            + plan.layers.reduce(UInt64(0)) { $0 + $1.fileSize }
            + plan.plePools.reduce(UInt64(0)) { $0 + $1.fileSize }
        progress(.planning(downloadBytes: rangePlan.remoteBytesToDownload, outputBytes: outputBytes))
        let reusedDestinationBytes = checkpoint.completedRanges.reduce(UInt64(0)) { $0 + $1.destinationBytes }
        let diskRequirement = try DiskSpaceChecker.requireAvailable(
            path: paths.parentDirectory,
            bytes: (outputBytes > reusedDestinationBytes ? outputBytes - reusedDestinationBytes : 0)
                + UInt64(options.rangeChunkBytes),
            reserveBytes: options.minFreeReserveBytes)
        progress(.checkingDisk(diskRequirement))
        try Task.checkCancellation()

        audit.remoteRepoID = options.repoID
        audit.remoteRequestedRevision = options.revision
        audit.remoteResolvedCommit = pin.commit
        audit.remoteRangeStreamingSupported = true
        audit.remoteGapBytesDownloaded = rangePlan.remoteGapBytesDownloaded
        audit.sourceSnapshotSha256 = pin.fingerprint
        audit.packedExpertLayoutMode = "identity"

        let reusedBytes = checkpoint.completedRanges.reduce(UInt64(0)) { $0 + $1.sourceBytes }
        func result(dryRun: Bool, downloaded: UInt64) -> RemoteStreamingRepackResult {
            RemoteStreamingRepackResult(
                outputDir: options.outputDir, resolvedCommit: pin.commit, plan: plan,
                rangeRequestCount: rangePlan.coalescedCopies.count,
                remoteBytesToDownload: rangePlan.remoteBytesToDownload,
                remoteGapBytesDownloaded: rangePlan.remoteGapBytesDownloaded,
                remoteRetryCount: audit.remoteRangeRetries, reusedBytes: reusedBytes,
                downloadedThisRunBytes: downloaded, dryRun: dryRun, supportingFileBytes: 0,
                metadataReserveBytes: 0, sourceMetadataBytes: snapshot.metadataSourceBytes, impliedBiasBytes: 0)
        }
        if options.dryRunSpaceCheck {
            if saved == nil { try? FileManager.default.removeItem(atPath: paths.partialDirectory) }
            return result(dryRun: true, downloaded: 0)
        }

        if saved == nil {
            progress(.reservingOutput(bytes: outputBytes))
            try createOutputFiles(plan: plan, paths: paths)
            try checkpoint.write(to: paths.checkpointFile, parentDirectory: paths.parentDirectory)
        }
        let provider = HTTPRangeSourceByteProvider(remote: remote.pinned(commit: pin.commit),
                                                   files: snapshot.remoteFiles,
                                                   writeTileBytes: options.writeTileBytes)
        let payloadDownloadStart = audit.remoteBytesDownloaded
        progress(.copyingPayload(reusedBytes: reusedBytes, downloadedThisRunBytes: 0,
                                 totalBytes: rangePlan.remoteBytesToDownload))
        try await provider.copyBatch(
            rangePlan.coalescedCopies,
            completedRangeIDs: Set(checkpoint.completedRanges.map(\.id)),
            partialDirectory: paths.partialDirectory,
            temporaryPath: paths.rangeTemporaryFile,
            audit: audit,
            progress: { downloadedBytes in
                progress(.copyingPayload(reusedBytes: reusedBytes, downloadedThisRunBytes: downloadedBytes,
                                         totalBytes: rangePlan.remoteBytesToDownload))
            },
            commit: { completed in
                checkpoint.completedRanges.removeAll { $0.id == completed.id }
                checkpoint.completedRanges.append(completed)
                checkpoint.completedRanges.sort { $0.id < $1.id }
                try checkpoint.write(to: paths.checkpointFile, parentDirectory: paths.parentDirectory)
            })

        try recordOutputFile(relativePath: "model_weights.bin", path: plan.resident.path, progress: progress)
        for layer in plan.layers where layer.expertsPerLayer > 0 {
            try Task.checkCancellation()
            try recordOutputFile(relativePath: "packed_experts/" + (layer.path as NSString).lastPathComponent,
                                 path: layer.path, progress: progress)
        }
        for pool in plan.plePools {
            try Task.checkCancellation()
            try recordOutputFile(relativePath: pool.relativePath, path: pool.path, progress: progress)
        }
        let layoutPath = ((paths.partialDirectory as NSString).appendingPathComponent("packed_experts")
            as NSString).appendingPathComponent("layout.json")
        try writeSmall(path: layoutPath, data: try GTurboJSON.encodeGGUFLayout(plan: plan))
        try recordOutputFile(relativePath: "packed_experts/layout.json", path: layoutPath, progress: progress)

        let tokenizerDir = (paths.partialDirectory as NSString).appendingPathComponent("tokenizer")
        for (name, data) in [("tokenizer.json", sidecars.tokenizerJSON),
                             ("tokenizer_config.json", sidecars.tokenizerConfigJSON),
                             ("chat_template.jinja", Data(sidecars.chatTemplate.utf8))] {
            let path = (tokenizerDir as NSString).appendingPathComponent(name)
            try writeSmall(path: path, data: data)
            try recordOutputFile(relativePath: "tokenizer/" + name, path: path, progress: progress)
        }
        try? FileManager.default.removeItem(atPath: paths.rangeTemporaryFile)
        progress(.finalizing)
        try Task.checkCancellation()
        try writeGGUFManifest(plan: plan, pin: pin, pleHash: pleHash, partialDir: paths.partialDirectory)

        try Task.checkCancellation()
        if try Posix.entryKind(paths.finalDirectory) == .directory {
            try Posix.renameSwap(paths.partialDirectory, paths.finalDirectory)
            try Posix.fsyncDirectory(paths.parentDirectory)
            try? FileManager.default.removeItem(atPath: paths.partialDirectory)
        } else {
            try Posix.rename(from: paths.partialDirectory, to: paths.finalDirectory)
            try Posix.fsyncDirectory(paths.parentDirectory)
        }
        try? FileManager.default.removeItem(atPath: paths.checkpointFile)
        audit.wallTimeSeconds = Date().timeIntervalSince(startTime)
        audit.wholeFileHeapBuffers = false
        if let auditPath = options.copyAuditPath {
            let data = try audit.toJSONData(outputDir: options.outputDir)
            try Posix.mkdirP((auditPath as NSString).deletingLastPathComponent)
            try data.write(to: URL(fileURLWithPath: auditPath))
        }
        return result(dryRun: false, downloaded: audit.remoteBytesDownloaded - payloadDownloadStart)
    }

    private func writeGGUFManifest(plan: RepackPlan, pin: GGUFSourcePin, pleHash: GTurboJSON.GGUFPleHash,
                                   partialDir: String) throws {
        let files = audit.outputFiles.map {
            ($0.relativePath, GTurboJSON.FileEntry(size: $0.size, sha256: $0.sha256))
        }
        let base = try GTurboJSON.encodeManifest(
            plan: plan,
            modelID: plan.matchedModelID ?? "unknown/snapshot",
            sourceSnapshotHash: "sha256:" + pin.fingerprint,
            files: files,
            expertsPerLayer: plan.layers.first(where: { $0.expertsPerLayer > 0 })?.expertsPerLayer ?? 0,
            numLayers: plan.arch.numLayers,
            expertStride: plan.layers.map(\.expertStride).max() ?? 0,
            bitWidths: GTurboJSON.QuantBitWidths(embedding: 0, attention: 0, router: 0, sharedExpert: 0,
                                                 routedExpert: 0))
        let data = try GTurboJSON.addGGUFManifestFields(base, plan: plan, repoID: options.repoID, pin: pin,
                                                        pleHash: pleHash)
        let tmp = (partialDir as NSString).appendingPathComponent("manifest.json.tmp")
        let final = (partialDir as NSString).appendingPathComponent("manifest.json")
        try writeSmall(path: tmp, data: data)
        try Posix.rename(from: tmp, to: final)
        let receipt = try VerifiedInstallReceiptWriter.encode(
            outputDir: options.outputDir,
            manifestSha256: try Sha256Stream.hashFile(path: final),
            manifestSize: UInt64(data.count),
            sourceRepoID: options.repoID,
            sourceRevision: pin.commit,
            files: audit.outputFiles)
        try writeSmall(path: (partialDir as NSString).appendingPathComponent(VerifiedInstallReceiptWriter.fileName),
                       data: receipt)
    }
}

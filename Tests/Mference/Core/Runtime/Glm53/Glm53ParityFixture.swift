import Foundation
import Metal
@testable import Mference
@testable import MferenceRepackCore

/// Installs the committed toy checkpoint (`Fixtures/glm53/toy-ckpt/`) into a
/// temporary `.gturbo` **through the real planner** — `IndexLoader`,
/// `ArchInfo.load`, `RepackPlanner.plan`, `RangeCopyPlanner.plan`,
/// `ResidentWriter`, `GTurboJSON` — and loads it with `Model.load(expecting:)`.
///
/// # Why a local executor rather than `RemoteStreamingRepacker`
///
/// The streaming installer resolves its snapshot over HTTP and the fake Hub
/// lives in the `MferenceRepackTests` target, which this target cannot import.
/// The plan is the part worth exercising: every byte the runner reads is
/// placed where the production planner put it, by the production writers, and
/// the copies themselves are identity ranges (PipeNetwork's conversion is
/// pre-quantized in exactly the storage the install keeps), executed here
/// from the local shards.
enum Glm53Parity {

    static func archConfig() -> ArchConfig { .glm53Toy() }

    /// Writes the install; returns its directory. The caller removes it.
    static func installToyCheckpoint() throws -> URL {
        let source = Glm53Fixtures.checkpointDirectory.path
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("gturbo-glm53-parity-\(UUID().uuidString)")
        let out = dir.path
        try Posix.mkdirP(out)

        let metadata = try IndexLoader.load(snapshotDir: source)
        let arch = try ArchInfo.load(configPath: metadata.configPath)
        var headers: [MferenceRepackCore.Safetensors.Header] = []
        for shard in metadata.shardFilenames {
            let path = (source as NSString).appendingPathComponent(shard)
            let data = try Data(contentsOf: URL(fileURLWithPath: path), options: .mappedIfSafe)
            var headerSize: UInt64 = 0
            _ = withUnsafeMutableBytes(of: &headerSize) { data.copyBytes(to: $0, from: 0..<8) }
            headers.append(try MferenceRepackCore.Safetensors.parseHeaderBytes(
                path: shard, fileSize: UInt64(data.count),
                headerBytes: data.subdata(in: 8..<(8 + Int(headerSize)))))
        }
        let plan = try RepackPlanner.plan(meta: metadata, arch: arch, shardHeaders: headers,
                                          outputDir: out)
        let rangePlan = try RangeCopyPlanner.plan(repackPlan: plan, rangeChunkBytes: 8 << 20)
        let audit = RepackAudit()

        // Output files, sized and indexed the way the installer creates them.
        try Posix.mkdirP((out as NSString).appendingPathComponent("packed_experts"))
        let residentFD = try ResidentWriter.createAndWriteIndex(plan: plan.resident, audit: audit)
        close(residentFD)
        for layer in plan.allExpertLayers {
            let fd = try Posix.openCreateRW(layer.path)
            try Posix.ftruncate(fd, path: layer.path, size: layer.fileSize)
            close(fd)
        }

        // Identity copies from the local shards.
        var shardData: [String: Data] = [:]
        func bytes(of shard: String) throws -> Data {
            if let cached = shardData[shard] { return cached }
            let data = try Data(contentsOf: URL(fileURLWithPath:
                (source as NSString).appendingPathComponent(shard)), options: .mappedIfSafe)
            shardData[shard] = data
            return data
        }
        var openFDs: [String: Int32] = [:]
        func fd(for path: String) throws -> Int32 {
            if let fd = openFDs[path] { return fd }
            let fd = try Posix.openExistingRW(path)
            openFDs[path] = fd
            return fd
        }
        defer { openFDs.values.forEach { close($0) } }
        for coalesced in rangePlan.coalescedCopies {
            let data = try bytes(of: coalesced.shardID)
            for copy in coalesced.destinations {
                precondition(copy.transform == .identity,
                             "the pre-quantized GLM-5.3 plan copies bytes verbatim; got \(copy.transform)")
                let begin = Int(copy.sourceOffset)
                let slice = data.subdata(in: begin..<(begin + Int(copy.size)))
                let destination = try fd(for: copy.destinationPath)
                try slice.withUnsafeBytes { raw in
                    try Posix.pwriteAll(fd: destination, path: copy.destinationPath,
                                        buf: raw.baseAddress!, count: raw.count,
                                        offset: copy.destinationOffset)
                }
            }
        }
        for (_, fd) in openFDs { fsync(fd) }

        // Layout and manifest — recorded with sizes and hashes like the installer.
        var files: [(relativePath: String, info: GTurboJSON.FileEntry)] = []
        func record(_ relative: String, _ path: String) throws {
            let f = try Posix.openRead(path)
            defer { close(f) }
            let size = try Posix.fileSize(fd: f, path: path)
            let sha = try WriterCore.hashEntireFile(path: path, size: size, audit: audit)
            files.append((relative, GTurboJSON.FileEntry(size: size, sha256: sha)))
        }
        try record("model_weights.bin", plan.resident.path)
        for layer in plan.allExpertLayers {
            try record("packed_experts/" + (layer.path as NSString).lastPathComponent, layer.path)
        }
        let expertStride = plan.layers.first(where: { $0.expertsPerLayer > 0 })?.expertStride ?? 0
        let layoutData = try GTurboJSON.encodeLayout(plan: plan, expertStride: expertStride)
        let layoutPath = ((out as NSString).appendingPathComponent("packed_experts") as NSString)
            .appendingPathComponent("layout.json")
        try layoutData.write(to: URL(fileURLWithPath: layoutPath))
        try GTurboLayoutValidator.validate(path: layoutPath, plan: plan)
        try record("packed_experts/layout.json", layoutPath)

        // The bit widths the installer records for this family: INT8 everywhere
        // but the INT4 experts and the unquantized (BF16) router.
        let bits = GTurboJSON.QuantBitWidths(embedding: 8, attention: 8, router: 16,
                                             sharedExpert: 8, routedExpert: 4)
        let manifest = try GTurboJSON.encodeManifest(
            plan: plan, modelID: "glm53flash-parity-toy",
            sourceSnapshotHash: "sha256:" + metadata.indexSha256Hex,
            files: files,
            expertsPerLayer: plan.layers.first(where: { $0.expertsPerLayer > 0 })?.expertsPerLayer ?? 0,
            numLayers: plan.arch.numLayers, expertStride: expertStride, bitWidths: bits)
        try manifest.write(to: dir.appendingPathComponent("manifest.json"))
        return dir
    }

    /// Opens the install with the family's toy baseline and the pread slot
    /// cache. `expecting:` is the loader entry the gated-family tests use; the
    /// family stays behind `familiesWithoutRunner` until its runner is proven.
    static func loadModel(at dir: URL, device: MTLDevice, slots: Int = 16,
                          mode: ExpertStreamingMode? = nil) throws -> Model {
        try Model.load(directoryURL: dir, device: device, expecting: archConfig(),
                       streamingMode: mode ?? .pread(slotCount: slots))
    }
}

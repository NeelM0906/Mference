import Foundation
import Metal

extension Native {
    /// The production `WeightStore` over an `.mfpack` v1 directory
    /// (`docs/MFERENCE_PACK.md`).
    ///
    /// `weights.bin` is mapped read-only and shared, and wrapped once with
    /// `makeBuffer(bytesNoCopy:)`; every `QuantTensor` is an offset into that
    /// one buffer, in the tile layout its manifest entry records. Nothing
    /// tensor-sized is copied into the Swift heap: the heap holds the manifest,
    /// the tensor index and the tokenizer tables.
    ///
    /// Opening checks, before any tensor is handed out:
    /// - `verified-install.json` exists and matches `manifest.json` (SHA-256),
    ///   `weights.bin` (size) and, when the writer recorded them,
    ///   `tokenizer.json` and `chat_template.jinja` (SHA-256);
    /// - format `mfpack` v1, architecture `qwen35`, a known layout table;
    /// - every tensor: a known block format, whole blocks, byte count, a
    ///   layout that fits (and, under `PackLayoutTable.version`, is the
    ///   table's), a 4096-aligned offset inside the file, no overlap;
    /// - the tensor set against `Qwen35TensorCatalog` for the manifest's config.
    ///
    /// `Verification.packedSHA256` also hashes every tensor's packed bytes
    /// (reads the whole file once, in parallel).
    public final class PackWeightStore: WeightStore {
        public enum Verification: Sendable, Equatable {
            /// Structural checks and the receipt (reads only metadata files).
            case receipt
            /// Also the packed SHA-256 of every tensor.
            case packedSHA256
        }

        public static let pageAlignment: UInt64 = 4096

        public let directory: URL
        public let manifest: PackManifest
        public let receipt: PackReceipt
        public var config: Qwen35Config { manifest.config }
        public let mappedByteCount: UInt64
        public let tokenizer: TokenizerSource
        public let chatTemplate: ChatTemplateSource
        /// The buffer every tensor points into.
        public let buffer: MTLBuffer
        private let mapping: PackMappedFile
        private let index: [String: Int]

        public init(directory: URL, device: MTLDevice, verification: Verification = .receipt) throws {
            self.directory = directory
            func file(_ name: String) -> URL { directory.appendingPathComponent(name) }
            let receiptURL = file("verified-install.json")
            guard FileManager.default.fileExists(atPath: receiptURL.path) else {
                throw NativeError.integrity("\(directory.path) has no verified-install.json (incomplete pack)")
            }
            let receipt: PackReceipt
            do {
                receipt = try JSONDecoder().decode(PackReceipt.self, from: Data(contentsOf: receiptURL))
            } catch {
                throw NativeError.integrity("verified-install.json: \(error)")
            }
            let manifestData = try Self.read(file("manifest.json"))
            guard PackHash.sha256Hex(manifestData) == receipt.manifestSHA256 else {
                throw NativeError.integrity("manifest.json does not match verified-install.json")
            }
            let manifest: PackManifest
            do {
                manifest = try JSONDecoder().decode(PackManifest.self, from: manifestData)
            } catch {
                throw NativeError.integrity("manifest.json: \(error)")
            }
            let tokenizerData = try Self.read(file("tokenizer.json"))
            let templateData = try Self.read(file("chat_template.jinja"))
            if let want = receipt.tokenizerSHA256, PackHash.sha256Hex(tokenizerData) != want {
                throw NativeError.integrity("tokenizer.json does not match verified-install.json")
            }
            if let want = receipt.chatTemplateSHA256, PackHash.sha256Hex(templateData) != want {
                throw NativeError.integrity("chat_template.jinja does not match verified-install.json")
            }

            let mapping = try PackMappedFile(url: file("weights.bin"))
            guard receipt.format == manifest.format, receipt.formatVersion == manifest.formatVersion,
                  receipt.label == manifest.label, receipt.sourceSHA256 == manifest.source.sha256,
                  receipt.tensorCount == manifest.tensors.count,
                  receipt.weightsByteCount == UInt64(mapping.size) else {
                throw NativeError.integrity("verified-install.json does not describe this pack")
            }
            index = try Self.validate(manifest, fileSize: UInt64(mapping.size))

            let tokData: TokenizerData
            do {
                tokData = try JSONDecoder().decode(TokenizerData.self, from: tokenizerData)
            } catch {
                throw NativeError.integrity("tokenizer.json: \(error)")
            }
            guard tokData.tokens.count == manifest.config.vocabSize else {
                throw NativeError.integrity("tokenizer.json has \(tokData.tokens.count) tokens, "
                                            + "config vocabSize \(manifest.config.vocabSize)")
            }
            guard let template = String(data: templateData, encoding: .utf8) else {
                throw NativeError.integrity("chat_template.jinja is not UTF-8")
            }

            if verification == .packedSHA256 {
                try Self.verifyPackedHashes(manifest, mapping: mapping)
            }
            guard let buffer = device.makeBuffer(bytesNoCopy: UnsafeMutableRawPointer(mutating: mapping.base),
                                                 length: mapping.mappedLength, options: .storageModeShared,
                                                 deallocator: nil) else {
                throw NativeError.resource("cannot wrap weights.bin (\(mapping.size) bytes)")
            }
            buffer.label = "mfpack \(manifest.label) weights"
            self.manifest = manifest
            self.receipt = receipt
            self.mapping = mapping
            self.buffer = buffer
            self.mappedByteCount = UInt64(mapping.mappedLength)
            self.tokenizer = PackTokenizerSource(data: tokData)
            self.chatTemplate = PackChatTemplateSource(template: template, data: tokData)
        }

        public func tensor(_ name: String) throws -> QuantTensor {
            guard let i = index[name] else { throw NativeError.missing("tensor \(name)") }
            let t = manifest.tensors[i]
            return QuantTensor(name: t.name, type: t.type, shape: t.shape, layout: t.layout, buffer: buffer,
                               offset: Int(t.offset), byteCount: Int(t.byteCount), sourceSHA256: t.sourceSHA256)
        }

        public func contains(_ name: String) -> Bool { index[name] != nil }

        /// The tensor's packed bytes in the mapping (valid for the store's lifetime).
        public func packedBytes(_ name: String) throws -> UnsafeRawBufferPointer {
            guard let i = index[name] else { throw NativeError.missing("tensor \(name)") }
            let t = manifest.tensors[i]
            return mapping.bytes(offset: Int(t.offset), count: Int(t.byteCount))
        }

        // MARK: Validation

        static func read(_ url: URL) throws -> Data {
            do {
                return try Data(contentsOf: url)
            } catch {
                throw NativeError.missing("\(url.path): \(error.localizedDescription)")
            }
        }

        /// Structural checks of `manifest` against a `weights.bin` of
        /// `fileSize` bytes; returns the name → tensor-index map.
        static func validate(_ manifest: PackManifest, fileSize: UInt64) throws -> [String: Int] {
            guard manifest.format == "mfpack", manifest.formatVersion == 1 else {
                throw NativeError.unsupported("pack format \(manifest.format) v\(manifest.formatVersion)")
            }
            guard manifest.architecture == "qwen35" else {
                throw NativeError.unsupported("architecture \(manifest.architecture)")
            }
            // 0: per-tensor layouts chosen outside the table (tests, experiments).
            guard manifest.layoutTableVersion == 0 || manifest.layoutTableVersion == PackLayoutTable.version else {
                throw NativeError.unsupported("pack layout table v\(manifest.layoutTableVersion) "
                                              + "(this build knows v\(PackLayoutTable.version))")
            }
            var index: [String: Int] = [:]
            index.reserveCapacity(manifest.tensors.count)
            var spans: [(start: UInt64, end: UInt64, name: String)] = []
            spans.reserveCapacity(manifest.tensors.count)
            for (i, t) in manifest.tensors.enumerated() {
                guard index[t.name] == nil else { throw NativeError.integrity("tensor \(t.name) appears twice") }
                guard let format = QuantBlockFormat.of(t.type) else {
                    throw NativeError.unsupported("\(t.name): \(t.type)")
                }
                guard (1...4).contains(t.shape.count), t.shape.allSatisfy({ $0 > 0 }),
                      t.shape[0] % format.elements == 0 else {
                    throw NativeError.invalidShape("\(t.name): shape \(t.shape) for \(t.type)")
                }
                var elements = 1
                for d in t.shape {
                    let (p, overflow) = elements.multipliedReportingOverflow(by: d)
                    guard !overflow else { throw NativeError.invalidShape("\(t.name): \(t.shape) overflows") }
                    elements = p
                }
                let rows = elements / t.shape[0]
                let blocksPerRow = t.shape[0] / format.elements
                let (bytes, overflow) = (elements / format.elements).multipliedReportingOverflow(by: format.bytes)
                guard !overflow, UInt64(bytes) == t.byteCount else {
                    throw NativeError.integrity("\(t.name): byte count \(t.byteCount), shape implies \(bytes)")
                }
                guard t.layout.fits(rows: rows, blocksPerRow: blocksPerRow) else {
                    throw NativeError.integrity("\(t.name): layout \(t.layout) does not fit \(rows)x\(blocksPerRow)")
                }
                if manifest.layoutTableVersion == PackLayoutTable.version {
                    let want = PackLayoutTable.layout(for: t.type, rows: rows, blocksPerRow: blocksPerRow)
                    guard t.layout == want else {
                        throw NativeError.integrity("\(t.name): layout \(t.layout), table v\(PackLayoutTable.version) "
                                                    + "says \(want)")
                    }
                }
                let (end, endOverflow) = t.offset.addingReportingOverflow(t.byteCount)
                guard t.offset % pageAlignment == 0, !endOverflow, end <= fileSize else {
                    throw NativeError.integrity("\(t.name): offset \(t.offset) + \(t.byteCount) outside "
                                                + "weights.bin (\(fileSize) bytes) or not 4096-aligned")
                }
                guard t.sourceSHA256.count == 64, t.packedSHA256.count == 64 else {
                    throw NativeError.integrity("\(t.name): malformed SHA-256")
                }
                if t.layout == .gguf, t.sourceSHA256 != t.packedSHA256 {
                    throw NativeError.integrity("\(t.name): identity layout but source and packed hashes differ")
                }
                index[t.name] = i
                spans.append((t.offset, end, t.name))
            }
            spans.sort { $0.start < $1.start }
            for (a, b) in zip(spans, spans.dropFirst()) where b.start < a.end {
                throw NativeError.integrity("tensors \(a.name) and \(b.name) overlap")
            }
            try Qwen35TensorCatalog.validate(manifest)
            return index
        }

        /// Every tensor's packed SHA-256 against the manifest, hashed in
        /// parallel (each worker streams its tensor in bounded chunks).
        static func verifyPackedHashes(_ manifest: PackManifest, mapping: PackMappedFile) throws {
            let tensors = manifest.tensors
            // Largest first, so the long hashes start early.
            let order = tensors.indices.sorted { tensors[$0].byteCount > tensors[$1].byteCount }
            let failures = PackFailureList()
            let next = PackCounter()
            let workers = max(1, min(ProcessInfo.processInfo.activeProcessorCount, 8))
            DispatchQueue.concurrentPerform(iterations: workers) { _ in
                while let k = next.take(below: order.count) {
                    autoreleasepool {
                        let t = tensors[order[k]]
                        let bytes = mapping.bytes(offset: Int(t.offset), count: Int(t.byteCount))
                        if PackHash.sha256Hex(bytes) != t.packedSHA256 { failures.append(t.name) }
                    }
                }
            }
            if let first = failures.sorted.first {
                throw NativeError.integrity("\(first): packed SHA-256 mismatch "
                                            + "(\(failures.sorted.count) tensor(s) failed)")
            }
        }
    }

    /// A lock-protected work counter for `concurrentPerform` workers.
    final class PackCounter: @unchecked Sendable {
        private let lock = NSLock()
        private var value = 0
        func take(below limit: Int) -> Int? {
            lock.lock()
            defer { lock.unlock() }
            guard value < limit else { return nil }
            value += 1
            return value - 1
        }
    }

    /// A lock-protected list of failed tensor names.
    final class PackFailureList: @unchecked Sendable {
        private let lock = NSLock()
        private var names: [String] = []
        func append(_ name: String) {
            lock.lock()
            names.append(name)
            lock.unlock()
        }
        var sorted: [String] {
            lock.lock()
            defer { lock.unlock() }
            return names.sorted()
        }
    }
}

import Foundation
import Metal

extension Native {
    /// The reference `WeightStore`: a slow but complete `.mfpack` v1 reader.
    /// `weights.bin` is mapped read-only and wrapped once with
    /// `makeBuffer(bytesNoCopy:)`; every `QuantTensor` is an offset into that
    /// buffer, in whatever layout the manifest records.
    public final class ReferenceWeightStore: WeightStore {
        public let manifest: PackManifest
        public var config: Qwen35Config { manifest.config }
        public let mappedByteCount: UInt64
        public let tokenizer: TokenizerSource
        public let chatTemplate: ChatTemplateSource
        /// The buffer every tensor points into.
        public let buffer: MTLBuffer
        private let mapping: ReferenceMappedFile?
        private let index: [String: PackManifest.Tensor]

        /// Open the pack at `directory`. With `verifyHashes`, every tensor's
        /// packed SHA-256 is checked against the manifest (reads the whole file).
        public convenience init(directory: URL, device: MTLDevice, verifyHashes: Bool = false) throws {
            let fm = FileManager.default
            let manifestURL = directory.appendingPathComponent("manifest.json")
            let receiptURL = directory.appendingPathComponent("verified-install.json")
            guard fm.fileExists(atPath: receiptURL.path) else {
                throw NativeError.integrity("\(directory.path) has no verified-install.json (incomplete pack)")
            }
            let manifestData = try Data(contentsOf: manifestURL)
            let manifest = try JSONDecoder().decode(PackManifest.self, from: manifestData)
            let receipt = try JSONDecoder().decode(ReferencePackWriter.Receipt.self,
                                                   from: Data(contentsOf: receiptURL))
            guard receipt.manifestSHA256 == ReferencePackWriter.sha256Hex(manifestData) else {
                throw NativeError.integrity("manifest.json does not match verified-install.json")
            }
            let mapping = try ReferenceMappedFile(url: directory.appendingPathComponent("weights.bin"))
            guard receipt.format == manifest.format, receipt.formatVersion == manifest.formatVersion,
                  receipt.sourceSHA256 == manifest.source.sha256, receipt.tensorCount == manifest.tensors.count,
                  receipt.weightsByteCount == UInt64(mapping.size) else {
                throw NativeError.integrity("verified-install.json does not describe this pack")
            }
            guard let buffer = device.makeBuffer(bytesNoCopy: UnsafeMutableRawPointer(mutating: mapping.base),
                                                 length: mapping.mappedLength, options: .storageModeShared,
                                                 deallocator: nil) else {
                throw NativeError.resource("cannot wrap weights.bin (\(mapping.size) bytes)")
            }
            let tokData = try JSONDecoder().decode(TokenizerData.self,
                                                   from: Data(contentsOf: directory.appendingPathComponent("tokenizer.json")))
            let template = try String(contentsOf: directory.appendingPathComponent("chat_template.jinja"), encoding: .utf8)
            try self.init(manifest: manifest, buffer: buffer, fileSize: mapping.size,
                          mappedByteCount: UInt64(mapping.mappedLength), mapping: mapping,
                          tokenizerData: tokData, chatTemplate: template)
            if verifyHashes {
                for t in manifest.tensors {
                    let bytes = UnsafeRawBufferPointer(start: mapping.base + Int(t.offset), count: Int(t.byteCount))
                    guard ReferencePackWriter.sha256Hex(bytes) == t.packedSHA256 else {
                        throw NativeError.integrity("\(t.name): packed SHA-256 mismatch")
                    }
                }
            }
        }

        /// A store over an existing buffer holding the packed tensors at their
        /// manifest offsets (tests and synthetic models).
        public convenience init(manifest: PackManifest, buffer: MTLBuffer, tokenizerData: TokenizerData,
                                chatTemplate: String) throws {
            try self.init(manifest: manifest, buffer: buffer, fileSize: buffer.length,
                          mappedByteCount: UInt64(buffer.length), mapping: nil,
                          tokenizerData: tokenizerData, chatTemplate: chatTemplate)
        }

        private init(manifest: PackManifest, buffer: MTLBuffer, fileSize: Int, mappedByteCount: UInt64,
                     mapping: ReferenceMappedFile?, tokenizerData: TokenizerData, chatTemplate: String) throws {
            guard manifest.format == "mfpack", manifest.formatVersion == 1 else {
                throw NativeError.unsupported("pack format \(manifest.format) v\(manifest.formatVersion)")
            }
            guard manifest.architecture == "qwen35" else {
                throw NativeError.unsupported("architecture \(manifest.architecture)")
            }
            var index: [String: PackManifest.Tensor] = [:]
            for t in manifest.tensors {
                guard let format = QuantBlockFormat.of(t.type), !t.shape.isEmpty else {
                    throw NativeError.unsupported("\(t.name): \(t.type)")
                }
                let elements = t.shape.reduce(1, *)
                let rows = t.shape.dropFirst().reduce(1, *)
                guard t.shape[0] % format.elements == 0,
                      UInt64(elements / format.elements * format.bytes) == t.byteCount,
                      t.layout.fits(rows: rows, blocksPerRow: t.shape[0] / format.elements),
                      t.offset % ReferencePackWriter.pageAlignment == 0,
                      t.offset + t.byteCount <= UInt64(fileSize) else {
                    throw NativeError.integrity("\(t.name): shape, size, layout or offset is inconsistent")
                }
                index[t.name] = t
            }
            self.manifest = manifest
            self.buffer = buffer
            self.mapping = mapping
            self.mappedByteCount = mappedByteCount
            self.index = index
            self.tokenizer = ReferenceTokenizerSource(data: tokenizerData)
            self.chatTemplate = ReferenceChatTemplateSource(template: chatTemplate, data: tokenizerData)
        }

        public func tensor(_ name: String) throws -> QuantTensor {
            guard let t = index[name] else { throw NativeError.missing("tensor \(name)") }
            return QuantTensor(name: t.name, type: t.type, shape: t.shape, layout: t.layout, buffer: buffer,
                               offset: Int(t.offset), byteCount: Int(t.byteCount), sourceSHA256: t.sourceSHA256)
        }

        public func contains(_ name: String) -> Bool { index[name] != nil }
    }

    /// `ChatTemplateSource` over the pack's `chat_template.jinja`.
    public final class ReferenceChatTemplateSource: ChatTemplateSource {
        public let template: String
        public let bosToken: String?
        public let eosToken: String?

        public init(template: String, data: TokenizerData) {
            self.template = template
            let ids = ReferenceTokenizer.specialIDs(data)
            func text(_ id: Int32?) -> String? { id.map { data.tokens[Int($0)] } }
            bosToken = text(ids.bos)
            eosToken = text(ids.eos)
        }
    }

    /// `TokenizerSource` over the pack's `tokenizer.json`.
    public final class ReferenceTokenizerSource: TokenizerSource {
        public let data: TokenizerData
        public init(data: TokenizerData) { self.data = data }
        public func makeTokenizer() throws -> any TextTokenizer { try ReferenceTokenizer(data: data) }
    }
}

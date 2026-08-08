import Darwin
import Foundation
import Testing
@testable import MferenceRepackCore

@Suite
struct RangeCopyPlannerTests {
    @Test func identityFingerprintRetainsLegacyV1Serialization() throws {
        let root = temporaryRoot("fingerprint-v1")
        defer { try? FileManager.default.removeItem(atPath: root) }
        let output = (root as NSString).appendingPathComponent("weights.bin")
        let copy = CoalescedRangeCopy(
            id: "range-00000000",
            shardID: "source.bin",
            sourceOffset: 4,
            size: 8,
            destinations: [RangeCopy(
                shardID: "source.bin",
                sourceOffset: 4,
                size: 8,
                destinationPath: output,
                destinationOffset: 12)])

        let fingerprint = try RangeCopyPlanner.canonicalFingerprint(
            copies: [copy],
            outputRoot: root,
            rangeChunkBytes: 4096,
            layoutMode: "identity",
            layoutOrderSha256: nil,
            residentIndexSha256: String(repeating: "a", count: 64),
            expectedOutputs: [
                RemoteExpectedOutput(relativePath: "weights.bin", size: 32),
            ])

        #expect(fingerprint
            == "fa14918215869b02f0fae5b0de6b05112c505b2c3f2734f8924a8a7adb926581")
    }

    @Test func transformedCopiesSplitOnInputUnitsAndAdvanceExpandedOffsets() throws {
        let root = temporaryRoot("transform-split")
        defer { try? FileManager.default.removeItem(atPath: root) }
        let output = (root as NSString).appendingPathComponent("weights.bin")
        let copy = RangeCopy(
            shardID: "source.bin",
            sourceOffset: 10,
            size: 12,
            destinationPath: output,
            destinationOffset: 100,
            transform: .unpackInt2ToInt4)

        let coalesced = try RangeCopyPlanner.coalesce(
            copies: [copy],
            rangeChunkBytes: 5)
        let pieces = coalesced.flatMap(\.destinations)

        #expect(coalesced.map(\.sourceOffset) == [10, 14, 18])
        #expect(coalesced.map(\.size) == [4, 4, 4])
        #expect(pieces.map(\.destinationOffset) == [100, 108, 116])
        #expect(pieces.allSatisfy { $0.transform == .unpackInt2ToInt4 })
    }

    @Test func invalidTransformRangesAreRejected() throws {
        let root = temporaryRoot("transform-invalid")
        defer { try? FileManager.default.removeItem(atPath: root) }
        let output = (root as NSString).appendingPathComponent("weights.bin")
        let invalidCopies = [
            RangeCopy(
                shardID: "source.bin",
                sourceOffset: 0,
                size: 0,
                destinationPath: output,
                destinationOffset: 0),
            RangeCopy(
                shardID: "source.bin",
                sourceOffset: 0,
                size: 6,
                destinationPath: output,
                destinationOffset: 0,
                transform: .unpackInt2ToInt4),
            RangeCopy(
                shardID: "source.bin",
                sourceOffset: 0,
                size: 2,
                destinationPath: output,
                destinationOffset: 0,
                transform: .repeatBF16(count: 0, negated: false)),
            RangeCopy(
                shardID: "source.bin",
                sourceOffset: 0,
                size: UInt64.max / 2 + 1,
                destinationPath: output,
                destinationOffset: 0,
                transform: .repeatBF16(count: 2, negated: false)),
            RangeCopy(
                shardID: "source.bin",
                sourceOffset: UInt64.max,
                size: 1,
                destinationPath: output,
                destinationOffset: 0),
        ]

        for copy in invalidCopies {
            #expect(throws: RepackError.self) {
                _ = try RangeCopyPlanner.coalesce(
                    copies: [copy],
                    rangeChunkBytes: 8)
            }
        }

        #expect(throws: RepackError.self) {
            try RangeCopyPlanner.validateDestinationIntervals(
                [RangeCopy(
                    shardID: "source.bin",
                    sourceOffset: 0,
                    size: 4,
                    destinationPath: output,
                    destinationOffset: UInt64.max - 3,
                    transform: .unpackInt2ToInt4)],
                outputRoot: root)
        }
    }

    @Test func canonicalFingerprintDoesNotDependOnAbsoluteOutputRoot() throws {
        let snapshotDirectory = temporaryRoot("snapshot")
        let firstOutput = temporaryRoot("first")
        let secondOutput = temporaryRoot("second")
        defer {
            try? FileManager.default.removeItem(atPath: snapshotDirectory)
            try? FileManager.default.removeItem(atPath: firstOutput)
            try? FileManager.default.removeItem(atPath: secondOutput)
        }
        let snapshot = try SyntheticSnapshot.build(
            at: snapshotDirectory,
            seed: 0x1020_3040)
        let metadata = try IndexLoader.load(snapshotDir: snapshotDirectory)
        let arch = try ArchInfo.load(
            configPath: (snapshotDirectory as NSString).appendingPathComponent("config.json"))
        let header = try parseHeader(path: snapshot.shardPath)
        let firstPlan = try RepackPlanner.plan(
            meta: metadata,
            arch: arch,
            shardHeaders: [header],
            outputDir: firstOutput)
        let secondPlan = try RepackPlanner.plan(
            meta: metadata,
            arch: arch,
            shardHeaders: [header],
            outputDir: secondOutput)

        let first = try RangeCopyPlanner.plan(
            repackPlan: firstPlan,
            rangeChunkBytes: 4096)
        let second = try RangeCopyPlanner.plan(
            repackPlan: secondPlan,
            rangeChunkBytes: 4096)

        #expect(first.canonicalFingerprint == second.canonicalFingerprint)
        #expect(first.coalescedCopies.map(\.id) == second.coalescedCopies.map(\.id))
    }

    @Test func overlappingDestinationIntervalsAreRejected() throws {
        let root = temporaryRoot("overlap")
        defer { try? FileManager.default.removeItem(atPath: root) }
        let output = (root as NSString).appendingPathComponent("file.bin")
        let copies = [
            RangeCopy(
                shardID: "source.bin",
                sourceOffset: 0,
                size: 10,
                destinationPath: output,
                destinationOffset: 0),
            RangeCopy(
                shardID: "source.bin",
                sourceOffset: 20,
                size: 10,
                destinationPath: output,
                destinationOffset: 9),
        ]

        #expect(throws: RepackError.self) {
            try RangeCopyPlanner.validateDestinationIntervals(
                copies,
                outputRoot: root)
        }
    }

    @Test func normalizedRelativePathRejectsEscape() throws {
        let root = temporaryRoot("escape")
        defer { try? FileManager.default.removeItem(atPath: root) }
        let outside = (root as NSString).deletingLastPathComponent
            + "/outside.bin"

        #expect(throws: RepackError.self) {
            _ = try RangeCopyPlanner.normalizedRelativePath(
                outside,
                root: root)
        }
    }

    private func temporaryRoot(_ tag: String) -> String {
        let path = (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("mference-range-plan-\(tag)-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(
            atPath: path,
            withIntermediateDirectories: true)
        return path
    }

    private func parseHeader(path: String) throws -> Safetensors.Header {
        let fd = try Posix.openRead(path)
        defer { close(fd) }
        var headerSize: UInt64 = 0
        try withUnsafeMutableBytes(of: &headerSize) {
            try Posix.preadAll(
                fd: fd,
                path: path,
                buf: $0.baseAddress!,
                count: 8,
                offset: 0)
        }
        headerSize = UInt64(littleEndian: headerSize)
        var headerData = Data(count: Int(headerSize))
        try headerData.withUnsafeMutableBytes {
            try Posix.preadAll(
                fd: fd,
                path: path,
                buf: $0.baseAddress!,
                count: $0.count,
                offset: 8)
        }
        return try Safetensors.parseHeaderBytes(
            path: path,
            fileSize: try Posix.fileSize(fd: fd, path: path),
            headerBytes: headerData)
    }
}

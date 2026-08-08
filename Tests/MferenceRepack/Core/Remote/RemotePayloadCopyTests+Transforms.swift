import Foundation
import Synchronization
import Testing

@testable import MferenceRepackCore

extension RemotePayloadCopyTests {
    @Test func int2TransformWritesExactBytesAcrossScratchTiles() async throws {
        let root = tmpDirForRemote("transform-int2")
        defer { cleanUpRemote([root]) }
        resetFakeHF()
        let source: [UInt8] = [
            0x00, 0x00, 0x00, 0x00,
            0xff, 0xff, 0xff, 0xff,
            0xe4, 0x1b, 0x4e, 0xb1,
        ]
        let expected: [UInt8] = [
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x33, 0x33, 0x33, 0x33, 0x33, 0x33, 0x33, 0x33,
            0x10, 0x32, 0x23, 0x01, 0x32, 0x10, 0x01, 0x23,
        ]
        let destination = (root as NSString).appendingPathComponent("int4.bin")
        try createTransformOutput(destination, size: expected.count)
        let copy = transformCopy(
            sourceSize: source.count,
            destinationPath: destination,
            transform: .unpackInt2ToInt4)
        let provider = transformProvider(
            source: source,
            root: root,
            writeTileBytes: 8)
        let audit = RepackAudit()
        let committed = Mutex<[RemoteCompletedRange]>([])

        try await provider.copyBatch(
            [copy],
            completedRangeIDs: [],
            partialDirectory: root,
            temporaryPath: (root as NSString).appendingPathComponent("range.tmp"),
            audit: audit,
            progress: { _ in },
            commit: { range in committed.withLock { $0.append(range) } })

        #expect(try Data(contentsOf: URL(fileURLWithPath: destination)) == Data(expected))
        #expect(audit.byteCopyTiles == 3)
        #expect(audit.largestScratchBytes == 8)
        #expect(committed.withLock { $0.first?.destinationBytes } == 24)
        #expect(FakeHFURLProtocol.requestCounts["GET:source.bin"] == 1)
    }

    @Test func repeatBF16WritesExactPositiveAndNegatedBytesAcrossScratchTiles() async throws {
        let root = tmpDirForRemote("transform-bf16")
        defer { cleanUpRemote([root]) }
        resetFakeHF()
        let source: [UInt8] = [
            0x80, 0x3f,
            0x00, 0xbf,
            0x01, 0x00,
            0xc1, 0x7f,
        ]
        let positive: [UInt8] = [
            0x80, 0x3f, 0x80, 0x3f,
            0x00, 0xbf, 0x00, 0xbf,
            0x01, 0x00, 0x01, 0x00,
            0xc1, 0x7f, 0xc1, 0x7f,
        ]
        let negated: [UInt8] = [
            0x80, 0xbf, 0x80, 0xbf,
            0x00, 0x3f, 0x00, 0x3f,
            0x01, 0x80, 0x01, 0x80,
            0xc1, 0xff, 0xc1, 0xff,
        ]
        let positivePath = (root as NSString).appendingPathComponent("positive.bin")
        let negatedPath = (root as NSString).appendingPathComponent("negated.bin")
        try createTransformOutput(positivePath, size: positive.count)
        try createTransformOutput(negatedPath, size: negated.count)
        let copy = CoalescedRangeCopy(
            id: "range-00000000",
            shardID: "source.bin",
            sourceOffset: 0,
            size: UInt64(source.count),
            destinations: [
                RangeCopy(
                    shardID: "source.bin",
                    sourceOffset: 0,
                    size: UInt64(source.count),
                    destinationPath: positivePath,
                    destinationOffset: 0,
                    transform: .repeatBF16(count: 2, negated: false)),
                RangeCopy(
                    shardID: "source.bin",
                    sourceOffset: 0,
                    size: UInt64(source.count),
                    destinationPath: negatedPath,
                    destinationOffset: 0,
                    transform: .repeatBF16(count: 2, negated: true)),
            ])
        let provider = transformProvider(
            source: source,
            root: root,
            writeTileBytes: 8)
        let audit = RepackAudit()
        let committed = Mutex<[RemoteCompletedRange]>([])

        try await provider.copyBatch(
            [copy],
            completedRangeIDs: [],
            partialDirectory: root,
            temporaryPath: (root as NSString).appendingPathComponent("range.tmp"),
            audit: audit,
            progress: { _ in },
            commit: { range in committed.withLock { $0.append(range) } })

        #expect(try Data(contentsOf: URL(fileURLWithPath: positivePath)) == Data(positive))
        #expect(try Data(contentsOf: URL(fileURLWithPath: negatedPath)) == Data(negated))
        #expect(audit.byteCopyTiles == 4)
        #expect(audit.largestScratchBytes == 8)
        #expect(committed.withLock { $0.first?.destinationBytes } == 32)
        #expect(FakeHFURLProtocol.requestCounts["GET:source.bin"] == 1)
    }

    @Test func insufficientTransformScratchIsRejectedBeforeNetwork() async throws {
        let root = tmpDirForRemote("transform-scratch")
        defer { cleanUpRemote([root]) }
        resetFakeHF()
        let source = [UInt8](repeating: 0, count: 4)
        let destination = (root as NSString).appendingPathComponent("int4.bin")
        let copy = transformCopy(
            sourceSize: source.count,
            destinationPath: destination,
            transform: .unpackInt2ToInt4)
        let provider = transformProvider(
            source: source,
            root: root,
            writeTileBytes: 7)

        await #expect(throws: RepackError.self) {
            try await provider.copyBatch(
                [copy],
                completedRangeIDs: [],
                partialDirectory: root,
                temporaryPath: (root as NSString).appendingPathComponent("range.tmp"),
                audit: RepackAudit(),
                progress: { _ in },
                commit: { _ in })
        }
        #expect(FakeHFURLProtocol.requestCounts.isEmpty)
    }

    @Test func completedRangeIDSkipsDownloadAndCommit() async throws {
        let root = tmpDirForRemote("transform-completed")
        defer { cleanUpRemote([root]) }
        resetFakeHF()
        let source = [UInt8](repeating: 0xff, count: 4)
        let destination = (root as NSString).appendingPathComponent("int4.bin")
        try createTransformOutput(destination, size: 8)
        let copy = transformCopy(
            sourceSize: source.count,
            destinationPath: destination,
            transform: .unpackInt2ToInt4)
        let provider = transformProvider(
            source: source,
            root: root,
            writeTileBytes: 8)
        let committed = Mutex<[RemoteCompletedRange]>([])

        try await provider.copyBatch(
            [copy],
            completedRangeIDs: [copy.id],
            partialDirectory: root,
            temporaryPath: (root as NSString).appendingPathComponent("range.tmp"),
            audit: RepackAudit(),
            progress: { _ in },
            commit: { range in committed.withLock { $0.append(range) } })

        #expect(FakeHFURLProtocol.requestCounts.isEmpty)
        #expect(committed.withLock { $0.isEmpty })
    }

    @Test func transformedCopyResumesAfterFirstCommittedRange() async throws {
        let root = tmpDirForRemote("transform-resume")
        defer { cleanUpRemote([root]) }
        resetFakeHF()
        let source: [UInt8] = [
            0xe4, 0x1b, 0x4e, 0xb1,
            0xff, 0xff, 0xff, 0xff,
        ]
        let destination = (root as NSString).appendingPathComponent("int4.bin")
        try createTransformOutput(destination, size: 16)
        let copies = [
            CoalescedRangeCopy(
                id: "range-00000000",
                shardID: "source.bin",
                sourceOffset: 0,
                size: 4,
                destinations: [RangeCopy(
                    shardID: "source.bin",
                    sourceOffset: 0,
                    size: 4,
                    destinationPath: destination,
                    destinationOffset: 0,
                    transform: .unpackInt2ToInt4)]),
            CoalescedRangeCopy(
                id: "range-00000001",
                shardID: "source.bin",
                sourceOffset: 4,
                size: 4,
                destinations: [RangeCopy(
                    shardID: "source.bin",
                    sourceOffset: 4,
                    size: 4,
                    destinationPath: destination,
                    destinationOffset: 8,
                    transform: .unpackInt2ToInt4)]),
        ]
        let firstCommits = Mutex<[RemoteCompletedRange]>([])
        let firstProvider = transformProvider(
            source: source,
            root: root,
            writeTileBytes: 8)
        let firstRun = Task {
            try await firstProvider.copyBatch(
                copies,
                completedRangeIDs: [],
                partialDirectory: root,
                temporaryPath: (root as NSString).appendingPathComponent("range.tmp"),
                audit: RepackAudit(),
                progress: { _ in },
                commit: { range in
                    firstCommits.withLock { $0.append(range) }
                    withUnsafeCurrentTask { $0?.cancel() }
                })
        }

        await #expect(throws: CancellationError.self) {
            try await firstRun.value
        }
        #expect(firstCommits.withLock { $0.map(\.id) } == ["range-00000000"])
        #expect(FakeHFURLProtocol.requestedRanges["source.bin"] == ["bytes=0-3"])

        let resumedCommits = Mutex<[RemoteCompletedRange]>([])
        try await transformProvider(
            source: source,
            root: root,
            writeTileBytes: 8
        ).copyBatch(
            copies,
            completedRangeIDs: ["range-00000000"],
            partialDirectory: root,
            temporaryPath: (root as NSString).appendingPathComponent("range.tmp"),
            audit: RepackAudit(),
            progress: { _ in },
            commit: { range in resumedCommits.withLock { $0.append(range) } })

        #expect(resumedCommits.withLock { $0.map(\.id) } == ["range-00000001"])
        #expect(FakeHFURLProtocol.requestedRanges["source.bin"]
            == ["bytes=0-3", "bytes=4-7"])
        let expected: [UInt8] = [
            0x10, 0x32, 0x23, 0x01, 0x32, 0x10, 0x01, 0x23,
            0x33, 0x33, 0x33, 0x33, 0x33, 0x33, 0x33, 0x33,
        ]
        #expect(try Data(contentsOf: URL(fileURLWithPath: destination)) == Data(expected))
    }
}

private func transformCopy(
    sourceSize: Int,
    destinationPath: String,
    transform: RangeCopyTransform
) -> CoalescedRangeCopy {
    CoalescedRangeCopy(
        id: "range-00000000",
        shardID: "source.bin",
        sourceOffset: 0,
        size: UInt64(sourceSize),
        destinations: [RangeCopy(
            shardID: "source.bin",
            sourceOffset: 0,
            size: UInt64(sourceSize),
            destinationPath: destinationPath,
            destinationOffset: 0,
            transform: transform)])
}

private func transformProvider(
    source: [UInt8],
    root: String,
    writeTileBytes: Int
) -> HTTPRangeSourceByteProvider {
    FakeHFURLProtocol.files["source.bin"] = Data(source)
    let commit = FakeHFURLProtocol.commit
    let remote = HuggingFaceRemoteSource(
        repoID: "owner/model",
        requestedRevision: "main",
        resolvedCommit: commit,
        downloadSession: fakeHFSession(),
        baseURL: URL(string: "https://hf.test")!,
        tempDirectory: root,
        retryPolicy: RemoteRetryPolicy(attempts: 1, baseDelayNs: 0))
    let info = RemoteFileInfo(
        filename: "source.bin",
        resolvedCommit: commit,
        size: UInt64(source.count),
        etag: "source-etag",
        xetHash: nil,
        acceptsRanges: true)
    return HTTPRangeSourceByteProvider(
        remote: remote,
        files: ["source.bin": info],
        writeTileBytes: writeTileBytes)
}

private func createTransformOutput(_ path: String, size: Int) throws {
    let descriptor = try Posix.openCreateRW(path)
    defer { close(descriptor) }
    try Posix.ftruncate(descriptor, path: path, size: UInt64(size))
}

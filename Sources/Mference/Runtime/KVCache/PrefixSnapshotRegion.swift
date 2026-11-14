import Darwin
import Foundation

/// Why a prefix snapshot could not be taken or restored. Any of these leaves
/// the live state untouched, except where a restore says otherwise: a restore
/// that throws after it started copying requires the owner's `reset()`.
public enum PrefixSnapshotError: Error, Equatable, CustomStringConvertible {
    /// The runner or its state layout cannot be snapshotted (sliding rings,
    /// paged or compressed attention, an unsupported family).
    case unsupported(String)
    /// The region is smaller than the bytes the copy needs.
    case regionTooSmall(needed: Int, available: Int)
    /// The requested rows or position do not fit the cache.
    case outOfRange(String)
    /// Anonymous memory for a region could not be mapped.
    case allocationFailed(bytes: Int)

    public var description: String {
        switch self {
        case .unsupported(let why): return "prefix snapshot unsupported: \(why)"
        case .regionTooSmall(let needed, let available):
            return "prefix snapshot region holds \(available) bytes, \(needed) needed"
        case .outOfRange(let why): return "prefix snapshot out of range: \(why)"
        case .allocationFailed(let bytes): return "prefix snapshot could not map \(bytes) bytes"
        }
    }
}

/// Anonymous, page-aligned memory that holds one prefix snapshot payload: a
/// copy of full-attention KV rows or of recurrent state. Never Swift heap
/// (the bounded-memory invariant): the bytes are `mmap`ed on creation and
/// `munmap`ed when the region is released. The CPU alone reads and writes it;
/// a restore copies it back into the runner's own buffers.
///
/// `@unchecked Sendable`: the owner serializes access (the server's single
/// generation gate); the region itself holds no mutable Swift state.
public final class PrefixSnapshotRegion: @unchecked Sendable {
    public let base: UnsafeMutableRawPointer
    /// Bytes mapped, rounded up to whole pages.
    public let mappedLength: Int
    /// Bytes the payload occupies (`<= mappedLength`).
    public let byteCount: Int

    public init(byteCount: Int) throws {
        precondition(byteCount > 0, "prefix snapshot region must be non-empty")
        let page = Int(getpagesize())
        let length = (byteCount + page - 1) / page * page
        let mapped = mmap(nil, length, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0)
        guard mapped != MAP_FAILED, let mapped else {
            throw PrefixSnapshotError.allocationFailed(bytes: length)
        }
        self.base = mapped
        self.mappedLength = length
        self.byteCount = byteCount
    }

    deinit { munmap(base, mappedLength) }
}

import Darwin
import Foundation

extension Native {
    /// A read-only, shared file mapping for an `.mfpack`'s `weights.bin`.
    ///
    /// The pages are file-backed and clean, so they never count against the
    /// Swift heap and the kernel can drop and re-read them under pressure
    /// (CLAUDE.md "Bounded memory"). `mappedLength` is `size` rounded up to the
    /// VM page size, which is what `makeBuffer(bytesNoCopy:)` requires.
    public final class PackMappedFile: @unchecked Sendable {
        public let url: URL
        public let base: UnsafeRawPointer
        /// Bytes in the file.
        public let size: Int
        /// Bytes mapped (`size` rounded up to the page size, at least one page).
        public let mappedLength: Int

        public init(url: URL) throws {
            self.url = url
            let fd = open(url.path, O_RDONLY | O_CLOEXEC)
            guard fd >= 0 else { throw NativeError.missing("\(url.path) (errno \(errno))") }
            defer { close(fd) }
            var st = stat()
            guard fstat(fd, &st) == 0 else { throw NativeError.resource("stat \(url.path) (errno \(errno))") }
            guard (st.st_mode & S_IFMT) == S_IFREG else {
                throw NativeError.integrity("\(url.path) is not a regular file")
            }
            size = Int(st.st_size)
            let page = Int(getpagesize())
            mappedLength = max(page, (size + page - 1) / page * page)
            guard let p = mmap(nil, mappedLength, PROT_READ, MAP_SHARED, fd, 0), p != MAP_FAILED else {
                throw NativeError.resource("mmap \(url.path) (errno \(errno))")
            }
            base = UnsafeRawPointer(p)
        }

        deinit { munmap(UnsafeMutableRawPointer(mutating: base), mappedLength) }

        /// The bytes `[offset, offset + count)`; the caller has checked the range.
        public func bytes(offset: Int, count: Int) -> UnsafeRawBufferPointer {
            UnsafeRawBufferPointer(start: base + offset, count: count)
        }
    }
}

import Darwin
import Foundation

/// An IP address, parsed or resolved.
public enum IPAddress: Sendable, Hashable, CustomStringConvertible {
    case v4(UInt32)
    case v6([UInt8])

    /// Parses a literal ("10.0.0.1", "::1", "[fe80::1]"); nil for names.
    public init?(_ string: String) {
        var text = string
        if text.hasPrefix("["), text.hasSuffix("]") { text = String(text.dropFirst().dropLast()) }
        if let zone = text.firstIndex(of: "%") { text = String(text[..<zone]) }
        var v4 = in_addr()
        if inet_pton(AF_INET, text, &v4) == 1 {
            self = .v4(UInt32(bigEndian: v4.s_addr))
            return
        }
        var v6 = in6_addr()
        if inet_pton(AF_INET6, text, &v6) == 1 {
            self = .v6(withUnsafeBytes(of: v6) { Array($0) })
            return
        }
        return nil
    }

    init?(sockaddr pointer: UnsafePointer<sockaddr>) {
        switch Int32(pointer.pointee.sa_family) {
        case AF_INET:
            let address = pointer.withMemoryRebound(to: sockaddr_in.self, capacity: 1) { $0.pointee.sin_addr }
            self = .v4(UInt32(bigEndian: address.s_addr))
        case AF_INET6:
            let address = pointer.withMemoryRebound(to: sockaddr_in6.self, capacity: 1) { $0.pointee.sin6_addr }
            self = .v6(withUnsafeBytes(of: address) { Array($0) })
        default:
            return nil
        }
    }

    public var description: String {
        switch self {
        case .v4(let value):
            return [24, 16, 8, 0].map { String((value >> $0) & 0xFF) }.joined(separator: ".")
        case .v6(let bytes):
            return stride(from: 0, to: 16, by: 2).map { String(format: "%x", Int(bytes[$0]) << 8 | Int(bytes[$0 + 1])) }
                .joined(separator: ":")
        }
    }

    /// True only for globally routable unicast addresses: not private,
    /// loopback, link-local, CGNAT, ULA, multicast, documentation, benchmark,
    /// reserved, or an IPv4 address embedded in IPv6 that is any of those.
    public var isPublic: Bool {
        switch self {
        case .v4(let a):
            func inRange(_ base: UInt32, _ bits: UInt32) -> Bool {
                let mask: UInt32 = bits == 0 ? 0 : ~UInt32(0) << (32 - bits)
                return a & mask == base & mask
            }
            let blocked: [(UInt32, UInt32)] = [
                (0x0000_0000, 8),   // 0.0.0.0/8 "this network"
                (0x0A00_0000, 8),   // 10/8
                (0x6440_0000, 10),  // 100.64/10 CGNAT
                (0x7F00_0000, 8),   // 127/8 loopback
                (0xA9FE_0000, 16),  // 169.254/16 link-local (cloud metadata)
                (0xAC10_0000, 12),  // 172.16/12
                (0xC000_0000, 24),  // 192.0.0/24 IETF
                (0xC000_0200, 24),  // 192.0.2/24 TEST-NET-1
                (0xC058_6300, 24),  // 192.88.99/24 6to4 relay
                (0xC0A8_0000, 16),  // 192.168/16
                (0xC612_0000, 15),  // 198.18/15 benchmarking
                (0xC633_6400, 24),  // 198.51.100/24 TEST-NET-2
                (0xCB00_7100, 24),  // 203.0.113/24 TEST-NET-3
                (0xE000_0000, 4),   // 224/4 multicast
                (0xF000_0000, 4),   // 240/4 reserved and broadcast
            ]
            return !blocked.contains { inRange($0.0, $0.1) }
        case .v6(let b):
            guard b.count == 16 else { return false }
            let embedded = IPAddress.v4(UInt32(b[12]) << 24 | UInt32(b[13]) << 16 | UInt32(b[14]) << 8 | UInt32(b[15]))
            if b[0..<10].allSatisfy({ $0 == 0 }) {
                // ::/128, ::1, IPv4-compatible ::a.b.c.d, IPv4-mapped ::ffff:a.b.c.d
                if b[10] == 0xFF, b[11] == 0xFF { return embedded.isPublic }
                return false
            }
            if b[0] == 0x00, b[1] == 0x64, b[2] == 0xFF, b[3] == 0x9B { return embedded.isPublic } // 64:ff9b::/96 NAT64
            if b[0] == 0x20, b[1] == 0x02 {                                                        // 2002::/16 6to4
                let v4 = IPAddress.v4(UInt32(b[2]) << 24 | UInt32(b[3]) << 16 | UInt32(b[4]) << 8 | UInt32(b[5]))
                return v4.isPublic
            }
            if b[0] & 0xFE == 0xFC { return false }                  // fc00::/7 ULA
            if b[0] == 0xFE, b[1] & 0xC0 == 0x80 { return false }    // fe80::/10 link-local
            if b[0] == 0xFE, b[1] & 0xC0 == 0xC0 { return false }    // fec0::/10 site-local
            if b[0] == 0xFF { return false }                          // ff00::/8 multicast
            if b[0] == 0x20, b[1] == 0x01, b[2] == 0x0D, b[3] == 0xB8 { return false } // 2001:db8::/32 docs
            if b[0] == 0x01, b[1] == 0x00, b[2..<8].allSatisfy({ $0 == 0 }) { return false } // 100::/64 discard
            return b[0] & 0xE0 == 0x20                                // global unicast is 2000::/3
        }
    }
}

/// Which hosts the page reader may contact (spec 4.4, review finding 1).
public enum HostPolicy {
    /// Names that only exist on a local network or this Mac.
    static let localSuffixes = [".local", ".localhost", ".localdomain", ".internal", ".intranet", ".lan",
                                ".home", ".home.arpa", ".corp", ".private", ".arpa"]

    /// Whether `host` may be fetched at all, before resolving it: IP literals
    /// must be public; names must have a dot, a public-looking suffix, and not
    /// be a numeric form some resolvers read as an address ("0177.1", "0x7f.1",
    /// "2130706433").
    public static func isAllowedHost(_ rawHost: String) -> Bool {
        let host = rawHost.lowercased().trimmingCharacters(in: CharacterSet(charactersIn: "."))
        guard !host.isEmpty else { return false }
        // "0177.0.0.1": inet_pton reads decimal 177, some resolvers read octal 127.
        let dotted = host.split(separator: ".", omittingEmptySubsequences: false)
        if dotted.allSatisfy({ !$0.isEmpty && $0.allSatisfy(\.isNumber) }),
           dotted.contains(where: { $0.count > 1 && $0.hasPrefix("0") }) { return false }
        if let address = IPAddress(host) { return address.isPublic }
        if host.hasPrefix("[") || host.contains(":") { return false }
        guard host.contains("."), host != "localhost" else { return false }
        if localSuffixes.contains(where: { host.hasSuffix($0) }) { return false }
        let labels = host.split(separator: ".")
        let numericLike = labels.allSatisfy { label in
            label.allSatisfy(\.isNumber) || (label.hasPrefix("0x") && label.dropFirst(2).allSatisfy(\.isHexDigit))
        }
        if numericLike { return false }
        if let last = labels.last, last.allSatisfy(\.isNumber) { return false }
        return true
    }

    /// Resolves `host` and throws unless every address is public (so a name
    /// that points into the LAN, or one of whose addresses does, is refused).
    public typealias Resolver = @Sendable (String) async throws -> [IPAddress]

    public static let systemResolver: Resolver = { host in
        try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                continuation.resume(with: Result { try resolve(host) })
            }
        }
    }

    static func resolve(_ host: String) throws -> [IPAddress] {
        var hints = addrinfo()
        hints.ai_family = AF_UNSPEC
        hints.ai_socktype = SOCK_STREAM
        var result: UnsafeMutablePointer<addrinfo>?
        let status = getaddrinfo(host, nil, &hints, &result)
        guard status == 0, let first = result else {
            throw WebError.transport("could not find \(host)")
        }
        defer { freeaddrinfo(first) }
        var addresses: [IPAddress] = []
        var node: UnsafeMutablePointer<addrinfo>? = first
        while let current = node {
            if let sockaddr = current.pointee.ai_addr, let address = IPAddress(sockaddr: sockaddr) {
                addresses.append(address)
            }
            node = current.pointee.ai_next
        }
        return addresses
    }
}

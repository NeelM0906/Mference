import Foundation

/// Which peer addresses the link talks to: the local network only (decision
/// D9, same Wi-Fi). The Mac's listener refuses a connection from any other
/// address before it reads a byte, and the phone refuses to connect to one.
///
/// Local means: IPv4 private (10/8, 172.16/12, 192.168/16) and link-local
/// (169.254/16); IPv6 link-local (fe80::/10) and unique local (fc00::/7); an
/// IPv4 address in IPv4-mapped IPv6 form by its IPv4 rule. Loopback (127/8,
/// ::1) counts only when `allowLoopback` is set, which the tests' loopback
/// transport uses. Everything else is refused, including carrier-grade NAT
/// and VPN ranges (100.64/10): those are not the same Wi-Fi.
public struct LinkAddressPolicy: Sendable {
    public var allowLoopback: Bool

    public init(allowLoopback: Bool = false) {
        self.allowLoopback = allowLoopback
    }

    /// `host` is a literal IP address, optionally with an IPv6 zone
    /// ("fe80::1%en0"). Host names are refused: the transport checks the
    /// address it actually connected to, never a name.
    public func isAllowed(host: String) -> Bool {
        let address = host.split(separator: "%", maxSplits: 1).first.map(String.init) ?? host
        if let v4 = Self.ipv4(address) { return allows(v4: v4) }
        if let v6 = Self.ipv6(address) {
            // IPv4-mapped (::ffff:a.b.c.d).
            if v6[0..<10].allSatisfy({ $0 == 0 }) && v6[10] == 0xFF && v6[11] == 0xFF {
                return allows(v4: Array(v6[12..<16]))
            }
            if v6[0..<15].allSatisfy({ $0 == 0 }) && v6[15] == 1 { return allowLoopback }
            if v6[0] == 0xFE && (v6[1] & 0xC0) == 0x80 { return true }  // fe80::/10
            if (v6[0] & 0xFE) == 0xFC { return true }                    // fc00::/7
            return false
        }
        return false
    }

    private func allows(v4 a: [UInt8]) -> Bool {
        switch (a[0], a[1]) {
        case (10, _): true
        case (172, 16...31): true
        case (192, 168): true
        case (169, 254): true
        case (127, _): allowLoopback
        default: false
        }
    }

    private static func ipv4(_ text: String) -> [UInt8]? {
        var address = in_addr()
        guard inet_pton(AF_INET, text, &address) == 1 else { return nil }
        return withUnsafeBytes(of: address.s_addr) { Array($0) }
    }

    private static func ipv6(_ text: String) -> [UInt8]? {
        var address = in6_addr()
        guard inet_pton(AF_INET6, text, &address) == 1 else { return nil }
        return withUnsafeBytes(of: address) { Array($0) }
    }
}

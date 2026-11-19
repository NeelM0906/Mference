import Foundation

extension Native {
    /// `std::mt19937` (32-bit Mersenne Twister) as libc++ implements it, plus
    /// libc++'s `uniform_real_distribution<double>(0, 1)` draw. The seeded draw
    /// of the sampler chain (`dist`) uses exactly this generator, so the
    /// sequence must match llama.cpp's bit for bit.
    public struct SamplingMT19937 {
        private static let n = 624
        private static let m = 397
        private var state: ContiguousArray<UInt32>
        private var index: Int

        public init(seed: UInt32) {
            state = ContiguousArray(repeating: 0, count: Self.n)
            state[0] = seed
            for i in 1..<Self.n {
                state[i] = 1_812_433_253 &* (state[i - 1] ^ (state[i - 1] >> 30)) &+ UInt32(i)
            }
            index = Self.n
        }

        private mutating func twist() {
            state.withUnsafeMutableBufferPointer { s in
                @inline(__always) func mix(_ y: UInt32, _ far: UInt32) -> UInt32 {
                    far ^ (y >> 1) ^ ((y & 1) != 0 ? 0x9908_B0DF : 0)
                }
                let n = Self.n, m = Self.m
                var i = 0
                while i < n - m {
                    let y = (s[i] & 0x8000_0000) | (s[i + 1] & 0x7FFF_FFFF)
                    s[i] = mix(y, s[i + m])
                    i += 1
                }
                while i < n - 1 {
                    let y = (s[i] & 0x8000_0000) | (s[i + 1] & 0x7FFF_FFFF)
                    s[i] = mix(y, s[i + m - n])
                    i += 1
                }
                let y = (s[n - 1] & 0x8000_0000) | (s[0] & 0x7FFF_FFFF)
                s[n - 1] = mix(y, s[m - 1])
            }
            index = 0
        }

        public mutating func next() -> UInt32 {
            if index >= Self.n { twist() }
            var y = state[index]
            index += 1
            y ^= y >> 11
            y ^= (y << 7) & 0x9D2C_5680
            y ^= (y << 15) & 0xEFC6_0000
            y ^= y >> 18
            return y
        }

        /// libc++ `generate_canonical<double, 53>` from two draws.
        public mutating func nextCanonical() -> Double {
            let r = 4_294_967_296.0
            var sum = Double(next())
            sum += Double(next()) * r
            return sum / (r * r)
        }
    }
}

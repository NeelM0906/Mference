import Foundation
import Hub
import Testing
@testable import Mference

/// `GemmaDecoding` and `ByteFallbackRun` driven by `Config` literals and raw
/// bytes: no tokenizer download and no fixture required.
@Suite("Gemma decoding rules")
struct GemmaDecodingTests {
    private static let pinned: Config = [
        "type": "Sequence",
        "decoders": [
            ["type": "Replace", "pattern": ["String": "▁"], "content": " "] as Config,
            ["type": "ByteFallback"] as Config,
            ["type": "Fuse"] as Config,
        ] as Config,
    ]

    private static func data(decoder: Config) -> Config {
        ["decoder": decoder]
    }

    // MARK: - Decoder declaration

    @Test("The pinned Gemma decoder sequence is recognised")
    func pinnedSequenceAccepted() {
        #expect(GemmaDecoding.declaresPinnedDecoder(Self.data(decoder: Self.pinned)))
    }

    @Test("Any other decoder declaration is refused", arguments: [
        // Metaspace instead of the Replace chain.
        ["type": "Metaspace", "replacement": "▁", "prependScheme": "first"] as Config,
        // Fuse missing.
        ["type": "Sequence", "decoders": [
            ["type": "Replace", "pattern": ["String": "▁"], "content": " "] as Config,
            ["type": "ByteFallback"] as Config,
        ] as Config] as Config,
        // Replace with another pattern.
        ["type": "Sequence", "decoders": [
            ["type": "Replace", "pattern": ["String": "_"], "content": " "] as Config,
            ["type": "ByteFallback"] as Config,
            ["type": "Fuse"] as Config,
        ] as Config] as Config,
        // Replace with other content.
        ["type": "Sequence", "decoders": [
            ["type": "Replace", "pattern": ["String": "▁"], "content": ""] as Config,
            ["type": "ByteFallback"] as Config,
            ["type": "Fuse"] as Config,
        ] as Config] as Config,
        // An extra step.
        ["type": "Sequence", "decoders": [
            ["type": "Replace", "pattern": ["String": "▁"], "content": " "] as Config,
            ["type": "ByteFallback"] as Config,
            ["type": "Fuse"] as Config,
            ["type": "Strip", "content": " ", "start": 1, "stop": 0] as Config,
        ] as Config] as Config,
        // Steps out of order.
        ["type": "Sequence", "decoders": [
            ["type": "ByteFallback"] as Config,
            ["type": "Replace", "pattern": ["String": "▁"], "content": " "] as Config,
            ["type": "Fuse"] as Config,
        ] as Config] as Config,
    ])
    func foreignDecoderRefused(_ decoder: Config) {
        #expect(!GemmaDecoding.declaresPinnedDecoder(Self.data(decoder: decoder)))
    }

    @Test("A tokenizer.json without a decoder is refused")
    func missingDecoderRefused() {
        let data: Config = ["model": ["type": "BPE"] as Config]
        #expect(!GemmaDecoding.declaresPinnedDecoder(data))
    }

    // MARK: - Special-token IDs

    @Test("Special IDs are the added tokens flagged special, read defensively")
    func specialIDsFromAddedTokens() {
        let data: Config = [
            "added_tokens": [
                ["id": 0, "content": "<pad>", "special": true] as Config,
                ["id": 2, "content": "<bos>", "special": true] as Config,
                // Added but not special: the library keeps it when skipping.
                ["id": 7, "content": "<think>", "special": false] as Config,
                // No flag at all counts as not special, as in the library.
                ["id": 8, "content": "<plain>"] as Config,
                // Outside Int32: skipped instead of trapping on conversion.
                ["id": 3_000_000_000, "content": "<huge>", "special": true] as Config,
                // No id: skipped, as the library skips it.
                ["content": "<noid>", "special": true] as Config,
            ] as Config,
        ]
        #expect(GemmaDecoding.specialTokenIDs(data) == Set<Int32>([0, 2]))
    }

    // MARK: - Fragments

    @Test("Fragments map the metaspace to a space and nothing else", arguments: [
        ("▁the", " the"),
        ("the", "the"),
        ("▁▁▁▁", "    "),
        ("a▁b", "a b"),
        ("<|channel>", "<|channel>"),
        ("_under", "_under"),
    ])
    func fragmentMapsMetaspace(_ probe: (token: String, text: String)) {
        #expect(GemmaDecoding.fragment(probe.token) == probe.text)
    }

    @Test("Only well-formed <0xXX> tokens are byte-fallback tokens", arguments: [
        ("<0x00>", UInt8?.some(0x00)),
        ("<0xE2>", UInt8?.some(0xE2)),
        ("<0xff>", UInt8?.some(0xFF)),
        ("<0xG0>", UInt8?.none),
        ("<0xE2", UInt8?.none),
        ("<0x0E2>", UInt8?.none),
        ("0xE2>", UInt8?.none),
        ("the", UInt8?.none),
    ])
    func byteValueParsesOnlyByteTokens(_ probe: (token: String, byte: UInt8?)) {
        #expect(GemmaDecoding.byteValue(probe.token) == probe.byte)
    }

    // MARK: - Byte-fallback runs

    private static func stream(_ bytes: [UInt8]) -> (pushed: [String], committed: String) {
        var run = ByteFallbackRun()
        let pushed = bytes.map { run.push($0) }
        return (pushed, run.commit())
    }

    @Test("A valid run is held until it commits whole")
    func validRunHeldUntilCommit() {
        let result = Self.stream([0xF0, 0x9F, 0x98, 0x80, 0xC3, 0xA9])
        #expect(result.pushed == ["", "", "", "", "", ""])
        #expect(result.committed == "😀é")
    }

    @Test("An invalid run streams one replacement character per byte")
    func invalidRunStreamsLive() {
        // A stray continuation byte can never become valid.
        var result = Self.stream([0x80, 0x80, 0x41])
        #expect(result.pushed == ["\u{FFFD}", "\u{FFFD}", "\u{FFFD}"])
        #expect(result.committed == "")

        // The byte that breaks a held lead releases the lead's byte as well.
        result = Self.stream([0xC3, 0xFF])
        #expect(result.pushed == ["", "\u{FFFD}\u{FFFD}"])
        #expect(result.committed == "")
    }

    @Test("First-continuation limits follow RFC 3629", arguments: [
        // E0 needs A0...BF (else overlong).
        ([UInt8]([0xE0, 0x80]), true),
        ([UInt8]([0xE0, 0xA0]), false),
        // ED needs 80...9F (else a UTF-16 surrogate).
        ([UInt8]([0xED, 0xA0]), true),
        ([UInt8]([0xED, 0x9F]), false),
        // F0 needs 90...BF (else overlong).
        ([UInt8]([0xF0, 0x80]), true),
        ([UInt8]([0xF0, 0x90]), false),
        // F4 needs 80...8F (else above U+10FFFF).
        ([UInt8]([0xF4, 0x90]), true),
        ([UInt8]([0xF4, 0x8F]), false),
        // Overlong two-byte leads and bytes above F4 never start a scalar.
        ([UInt8]([0xC0]), true),
        ([UInt8]([0xC1]), true),
        ([UInt8]([0xF5]), true),
    ])
    func continuationLimits(_ probe: (bytes: [UInt8], poisons: Bool)) {
        let last = Self.stream(probe.bytes).pushed.last ?? ""
        #expect(!last.isEmpty == probe.poisons, "bytes \(probe.bytes)")
    }

    @Test("An incomplete trailing sequence invalidates the whole run")
    func incompleteTailInvalidatesRun() {
        // Reference semantics: the run is validated as a whole, so a valid
        // prefix does not survive a truncated final scalar.
        #expect(Self.stream([0xE2, 0x82]).committed == "\u{FFFD}\u{FFFD}")
        #expect(Self.stream([0xC3, 0xA9, 0xE2, 0x82]).committed
            == String(repeating: "\u{FFFD}", count: 4))
    }

    @Test("Commit resets the run, including a poisoned one")
    func commitResetsRun() {
        var run = ByteFallbackRun()
        #expect(run.push(0x80) == "\u{FFFD}")
        #expect(run.commit() == "")
        #expect(run.push(0xC3) == "")
        #expect(run.push(0xA9) == "")
        #expect(run.commit() == "é")
        #expect(run.commit() == "")
    }

    // MARK: - Which tokenizers take the lossless path

    @Test("A non-Gemma dialect never takes the lossless path")
    func nonGemmaDialectKeepsLibraryDecode() async throws {
        let chatML = try await MFTokenizer.load(from: ChatMLTemplateTests.fixtureFolder())
        #expect(chatML.dialect == .chatml)
        let rebuilt = try MFTokenizer(tokenizer: chatML.tokenizer, family: nil,
                                      tokenizerData: Self.data(decoder: Self.pinned))
        #expect(rebuilt.losslessGemmaSpecialTokenIDs == nil)
    }
}

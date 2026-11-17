import Testing
@testable import Mference

/// `--verify` values shared by the CLI and the server.
@Suite struct ModelIntegrityPolicyFlagTests {
    @Test func everyPolicyHasAFlagValue() {
        #expect(ModelIntegrityPolicy(verifyFlag: "auto") == .trustedReceiptWhenValid)
        #expect(ModelIntegrityPolicy(verifyFlag: "full-sha256") == .fullSha256)
        #expect(ModelIntegrityPolicy(verifyFlag: "trusted-receipt") == .sizeCheckTrustedReceipt)
    }

    @Test(arguments: ["", "Auto", "sha256", "trusted", "full"])
    func otherValuesAreNotPolicies(value: String) {
        #expect(ModelIntegrityPolicy(verifyFlag: value) == nil)
    }

    /// The `--verify auto` fallback notice is one stderr line that names the
    /// failed receipt check, even when that check's detail spans lines.
    @Test func autoFallbackNoticeIsOneLineWithTheReason() {
        let notice = ModelIntegrityPolicy.autoFallbackNotice(detail: "verified-install.json: bad\nkey")
        #expect(notice == "notice: --verify auto: install receipt did not validate "
            + "(verified-install.json: bad key); hashing routed-expert files with full SHA-256 on first use")
        let hasNewline = notice.contains { $0.isNewline }
        #expect(!hasNewline)
    }
}

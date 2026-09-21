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
}

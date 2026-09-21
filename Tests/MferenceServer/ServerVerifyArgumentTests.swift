import Testing
import Mference
@testable import MferenceServerCore

@Suite("Server model verification") struct ServerVerifyArgumentTests {
    @Test func theServerUsesTheInstallReceiptWhenItIsValidByDefault() throws {
        #expect(try ServerArguments.parse(["--model", "m.gturbo"]).verification == .trustedReceiptWhenValid)
        #expect(try ServerArguments.parse(["--library"]).verification == .trustedReceiptWhenValid)
    }

    @Test func fullHashingIsAnExplicitChoice() throws {
        let arguments = try ServerArguments.parse(["--model", "m.gturbo", "--verify", "full-sha256"])
        #expect(arguments.verification == .fullSha256)
    }

    @Test func theOtherPoliciesCanBeNamedToo() throws {
        #expect(try ServerArguments.parse(["--model", "m.gturbo", "--verify", "trusted-receipt"])
            .verification == .sizeCheckTrustedReceipt)
        #expect(try ServerArguments.parse(["--model", "m.gturbo", "--verify", "auto"])
            .verification == .trustedReceiptWhenValid)
    }

    @Test(arguments: ["", "none", "sha256", "full"])
    func unknownPoliciesAreRejected(value: String) {
        #expect(throws: ServerArgumentError.self) {
            _ = try ServerArguments.parse(["--model", "m.gturbo", "--verify", value])
        }
    }
}

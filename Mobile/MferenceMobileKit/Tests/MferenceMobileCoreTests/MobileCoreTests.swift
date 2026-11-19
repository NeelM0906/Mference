import Testing
import MferenceLink
@testable import MferenceMobileCore

@Suite("Mobile core")
struct MobileCoreTests {
    @Test func speaksTheSharedLinkProtocol() {
        #expect(MobileCore.linkProtocolVersion == LinkProtocol.version)
    }
}

import Testing
import Mference
@testable import MferenceAppCore

@Suite struct AppModelInstallDescriptorTests {
    @Test func mapleDescriptorMatchesThePinnedInstallContract() {
        let maple = AppModelInstallDescriptor.maple
        #expect(maple.family == .maple)
        #expect(maple.repoID == "deepgrove/maple-preview-2bit-mlx")
        #expect(maple.revision == "361db5da5e74ff6fcdd852d478e1f266ce11013a")
        #expect(maple.sourceIndexSHA256 ==
            "56000110535c5023b43209a5c142035e12c1cde7b1118759cc9f86335d46ef95")
        #expect(maple.approximateDownloadBytes == 5_330_000_000)
        #expect(maple.installedBytes == 6_650_000_000)
        #expect(maple.installDirectoryName == "maple.gturbo")
        #expect(AppModelInstallDescriptor.descriptor(for: .maple) == maple)
    }

    @Test func everyExistingFamilyStillResolvesItsOwnDescriptor() {
        let expected: [(ModelFamily, AppModelInstallDescriptor)] = [
            (.gemma4, .default),
            (.qwen36, .qwen36),
            (.deepseekV4Flash, .deepseekV4Flash),
            (.inklingSmall, .inklingSmall),
        ]

        for (family, descriptor) in expected {
            #expect(AppModelInstallDescriptor.descriptor(for: family) == descriptor)
            #expect(descriptor.family == family)
            #expect(descriptor.installDirectoryName.hasSuffix(".gturbo"))
        }
    }
}

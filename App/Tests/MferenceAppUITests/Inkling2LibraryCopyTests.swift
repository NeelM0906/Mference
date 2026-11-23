import Foundation
import MferenceAppCore
@testable import MferenceAppUI
import Testing

/// The Inkling 2 Mini copy in the model library and picker: no claim that
/// its experts stream from the SSD (only partly true on 32 GB Macs, untrue
/// from 48 GB, where every layer is resident).
@Suite struct Inkling2LibraryCopyTests {
    @Test func libraryAndPickerCopyMakeNoSSDClaim() {
        #expect(LiveModelLibraryModel.formatDetail(.inkling2Mini) == "Mference install of Unsloth UD-Q4_K_XL")
        #expect(LiveModelLibraryModel.variantName(.inkling2Mini) == "Mini")
        #expect(AppModel.inkling2Mini.summary == "Deep-tier mixture-of-experts model.")
        for text in [LiveModelLibraryModel.formatDetail(.inkling2Mini), AppModel.inkling2Mini.summary] {
            #expect(!text.contains("SSD"))
        }
        // Qwen 3.6 does stream its experts, and still says so.
        #expect(LiveModelLibraryModel.formatDetail(.qwen36) == "Mference install, experts stream from the SSD")
    }
}

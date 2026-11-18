import Foundation

// Must run before SwiftUI creates any Metal device, or long prefills can hit
// the GPU watchdog (spec 4.3.2).
setenv("AGX_RELAX_CDM_CTXSTORE_TIMEOUT", "1", 1)

MferenceMacApp.main()

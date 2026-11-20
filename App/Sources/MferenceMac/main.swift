import Foundation
import MferenceAppCore

// Must run before SwiftUI creates any Metal device, or long prefills can hit
// the GPU watchdog (spec 4.3.2).
setenv("AGX_RELAX_CDM_CTXSTORE_TIMEOUT", "1", 1)

// Find's file reader runs as a helper copy of this executable, so one bad
// file cannot take the app down (FindExtractionWorker).
if CommandLine.arguments.contains(HelperFindExtractor.argument) {
    HelperFindExtractor.runHelper()
}

if CommandLine.arguments.contains("--verify-bundle") {
    let failures = BundleSelfCheck.run()
    for failure in failures {
        FileHandle.standardError.write(Data("verify-bundle: \(failure)\n".utf8))
    }
    print(failures.isEmpty ? "verify-bundle: ok" : "verify-bundle: \(failures.count) failure(s)")
    exit(failures.isEmpty ? 0 : 1)
}

MferenceMacApp.main()

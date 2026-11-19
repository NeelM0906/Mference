import Foundation

// Must run before SwiftUI creates any Metal device, or long prefills can hit
// the GPU watchdog (spec 4.3.2).
setenv("AGX_RELAX_CDM_CTXSTORE_TIMEOUT", "1", 1)

if CommandLine.arguments.contains("--verify-bundle") {
    let failures = BundleSelfCheck.run()
    for failure in failures {
        FileHandle.standardError.write(Data("verify-bundle: \(failure)\n".utf8))
    }
    print(failures.isEmpty ? "verify-bundle: ok" : "verify-bundle: \(failures.count) failure(s)")
    exit(failures.isEmpty ? 0 : 1)
}

MferenceMacApp.main()

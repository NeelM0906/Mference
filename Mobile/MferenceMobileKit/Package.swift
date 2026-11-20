// swift-tools-version: 6.2
import PackageDescription

// The iPhone and iPhone Duo app (design locked 2026-11-19 on the mobile design
// canvas). The app target itself is Mobile/MferenceMobile.xcodeproj, a thin
// @main over MferenceMobileUI. Design tokens come from the Mac app's
// MferenceDesign target, so both apps share one token set.
//
// MferenceMobileCore holds the phone-only services (model storage and
// download, the Mac-link client, camera, Files export, memory budget, app
// lifecycle); no SwiftUI views. MferenceMobileUI is SwiftUI only. The shared
// Mac core (MferenceAppCore) joins once it builds for iOS (decision M5).
// MferenceMobileSamples holds the sample screen models (SampleCalendarModel,
// SampleScanModel, ...) for previews, tests and, until each owner's live model
// lands, the skeleton app. MferenceMobileUI and MferenceMobileCore never
// import it, and it does not ship at C-final.
// This target layout is a mobile contract (Mobile/CONTRACTS.md).
//
// iOS only: build and test with xcodebuild against an iOS Simulator (see
// Mobile/README.md). This first slice runs on sample data and a scripted
// reply source; it never loads a model.

let settings: [SwiftSetting] = [
    .treatAllWarnings(as: .error),
    .defaultIsolation(MainActor.self),
]

let package = Package(
    name: "MferenceMobileKit",
    platforms: [
        .iOS(.v26),
    ],
    products: [
        .library(name: "MferenceMobileCore", targets: ["MferenceMobileCore"]),
        .library(name: "MferenceMobileUI", targets: ["MferenceMobileUI"]),
        .library(name: "MferenceMobileSamples", targets: ["MferenceMobileSamples"]),
    ],
    dependencies: [
        .package(name: "MferenceApp", path: "../../App"),
    ],
    targets: [
        .target(
            name: "MferenceMobileCore",
            dependencies: [.product(name: "MferenceLink", package: "MferenceApp")],
            swiftSettings: settings),
        .target(
            name: "MferenceMobileUI",
            dependencies: [
                "MferenceMobileCore",
                .product(name: "MferenceDesign", package: "MferenceApp"),
            ],
            swiftSettings: settings),
        .target(
            name: "MferenceMobileSamples",
            dependencies: ["MferenceMobileUI"],
            swiftSettings: settings),
        .testTarget(
            name: "MferenceMobileCoreTests",
            dependencies: ["MferenceMobileCore"],
            swiftSettings: settings),
        .testTarget(
            name: "MferenceMobileUITests",
            dependencies: ["MferenceMobileUI", "MferenceMobileSamples"],
            swiftSettings: settings),
        .testTarget(
            name: "MferenceMobileSamplesTests",
            dependencies: ["MferenceMobileSamples"],
            swiftSettings: settings),
    ]
)

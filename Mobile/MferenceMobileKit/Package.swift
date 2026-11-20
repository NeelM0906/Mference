// swift-tools-version: 6.2
import PackageDescription

// The iPhone and iPhone Duo app's screens (design locked 2026-11-19 on the
// mobile design canvas). The app target itself is Mobile/MferenceMobile.xcodeproj,
// a thin @main over MferenceMobileUI. Design tokens come from the Mac app's
// MferenceDesign target, so both apps share one token set.
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
        .library(name: "MferenceMobileUI", targets: ["MferenceMobileUI"]),
    ],
    dependencies: [
        .package(name: "MferenceApp", path: "../../App"),
    ],
    targets: [
        .target(
            name: "MferenceMobileUI",
            dependencies: [.product(name: "MferenceDesign", package: "MferenceApp")],
            swiftSettings: settings),
        .testTarget(
            name: "MferenceMobileUITests",
            dependencies: ["MferenceMobileUI"],
            swiftSettings: settings),
    ]
)

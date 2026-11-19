// swift-tools-version: 6.2
import PackageDescription

// The Mference Mac app. A separate package so the engine package at the
// repository root stays clean; it links the engine in-process through a local
// path dependency (LlamaCpp uses unsafeFlags, which SwiftPM refuses from a
// versioned remote package).
//
// Ownership (App Program spec section 8): U1 owns this file, MferenceDesign,
// MferenceMac, Scripts/ and the shell, Settings, Model library and Onboarding
// views; U2 owns MferenceAppCore/{Engine,Models,Persistence,Permissions,Memory};
// U3 owns MferenceAppCore/{Chat,Web} and MferenceAppUI/Chat; U4 owns
// MferenceAppCore/Apple and MferenceAppUI/{Scan,Calendar,Mail}. The shared
// contracts listed in App/CONTRACTS.md change only through a PR the
// coordinator approves.

let appSwiftSettings: [SwiftSetting] = [
    .treatAllWarnings(as: .error),
]

let uiSwiftSettings: [SwiftSetting] = appSwiftSettings + [
    .defaultIsolation(MainActor.self),
]

let package = Package(
    name: "MferenceApp",
    platforms: [
        .macOS(.v26),
        // The iPhone app (Mobile/) builds MferenceDesign and MferenceLink, and
        // MferenceAppCore once its Mac-only parts are guarded (mobile program
        // decision M5). The executables and MferenceAppUI stay Mac only.
        .iOS(.v26),
    ],
    products: [
        .executable(name: "MferenceMac", targets: ["MferenceMac"]),
        .library(name: "MferenceAppCore", targets: ["MferenceAppCore"]),
        .library(name: "MferenceDesign", targets: ["MferenceDesign"]),
        .library(name: "MferenceLink", targets: ["MferenceLink"]),
        .library(name: "MferenceAppUI", targets: ["MferenceAppUI"]),
    ],
    dependencies: [
        .package(name: "Mference", path: ".."),
    ],
    targets: [
        .target(
            name: "MferenceDesign",
            swiftSettings: appSwiftSettings),
        // The iPhone to Mac link protocol (messages only, Foundation only),
        // shared by the Mac's pairing listener and the phone. A mobile contract.
        .target(
            name: "MferenceLink",
            swiftSettings: appSwiftSettings),
        // Shared by the Mac and the iPhone (mobile decision M5). The engine is
        // linked on macOS only until engine ask E1 lets MferenceEngine build for
        // iOS; iOS code reaches it behind `#if canImport(MferenceEngine)`.
        .target(
            name: "MferenceAppCore",
            dependencies: [
                "MferenceLink",
                .product(name: "MferenceEngine", package: "Mference", condition: .when(platforms: [.macOS])),
                .product(name: "Mference", package: "Mference", condition: .when(platforms: [.macOS])),
            ],
            // Versioned web-answer prompt and tool schemas (spec 4.5.3), U3.
            resources: [.process("Web/Resources")],
            swiftSettings: appSwiftSettings),
        .target(
            name: "MferenceAppUI",
            dependencies: ["MferenceAppCore", "MferenceDesign"],
            swiftSettings: uiSwiftSettings),
        // End-to-end smoke, one request per model through EngineService.live()
        // (U2). Built release and run only under the machine lock.
        .executableTarget(
            name: "MferenceSmoke",
            dependencies: ["MferenceAppCore"],
            swiftSettings: appSwiftSettings),
        .executableTarget(
            name: "MferenceMac",
            dependencies: ["MferenceAppUI", "MferenceAppCore", "MferenceDesign"],
            swiftSettings: uiSwiftSettings),
        .testTarget(
            name: "MferenceAppCoreTests",
            dependencies: [
                "MferenceAppCore",
                .product(name: "MferenceEngine", package: "Mference"),
                .product(name: "Mference", package: "Mference"),
            ],
            swiftSettings: appSwiftSettings),
        .testTarget(
            name: "MferenceLinkTests",
            dependencies: ["MferenceLink"],
            swiftSettings: appSwiftSettings),
        .testTarget(
            name: "MferenceAppUITests",
            dependencies: ["MferenceAppUI", "MferenceAppCore", "MferenceDesign", "MferenceMac"],
            swiftSettings: uiSwiftSettings),
    ]
)

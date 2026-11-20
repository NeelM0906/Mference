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
        // MferenceDesign also builds for the iPhone app (Mobile/); the other
        // targets stay Mac only.
        .iOS(.v26),
    ],
    products: [
        .executable(name: "MferenceMac", targets: ["MferenceMac"]),
        .library(name: "MferenceAppCore", targets: ["MferenceAppCore"]),
        .library(name: "MferenceDesign", targets: ["MferenceDesign"]),
        .library(name: "MferenceAppUI", targets: ["MferenceAppUI"]),
    ],
    dependencies: [
        .package(name: "Mference", path: ".."),
        // Speech for Notes (Notes spec 3, 5.4; spike S1 2026-11-20): upstream
        // FluidAudio loads Fluid Speech Pro as `.v3` from a local folder.
        // Pinned exactly. Traits: ASR, VAD and diarization only; no TTS and
        // no prebuilt NemoTextProcessing binary. Only MferenceAppCore/Speech
        // imports it, and only `loadLocal`-style APIs that never download.
        .package(url: "https://github.com/FluidInference/FluidAudio.git", exact: "0.18.4",
                 traits: ["Diarizer"]),
    ],
    targets: [
        .target(
            name: "MferenceDesign",
            swiftSettings: appSwiftSettings),
        .target(
            name: "MferenceAppCore",
            dependencies: [
                .product(name: "MferenceEngine", package: "Mference"),
                .product(name: "Mference", package: "Mference"),
                .product(name: "FluidAudio", package: "FluidAudio"),
            ],
            // Versioned web-answer prompt and tool schemas (spec 4.5.3), U3;
            // versioned meeting-summary prompts (Notes spec 6.5), N4.
            resources: [.process("Web/Resources"), .process("Meetings/Resources")],
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
            name: "MferenceAppUITests",
            dependencies: ["MferenceAppUI", "MferenceAppCore", "MferenceDesign", "MferenceMac"],
            swiftSettings: uiSwiftSettings),
    ]
)

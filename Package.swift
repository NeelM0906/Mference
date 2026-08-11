// swift-tools-version: 6.1
import PackageDescription

// macOS 15 / iOS 18 is the floor set by `Synchronization.Mutex` and SwiftUI's
// `WindowDragGesture`. Building against a newer SDK still lights up the MSL 4.0
// tensor-ops kernels at runtime; see `MetalSDKCompatibility.swift`.
let package = Package(
    name: "Mference",
    platforms: [
        .macOS(.v15),
        .iOS(.v18),
    ],
    products: [
        .library(name: "Mference", targets: ["Mference"]),
        .executable(name: "MferenceRepack", targets: ["MferenceRepack"]),
        .executable(name: "MferenceCLI", targets: ["MferenceCLI"]),
        .executable(name: "MferenceMac", targets: ["MferenceMac"]),
        .executable(name: "MferenceDecodeService", targets: ["MferenceDecodeService"]),
        .executable(name: "MferenceServer", targets: ["MferenceServer"]),
    ],
    dependencies: [
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.0"),
        .package(url: "https://github.com/apple/swift-nio.git", exact: "2.99.0"),
    ],
    targets: [
        .target(
            name: "Mference",
            dependencies: [
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            path: "Sources/Mference",
            resources: [
                .copy("Metal"),
            ]
        ),
        .target(
            name: "MferenceRepackCore",
            path: "Sources/MferenceRepack/Core"
        ),
        .executableTarget(
            name: "MferenceRepack",
            dependencies: ["MferenceRepackCore"],
            path: "Sources/MferenceRepack/Command"
        ),
        .target(
            name: "MferenceCLICore",
            dependencies: ["Mference"],
            path: "Sources/MferenceCLI",
            exclude: ["Command"]
        ),
        .executableTarget(
            name: "MferenceCLI",
            dependencies: ["MferenceCLICore"],
            path: "Sources/MferenceCLI/Command"
        ),
        .target(
            name: "MferenceAppCore",
            dependencies: ["Mference", "MferenceRepackCore", "MferenceDecodeProtocol"],
            path: "Sources/MferenceApp/Core",
            resources: [
                .copy("Resources/app-prompts.json"),
            ]
        ),
        .target(
            name: "MferenceMacPresentation",
            dependencies: ["MferenceAppCore"],
            path: "Sources/MferenceApp/MacPresentation"
        ),
        .target(
            name: "MferenceDecodeProtocol",
            path: "Sources/MferenceDecodeProtocol"
        ),
        .executableTarget(
            name: "MferenceDecodeService",
            dependencies: ["MferenceAppCore", "MferenceDecodeProtocol"],
            path: "Sources/MferenceDecodeService"
        ),
        .target(
            name: "MferenceServerCore",
            dependencies: [
                "Mference",
                .product(name: "NIOCore", package: "swift-nio"),
                .product(name: "NIOPosix", package: "swift-nio"),
                .product(name: "NIOHTTP1", package: "swift-nio"),
            ],
            path: "Sources/MferenceServer/Core"
        ),
        .executableTarget(
            name: "MferenceServer",
            dependencies: ["MferenceServerCore"],
            path: "Sources/MferenceServer/Command"
        ),
        .executableTarget(
            name: "MferenceMac",
            dependencies: ["MferenceAppCore", "MferenceMacPresentation"],
            path: "Sources/MferenceApp/Mac",
            resources: [
                .copy("Resources/mference-app-icon.png"),
            ]
        ),
        .target(
            name: "MferenceValidationSupport",
            dependencies: ["Mference"],
            path: "Sources/MferenceValidation/Support"
        ),
        .testTarget(
            name: "MferenceTestsCore",
            dependencies: ["Mference", "MferenceValidationSupport", "MferenceRepackCore", "MferenceCLICore"],
            path: "Tests/Mference/Core",
            resources: [.copy("Tokenization/Fixtures")]
        ),
        .testTarget(
            name: "MferenceRepackTests",
            dependencies: ["MferenceRepackCore"],
            path: "Tests/MferenceRepack/Core"
        ),
        .testTarget(
            name: "MferenceAppCoreTests",
            dependencies: ["MferenceAppCore", "Mference", "MferenceRepackCore", "MferenceDecodeProtocol"],
            path: "Tests/MferenceApp/Core"
        ),
        .testTarget(
            name: "MferenceMacPresentationTests",
            dependencies: ["MferenceMacPresentation"],
            path: "Tests/MferenceApp/MacPresentation"
        ),
        .testTarget(
            name: "MferenceServerTests",
            dependencies: [
                "MferenceServerCore",
                .product(name: "NIOEmbedded", package: "swift-nio"),
            ],
            path: "Tests/MferenceServer",
            resources: [.copy("Fixtures")]
        ),
    ]
)

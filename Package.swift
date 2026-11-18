// swift-tools-version: 6.1
import PackageDescription

// llama.cpp (Sources/LlamaCpp) is built with the defines its CMake build uses
// on Apple arm64 with Metal on: CPU backend with llamafile, repack and
// Accelerate; no BLAS backend; Metal kernels compiled from source at device
// init. See Sources/LlamaCpp/vendoring/VENDORING.md.
let llamaCppDefines: [(String, String?)] = [
    ("GGML_USE_CPU", nil),
    ("GGML_USE_METAL", nil),
    ("GGML_USE_ACCELERATE", nil),
    ("ACCELERATE_NEW_LAPACK", nil),
    ("ACCELERATE_LAPACK_ILP64", nil),
    ("GGML_USE_LLAMAFILE", nil),
    ("GGML_USE_CPU_REPACK", nil),
    ("GGML_SCHED_MAX_COPIES", "4"),
    ("_DARWIN_C_SOURCE", nil),
    ("_XOPEN_SOURCE", "600"),
]
let llamaCppHeaderPaths = [
    "ggml/src", "ggml/src/ggml-cpu", "ggml/src/ggml-metal", "src", "common", "vendor",
]
// ggml-metal's Objective-C uses manual retain/release, so ARC (which SwiftPM
// enables for Objective-C) is off. Upstream's warnings are not ours to fix, so
// they are silenced for this target only, as upstream's own builds do.
let llamaCppCSettings: [CSetting] =
    llamaCppDefines.map { CSetting.define($0.0, to: $0.1) }
    + llamaCppHeaderPaths.map { CSetting.headerSearchPath($0) }
    + [.define("NDEBUG", .when(configuration: .release)), .unsafeFlags(["-fno-objc-arc", "-w"])]
let llamaCppCXXSettings: [CXXSetting] =
    llamaCppDefines.map { CXXSetting.define($0.0, to: $0.1) }
    + llamaCppHeaderPaths.map { CXXSetting.headerSearchPath($0) }
    + [.define("NDEBUG", .when(configuration: .release)), .unsafeFlags(["-w"])]

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
        .library(name: "MferenceEngine", targets: ["MferenceEngine"]),
        .executable(name: "MferenceRepack", targets: ["MferenceRepack"]),
        .executable(name: "MferenceCLI", targets: ["MferenceCLI"]),
        .executable(name: "MferenceServer", targets: ["MferenceServer"]),
    ],
    dependencies: [
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.0"),
        // Preserve canonical template bytes, including whitespace next to '{'.
        .package(path: "Vendor/swift-jinja"),
        .package(url: "https://github.com/apple/swift-nio.git", exact: "2.99.0"),
    ],
    targets: [
        .target(
            name: "Mference",
            dependencies: [
                .product(name: "Hub", package: "swift-transformers"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            path: "Sources/Mference",
            resources: [
                .copy("Metal"),
                .copy("Tokenization/Resources/Gemma4"),
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
            name: "MferenceServerCore",
            dependencies: [
                "Mference",
                "MferenceEngine",
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
        .target(
            name: "MferenceEngine",
            dependencies: ["Mference"],
            path: "Sources/MferenceEngine"
        ),
        .target(
            name: "LlamaCpp",
            path: "Sources/LlamaCpp",
            exclude: ["vendoring"],
            resources: [
                .copy("ggml/src/ggml-metal/kernels"),
                .copy("ggml/src/ggml-metal/ggml-metal-impl.h"),
                .copy("ggml/src/ggml-common.h"),
            ],
            cSettings: llamaCppCSettings,
            cxxSettings: llamaCppCXXSettings,
            linkerSettings: [
                .linkedFramework("Accelerate"),
                .linkedFramework("Foundation"),
                .linkedFramework("Metal"),
            ]
        ),
        .target(
            name: "MferenceValidationSupport",
            dependencies: ["Mference"],
            path: "Sources/MferenceValidation/Support"
        ),
        .testTarget(
            name: "MferenceTestsCore",
            dependencies: ["Mference", "MferenceValidationSupport", "MferenceRepackCore", "MferenceCLICore",
                           .product(name: "Jinja", package: "swift-jinja")],
            path: "Tests/Mference/Core",
            resources: [.copy("Tokenization/Fixtures")]
        ),
        .testTarget(
            name: "JinjaCompatibilityTests",
            dependencies: [.product(name: "Jinja", package: "swift-jinja")],
            path: "Vendor/swift-jinja/Tests/JinjaTests"
        ),
        .testTarget(
            name: "MferenceRepackTests",
            dependencies: ["MferenceRepackCore"],
            path: "Tests/MferenceRepack/Core"
        ),
        .testTarget(
            name: "LlamaCppTests",
            dependencies: ["LlamaCpp"],
            path: "Tests/LlamaCpp"
        ),
        .testTarget(
            name: "MferenceServerTests",
            dependencies: [
                "MferenceServerCore",
                "MferenceEngine",
                .product(name: "NIOEmbedded", package: "swift-nio"),
            ],
            path: "Tests/MferenceServer",
            resources: [.copy("Fixtures")]
        ),
        .testTarget(
            name: "MferenceEngineTests",
            dependencies: ["MferenceEngine", "Mference"],
            path: "Tests/MferenceEngine"
        ),
    ],
    cLanguageStandard: .gnu11,
    cxxLanguageStandard: .gnucxx17
)

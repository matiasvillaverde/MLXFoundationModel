// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "MLXFoundationModel",
    defaultLocalization: "en",
    platforms: [
        .macOS(.v15),
        .iOS(.v18),
        .visionOS(.v2)
    ],
    products: [
        .library(
            name: "MLXFoundationModel",
            targets: ["MLXFoundationModel"]
        ),
        .library(
            name: "MLXLocalModels",
            targets: ["MLXLocalModels"]
        ),
        .library(
            name: "MLXFoundationModelExamples",
            targets: ["MLXFoundationModelExamples"]
        ),
        .executable(
            name: "FoundationModelsPlayground",
            targets: ["FoundationModelsPlayground"]
        )
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", .upToNextMinor(from: "0.31.4")),
        .package(url: "https://github.com/mlc-ai/xgrammar", .upToNextMinor(from: "0.2.2")),
        .package(
            url: "https://github.com/huggingface/swift-transformers",
            .upToNextMajor(from: "1.3.3")
        )
    ],
    targets: [
        .target(
            name: "MLXLocalModels",
            dependencies: [
                "CXGrammarBridge",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "MLXLinalg", package: "mlx-swift"),
                .product(name: "Transformers", package: "swift-transformers")
            ]
        ),
        .target(
            name: "CXGrammarBridge",
            dependencies: [
                .product(name: "XGrammar", package: "xgrammar")
            ],
            publicHeadersPath: "include",
            cxxSettings: [
                .headerSearchPath("private_include"),
                .define("XGRAMMAR_ENABLE_LOG_DEBUG", to: "0"),
                .define("XGRAMMAR_ENABLE_CPPTRACE", to: "0")
            ]
        ),
        .target(
            name: "MLXFoundationModel",
            dependencies: ["MLXLocalModels"]
        ),
        .target(
            name: "MLXFoundationModelExamples",
            dependencies: [
                "MLXFoundationModel",
                "MLXLocalModels"
            ]
        ),
        .executableTarget(
            name: "FoundationModelsPlayground",
            dependencies: [
                "MLXFoundationModel",
                "MLXFoundationModelExamples",
                "MLXLocalModels"
            ],
            path: "Examples/FoundationModelsPlayground"
        ),
        .testTarget(
            name: "MLXFoundationModelTests",
            dependencies: [
                "MLXFoundationModel",
                "MLXLocalModels"
            ]
        ),
        .testTarget(
            name: "MLXRealModelTests",
            dependencies: [
                "MLXFoundationModel",
                "MLXFoundationModelExamples",
                "MLXLocalModels"
            ],
            resources: [
                .copy("Resources")
            ]
        )
    ],
    cxxLanguageStandard: .cxx17
)

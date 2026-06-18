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
        )
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", .upToNextMinor(from: "0.31.4")),
        .package(
            url: "https://github.com/huggingface/swift-transformers",
            .upToNextMajor(from: "1.3.3")
        )
    ],
    targets: [
        .target(
            name: "MLXLocalModels",
            dependencies: [
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
            name: "MLXFoundationModel",
            dependencies: ["MLXLocalModels"]
        ),
        .testTarget(
            name: "MLXFoundationModelTests",
            dependencies: ["MLXFoundationModel"]
        )
    ]
)

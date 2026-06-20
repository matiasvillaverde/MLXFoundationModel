import Foundation
import MLX
@testable import MLXLocalModels
import Testing

@Suite("MLX packed scale sanitizer")
struct MLXPackedScaleSanitizerTests {
    @Test(
        "MXFP8 dot-scale pair uses MLX native dequantization",
        .disabled(
            if: ProcessInfo.processInfo.environment["MLX_RUN_ARRAY_UNIT_TESTS"] != "1",
            "Set MLX_RUN_ARRAY_UNIT_TESTS=1 on a launcher with MLX Metal resources available"
        )
    )
    func mxfp8DotScalePairUsesMLXNativeDequantization() throws {
        try Device.withDefaultDevice(.cpu) {
            let source = MLXArray((0 ..< 32).map { Float($0) / 16 }).reshaped([1, 32])
            let quantizedWeight = quantized(source, groupSize: 32, mode: .mxfp8)
            let result = MLXQuantizedWeightSanitizer.sanitizePackedScalePairs(
                [
                    "layer.weight": quantizedWeight.wq.view(dtype: .uint8),
                    "layer.scale": quantizedWeight.scales
                ],
                metadata: [
                    Self.dtypeKey("layer.weight"): "F8_E4M3",
                    Self.dtypeKey("layer.scale"): "F8_E8M0"
                ]
            )
            let actual = try #require(result.weights["layer.weight"])
            let expected = dequantized(
                quantizedWeight.wq,
                scales: quantizedWeight.scales,
                biases: nil,
                groupSize: 32,
                bits: 8,
                mode: .mxfp8,
                dtype: .bfloat16
            )

            eval(actual, expected)
            #expect(actual.shape == expected.shape)
            #expect(actual.asArray(Float.self) == expected.asArray(Float.self))
            #expect(result.report.packedScaleCount == 1)
            #expect(result.report.dequantizedCount == 1)
            #expect(!result.weights.keys.contains("layer.scale"))
        }
    }

    @Test(
        "MXFP4 dot-scale pair uses MLX native dequantization",
        .disabled(
            if: ProcessInfo.processInfo.environment["MLX_RUN_ARRAY_UNIT_TESTS"] != "1",
            "Set MLX_RUN_ARRAY_UNIT_TESTS=1 on a launcher with MLX Metal resources available"
        )
    )
    func mxfp4DotScalePairUsesMLXNativeDequantization() throws {
        try Device.withDefaultDevice(.cpu) {
            let source = MLXArray(Self.mxfp4RepresentableValues).reshaped([1, 32])
            let quantizedWeight = quantized(source, groupSize: 32, mode: .mxfp4)
            let result = MLXQuantizedWeightSanitizer.sanitizePackedScalePairs(
                [
                    "layer.weight": quantizedWeight.wq.view(dtype: .uint8),
                    "layer.scale": quantizedWeight.scales
                ],
                metadata: [
                    Self.dtypeKey("layer.weight"): "I8",
                    Self.dtypeKey("layer.scale"): "F8_E8M0"
                ]
            )
            let actual = try #require(result.weights["layer.weight"])
            let expected = dequantized(
                quantizedWeight.wq,
                scales: quantizedWeight.scales,
                biases: nil,
                groupSize: 32,
                bits: 4,
                mode: .mxfp4,
                dtype: .bfloat16
            )

            eval(actual, expected)
            #expect(actual.shape == expected.shape)
            #expect(actual.asArray(Float.self) == expected.asArray(Float.self))
            #expect(result.report.packedScaleCount == 1)
            #expect(result.report.dequantizedCount == 1)
            #expect(!result.weights.keys.contains("layer.scale"))
        }
    }

    @Test(
        "dot-scale pairs require FP8 safetensors dtype metadata",
        .disabled(
            if: ProcessInfo.processInfo.environment["MLX_RUN_ARRAY_UNIT_TESTS"] != "1",
            "Set MLX_RUN_ARRAY_UNIT_TESTS=1 on a launcher with MLX Metal resources available"
        )
    )
    func dotScalePairsRequireFP8SafetensorsDTypeMetadata() throws {
        Device.withDefaultDevice(.cpu) {
            let source = MLXArray((0 ..< 32).map { Float($0) / 16 }).reshaped([1, 32])
            let quantizedWeight = quantized(source, groupSize: 32, mode: .mxfp8)
            let result = MLXQuantizedWeightSanitizer.sanitizePackedScalePairs(
                [
                    "layer.weight": quantizedWeight.wq.view(dtype: .uint8),
                    "layer.scale": quantizedWeight.scales
                ],
                metadata: [
                    Self.dtypeKey("layer.weight"): "F16",
                    Self.dtypeKey("layer.scale"): "F16"
                ]
            )

            #expect(result.report.packedScaleCount == 0)
            #expect(result.report.dequantizedCount == 0)
            #expect(result.weights.keys.contains("layer.weight"))
            #expect(result.weights.keys.contains("layer.scale"))
        }
    }

    private static func dtypeKey(_ tensorName: String) -> String {
        MLXSafetensorsTensorMetadata.dtypeMetadataKey(tensorName)
    }

    private static let mxfp4RepresentableValues: [Float] = [
        6, 0.5, 1, 1.5, 2, 3, 4, 6,
        -0.5, -1, -1.5, -2, -3, -4, -6, 0,
        6, 0.5, 1, 1.5, 2, 3, 4, 6,
        -0.5, -1, -1.5, -2, -3, -4, -6, 0
    ]
}

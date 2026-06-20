import Foundation
import MLX
@testable import MLXLocalModels
import Testing

@Suite("MLX quantized weight sanitizer")
struct MLXQuantizedWeightSanitizerTests {
    private struct SanitizerOutput {
        let rows: [[Float]]
        let report: MLXQuantizedWeightSanitizer.Report
        let keys: Set<String>
    }

    @Test(
        "automatic strategy applies same-shape scale inverse directly",
        .disabled(
            if: ProcessInfo.processInfo.environment["MLX_RUN_ARRAY_UNIT_TESTS"] != "1",
            "Set MLX_RUN_ARRAY_UNIT_TESTS=1 on a launcher with MLX Metal resources available"
        )
    )
    func automaticStrategyAppliesSameShapeScaleInverseDirectly() throws {
        let output = try Self.directScaleOutput()

        #expect(output.rows == [
            [2, 6],
            [12, 20]
        ])
        #expect(output.report.directScaleCount == 1)
        #expect(output.report.blockScaleCount == 0)
        #expect(output.report.droppedActivationScaleCount == 1)
        #expect(!output.keys.contains("layer.weight_scale_inv"))
        #expect(!output.keys.contains("layer.activation_scale"))
    }

    @Test(
        "block strategy applies tiled scale inverse and slices padding",
        .disabled(
            if: ProcessInfo.processInfo.environment["MLX_RUN_ARRAY_UNIT_TESTS"] != "1",
            "Set MLX_RUN_ARRAY_UNIT_TESTS=1 on a launcher with MLX Metal resources available"
        )
    )
    func blockStrategyAppliesTiledScaleInverseAndSlicesPadding() throws {
        let output = try Self.blockScaleOutput()

        #expect(output.rows == [
            [10, 10, 100],
            [10, 10, 100],
            [1_000, 1_000, 10_000]
        ])
        #expect(output.report.directScaleCount == 0)
        #expect(output.report.blockScaleCount == 1)
        #expect(output.report.dequantizedCount == 1)
        #expect(!output.keys.contains("mlp.weight_scale_inv"))
    }

    @Test(
        "automatic strategy infers partial FP8 block scale shape",
        .disabled(
            if: ProcessInfo.processInfo.environment["MLX_RUN_ARRAY_UNIT_TESTS"] != "1",
            "Set MLX_RUN_ARRAY_UNIT_TESTS=1 on a launcher with MLX Metal resources available"
        )
    )
    func automaticStrategyInfersPartialFP8BlockScaleShape() throws {
        try Device.withDefaultDevice(.cpu) {
            let result = MLXQuantizedWeightSanitizer.sanitize(
                [
                    "fp8.weight": MLXArray.ones([130, 70], type: Float32.self),
                    "fp8.weight_scale_inv": MLXArray([
                        Float(2), Float(7),
                        Float(3), Float(9),
                        Float(5), Float(11)
                    ]).reshaped([3, 2])
                ],
                strategy: .automatic(blockSize: 128),
                sidecarPolicy: .dropActivationScale
            )
            let scaled = try #require(result.weights["fp8.weight"])
            eval(scaled)
            let values = scaled.asArray(Float.self)

            #expect(result.report.blockScaleCount == 1)
            #expect(values[Self.offset(row: 0, column: 0, columnCount: 70)] == 2)
            #expect(values[Self.offset(row: 63, column: 63, columnCount: 70)] == 2)
            #expect(values[Self.offset(row: 64, column: 0, columnCount: 70)] == 3)
            #expect(values[Self.offset(row: 128, column: 0, columnCount: 70)] == 5)
            #expect(values[Self.offset(row: 0, column: 64, columnCount: 70)] == 7)
            #expect(values[Self.offset(row: 128, column: 69, columnCount: 70)] == 11)
            #expect(!Set(result.weights.keys).contains("fp8.weight_scale_inv"))
        }
    }

    @Test(
        "missing scale target is reported without leaking sidecar weights",
        .disabled(
            if: ProcessInfo.processInfo.environment["MLX_RUN_ARRAY_UNIT_TESTS"] != "1",
            "Set MLX_RUN_ARRAY_UNIT_TESTS=1 on a launcher with MLX Metal resources available"
        )
    )
    func missingScaleTargetIsReportedWithoutLeakingSidecars() {
        let output = Device.withDefaultDevice(.cpu) {
            let result = MLXQuantizedWeightSanitizer.sanitize(
                [
                    "orphan.weight_scale_inv": MLXArray([Float(1)]),
                    "other.weight": MLXArray([Float(2)])
                ],
                strategy: .automatic(blockSize: 2),
                sidecarPolicy: .dropActivationScale
            )
            return (result.report, Set(result.weights.keys))
        }

        #expect(output.0.missingWeightCount == 1)
        #expect(output.0.dequantizedCount == 0)
        #expect(!output.1.contains("orphan.weight_scale_inv"))
        #expect(output.1.contains("other.weight"))
    }

    private static func directScaleOutput() throws -> SanitizerOutput {
        try Device.withDefaultDevice(.cpu) {
            let result = MLXQuantizedWeightSanitizer.sanitize(
                directScaleWeights(),
                strategy: .automatic(blockSize: 2),
                sidecarPolicy: .dropActivationScale
            )
            return try output(from: result, key: "layer.weight", columnCount: 2)
        }
    }

    private static func blockScaleOutput() throws -> SanitizerOutput {
        try Device.withDefaultDevice(.cpu) {
            let result = MLXQuantizedWeightSanitizer.sanitize(
                blockScaleWeights(),
                strategy: .block(blockSize: 2),
                sidecarPolicy: .dropActivationScale
            )
            return try output(from: result, key: "mlp.weight", columnCount: 3)
        }
    }

    private static func output(
        from result: MLXQuantizedWeightSanitizer.Result,
        key: String,
        columnCount: Int
    ) throws -> SanitizerOutput {
        let scaled = try #require(result.weights[key])
        eval(scaled)
        return SanitizerOutput(
            rows: rows(scaled, columnCount: columnCount),
            report: result.report,
            keys: Set(result.weights.keys)
        )
    }

    private static func directScaleWeights() -> [String: MLXArray] {
        [
            "layer.weight": MLXArray([
                Float(1), Float(2),
                Float(3), Float(4)
            ]).reshaped([2, 2]),
            "layer.weight_scale_inv": MLXArray([
                Float(2), Float(3),
                Float(4), Float(5)
            ]).reshaped([2, 2]),
            "layer.activation_scale": MLXArray([Float(1)])
        ]
    }

    private static func blockScaleWeights() -> [String: MLXArray] {
        [
            "mlp.weight": MLXArray([
                Float(1), Float(1), Float(1),
                Float(1), Float(1), Float(1),
                Float(1), Float(1), Float(1)
            ]).reshaped([3, 3]),
            "mlp.weight_scale_inv": MLXArray([
                Float(10), Float(100),
                Float(1_000), Float(10_000)
            ]).reshaped([2, 2])
        ]
    }

    private static func rows(_ array: MLXArray, columnCount: Int) -> [[Float]] {
        let values = array.asArray(Float.self)
        return stride(from: 0, to: values.count, by: columnCount).map { start in
            Array(values[start ..< start + columnCount])
        }
    }

    private static func offset(row: Int, column: Int, columnCount: Int) -> Int {
        row * columnCount + column
    }
}

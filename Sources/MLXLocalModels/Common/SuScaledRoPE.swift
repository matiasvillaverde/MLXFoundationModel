import Foundation
import MLX
import MLXNN

// Port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/rope_utils.py

/// Su-scaled rotary position embedding used by LongRoPE configurations.
internal class SuScaledRoPE: Module, OffsetLayer, ArrayOffsetLayer {
    private let dimensions: Int
    private let originalMaxPositionEmbeddings: Int
    private let freqs: MLXArray
    private let scale: Float

    internal init(
        dimensions: Int,
        base: Float = 10000.0,
        maxPositionEmbeddings: Int = 131_072,
        originalMaxPositionEmbeddings: Int = 4_096,
        shortFactor _: [Float] = [1.0],
        longFactor: [Float] = [1.0],
        shortMScale _: Float? = nil,
        longMScale: Float? = nil
    ) {
        precondition(dimensions % 2 == 0, "Dimensions must be even")

        self.dimensions = dimensions
        self.originalMaxPositionEmbeddings = originalMaxPositionEmbeddings

        let baseFrequencies: MLXArray = pow(
            base,
            arange(0, dimensions, step: 2, dtype: .float32) / dimensions
        )
        self.freqs = MLXArray(longFactor) * baseFrequencies

        func defaultScale(_ factor: Float) -> Float {
            sqrt(1 + log(factor) / log(Float(originalMaxPositionEmbeddings)))
        }

        let factor: Float = Float(maxPositionEmbeddings) / Float(originalMaxPositionEmbeddings)
        self.scale = longMScale ?? (factor < 1 ? 1 : defaultScale(factor))
    }

    internal func callAsFunction(_ x: MLXArray, offset: Int = 0) -> MLXArray {
        let x: MLXArray = scaledInput(x)
        return MLXFast.RoPE(
            x,
            dimensions: dimensions,
            traditional: false,
            base: nil,
            scale: 1.0,
            offset: offset,
            freqs: freqs
        )
    }

    internal func callAsFunction(_ x: MLXArray, offset: MLXArray) -> MLXArray {
        let x: MLXArray = scaledInput(x)
        return MLXFast.RoPE(
            x,
            dimensions: dimensions,
            traditional: false,
            base: nil,
            scale: 1.0,
            offset: offset,
            freqs: freqs
        )
    }

    private func scaledInput(_ input: MLXArray) -> MLXArray {
        let x: MLXArray = input[0..., .ellipsis]
        x[.ellipsis, 0 ..< dimensions] *= scale
        return x
    }
}

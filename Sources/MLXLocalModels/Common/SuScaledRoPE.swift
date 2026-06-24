import Foundation
import MLX
import MLXNN

internal struct SuScaledRoPEPlan: Equatable, Sendable {
    internal let dimensions: Int
    internal let base: Float
    internal let maxPositionEmbeddings: Int
    internal let originalMaxPositionEmbeddings: Int
    internal let shortFactor: [Float]
    internal let longFactor: [Float]
    internal let shortScale: Float
    internal let longScale: Float

    internal init(
        dimensions: Int,
        base: Float = 10_000,
        maxPositionEmbeddings: Int = 131_072,
        originalMaxPositionEmbeddings: Int = 4_096,
        shortFactor: [Float] = [1],
        longFactor: [Float] = [1],
        shortMScale: Float? = nil,
        longMScale: Float? = nil
    ) {
        precondition(dimensions > 0 && dimensions % 2 == 0, "Dimensions must be positive and even")
        precondition(maxPositionEmbeddings > 0, "max_position_embeddings must be positive")
        precondition(
            originalMaxPositionEmbeddings > 0,
            "original_max_position_embeddings must be positive"
        )

        self.dimensions = dimensions
        self.base = base
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.originalMaxPositionEmbeddings = originalMaxPositionEmbeddings
        self.shortFactor = Self.normalizedFactors(
            shortFactor,
            expectedCount: dimensions / 2,
            label: "short_factor"
        )
        self.longFactor = Self.normalizedFactors(
            longFactor,
            expectedCount: dimensions / 2,
            label: "long_factor"
        )
        self.shortScale = shortMScale ?? 1

        let extensionFactor =
            Float(maxPositionEmbeddings) / Float(originalMaxPositionEmbeddings)
        self.longScale =
            longMScale ?? Self.defaultLongScale(
                extensionFactor: extensionFactor,
                originalMaxPositionEmbeddings: originalMaxPositionEmbeddings
            )
    }

    internal func usesLongFrequencies(positionLimit: Int) -> Bool {
        positionLimit > originalMaxPositionEmbeddings
    }

    internal func scale(positionLimit: Int) -> Float {
        usesLongFrequencies(positionLimit: positionLimit) ? longScale : shortScale
    }

    internal func frequencyValues(useLongFrequencies: Bool) -> [Float] {
        let factors = useLongFrequencies ? longFactor : shortFactor
        return (0 ..< dimensions / 2).map { index in
            let exponent = Float(index * 2) / Float(dimensions)
            return factors[index] * pow(base, exponent)
        }
    }

    private static func normalizedFactors(
        _ factors: [Float],
        expectedCount: Int,
        label: String
    ) -> [Float] {
        precondition(!factors.isEmpty, "\(label) cannot be empty")
        if factors.count == expectedCount {
            return factors
        }
        if factors.count == 1 {
            return Array(repeating: factors[0], count: expectedCount)
        }
        preconditionFailure(
            "\(label) must contain either 1 value or \(expectedCount) values, got \(factors.count)"
        )
    }

    private static func defaultLongScale(
        extensionFactor: Float,
        originalMaxPositionEmbeddings: Int
    ) -> Float {
        guard extensionFactor > 1 else { return 1 }
        return sqrt(1 + log(extensionFactor) / log(Float(originalMaxPositionEmbeddings)))
    }
}

/// Su-scaled rotary position embedding used by LongRoPE configurations.
internal final class SuScaledRoPE: Module, OffsetLayer, ArrayOffsetLayer {
    private let plan: SuScaledRoPEPlan
    private let shortFrequencies: MLXArray
    private let longFrequencies: MLXArray

    internal init(
        dimensions: Int,
        base: Float = 10000.0,
        maxPositionEmbeddings: Int = 131_072,
        originalMaxPositionEmbeddings: Int = 4_096,
        shortFactor: [Float] = [1.0],
        longFactor: [Float] = [1.0],
        shortMScale: Float? = nil,
        longMScale: Float? = nil
    ) {
        let plan = SuScaledRoPEPlan(
            dimensions: dimensions,
            base: base,
            maxPositionEmbeddings: maxPositionEmbeddings,
            originalMaxPositionEmbeddings: originalMaxPositionEmbeddings,
            shortFactor: shortFactor,
            longFactor: longFactor,
            shortMScale: shortMScale,
            longMScale: longMScale
        )
        self.plan = plan
        self.shortFrequencies = MLXArray(
            plan.frequencyValues(useLongFrequencies: false)
        ).asType(.float32)
        self.longFrequencies = MLXArray(
            plan.frequencyValues(useLongFrequencies: true)
        ).asType(.float32)

        super.init()
    }

    internal func callAsFunction(_ x: MLXArray, offset: Int = 0) -> MLXArray {
        let positionLimit = offset + x.dim(-2)
        return MLXFast.RoPE(
            scaledInput(x, positionLimit: positionLimit),
            dimensions: plan.dimensions,
            traditional: false,
            base: nil,
            scale: 1.0,
            offset: offset,
            freqs: frequencies(positionLimit: positionLimit)
        )
    }

    internal func callAsFunction(_ x: MLXArray, offset: MLXArray) -> MLXArray {
        let positionLimit = maxBatchOffset(offset) + x.dim(-2)
        return MLXFast.RoPE(
            scaledInput(x, positionLimit: positionLimit),
            dimensions: plan.dimensions,
            traditional: false,
            base: nil,
            scale: 1.0,
            offset: offset,
            freqs: frequencies(positionLimit: positionLimit)
        )
    }

    private func frequencies(positionLimit: Int) -> MLXArray {
        plan.usesLongFrequencies(positionLimit: positionLimit)
            ? longFrequencies
            : shortFrequencies
    }

    private func scaledInput(_ input: MLXArray, positionLimit: Int) -> MLXArray {
        precondition(
            input.dim(-1) >= plan.dimensions,
            "input head dimension must contain the rotated dimensions"
        )

        let scale = plan.scale(positionLimit: positionLimit)
        guard scale != 1 else { return input }

        if input.dim(-1) == plan.dimensions {
            return input * scale
        }

        return concatenated(
            [
                input[.ellipsis, 0 ..< plan.dimensions] * scale,
                input[.ellipsis, plan.dimensions...]
            ],
            axis: -1
        )
    }

    private func maxBatchOffset(_ offset: MLXArray) -> Int {
        offset.asArray(Int.self).max() ?? 0
    }
}

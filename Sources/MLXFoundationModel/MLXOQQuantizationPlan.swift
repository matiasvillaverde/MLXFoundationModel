import Foundation

/// Budget-aware oQ tensor plan derived from model tensor metadata.
public struct MLXOQQuantizationPlan: Equatable, Sendable {
    public let baselineBitsPerWeight: Double
    public let boosts: [String: MLXOQQuantizationSpec]
    public let decisions: [String: MLXOQQuantizationDecision]
    public let effectiveBitsPerWeight: Double
    public let estimatedSerializedBytes: Int
    public let fixedOverrides: [String: MLXOQQuantizationSpec]
    public let fullPrecisionParameterCount: Int
    public let hardCapBitsPerWeight: Double?
    public let level: MLXOQLevel
    public let quantizedParameterCount: Int
    public let targetBitsPerWeight: Double?
    public let totalParameterCount: Int

    public init(
        level: MLXOQLevel,
        decisions: [String: MLXOQQuantizationDecision],
        boosts: [String: MLXOQQuantizationSpec],
        fixedOverrides: [String: MLXOQQuantizationSpec],
        baselineBitsPerWeight: Double,
        effectiveBitsPerWeight: Double,
        targetBitsPerWeight: Double?,
        hardCapBitsPerWeight: Double?,
        totalParameterCount: Int,
        quantizedParameterCount: Int,
        fullPrecisionParameterCount: Int,
        estimatedSerializedBytes: Int
    ) {
        self.baselineBitsPerWeight = baselineBitsPerWeight
        self.boosts = boosts
        self.decisions = decisions
        self.effectiveBitsPerWeight = effectiveBitsPerWeight
        self.estimatedSerializedBytes = estimatedSerializedBytes
        self.fixedOverrides = fixedOverrides
        self.fullPrecisionParameterCount = fullPrecisionParameterCount
        self.hardCapBitsPerWeight = hardCapBitsPerWeight
        self.level = level
        self.quantizedParameterCount = quantizedParameterCount
        self.targetBitsPerWeight = targetBitsPerWeight
        self.totalParameterCount = totalParameterCount
    }
}

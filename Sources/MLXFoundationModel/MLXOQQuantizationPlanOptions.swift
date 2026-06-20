import Foundation

/// Inputs that control budget-aware oQ quantization planning.
public struct MLXOQQuantizationPlanOptions: Equatable, Sendable {
    public let fixedOverrides: [String: MLXOQQuantizationSpec]
    public let hardCapBitsPerWeight: Double?
    public let layerSensitivityScores: [Int: Double]
    public let targetBitsPerWeight: Double?

    public init(
        targetBitsPerWeight: Double? = nil,
        hardCapBitsPerWeight: Double? = nil,
        layerSensitivityScores: [Int: Double] = [:],
        fixedOverrides: [String: MLXOQQuantizationSpec] = [:]
    ) {
        self.fixedOverrides = fixedOverrides
        self.hardCapBitsPerWeight = hardCapBitsPerWeight
        self.layerSensitivityScores = layerSensitivityScores
        self.targetBitsPerWeight = targetBitsPerWeight
    }

    public init(
        calibrationScores: [MLXOQLayerSensitivityScore],
        targetBitsPerWeight: Double? = nil,
        hardCapBitsPerWeight: Double? = nil,
        fixedOverrides: [String: MLXOQQuantizationSpec] = [:]
    ) {
        self.init(
            targetBitsPerWeight: targetBitsPerWeight,
            hardCapBitsPerWeight: hardCapBitsPerWeight,
            layerSensitivityScores: Self.layerSensitivityScores(from: calibrationScores),
            fixedOverrides: fixedOverrides
        )
    }

    private static func layerSensitivityScores(
        from calibrationScores: [MLXOQLayerSensitivityScore]
    ) -> [Int: Double] {
        let groupedScores = Dictionary(grouping: calibrationScores, by: \.layerIndex)
        return groupedScores.mapValues { scores in
            scores.map(\.score).max() ?? 0
        }
    }
}

import Foundation
import MLX

/// One calibration comparison for a tensor or layer output.
public struct MLXOQCalibrationObservation {
    /// Output produced by the quantized candidate path.
    public let candidateOutput: MLXArray
    /// Transformer layer index associated with this observation.
    public let layerIndex: Int
    /// Output produced by the full-precision reference path.
    public let referenceOutput: MLXArray
    /// Source tensor name used for diagnostics and layer-index inference.
    public let tensorName: String

    /// Creates a calibration observation.
    public init(
        tensorName: String,
        referenceOutput: MLXArray,
        candidateOutput: MLXArray,
        layerIndex: Int? = nil
    ) throws {
        guard let resolvedLayerIndex = layerIndex ?? MLXOQLayerIndexParser.layerIndex(in: tensorName) else {
            throw MLXOQCalibrationSensitivityError.invalidLayerIndex(tensorName)
        }
        self.candidateOutput = candidateOutput
        self.layerIndex = resolvedLayerIndex
        self.referenceOutput = referenceOutput
        self.tensorName = tensorName
    }
}

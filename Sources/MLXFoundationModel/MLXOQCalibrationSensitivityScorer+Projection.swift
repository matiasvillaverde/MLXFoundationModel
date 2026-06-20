import Foundation
import MLX

extension MLXOQCalibrationSensitivityScorer {
    /// Scores a linear projection by quantizing its weight and comparing real MLX outputs.
    public func scoreProjection(
        tensorName: String,
        inputs: MLXArray,
        weight: MLXArray,
        spec: MLXOQQuantizationSpec,
        layerIndex: Int? = nil
    ) throws -> MLXOQLayerSensitivityScore {
        try score(projectionObservation(
            tensorName: tensorName,
            inputs: inputs,
            weight: weight,
            spec: spec,
            layerIndex: layerIndex
        ))
    }

    /// Builds a calibration observation for a linear projection weight.
    public func projectionObservation(
        tensorName: String,
        inputs: MLXArray,
        weight: MLXArray,
        spec: MLXOQQuantizationSpec,
        layerIndex: Int? = nil
    ) throws -> MLXOQCalibrationObservation {
        try validateProjectionShape(tensorName: tensorName, inputs: inputs, weight: weight)
        let quantizedWeight = try quantizedProjectionWeight(weight, spec: spec)
        return try MLXOQCalibrationObservation(
            tensorName: tensorName,
            referenceOutput: inputs.matmul(weight.T),
            candidateOutput: inputs.matmul(quantizedWeight.T),
            layerIndex: layerIndex
        )
    }

    private func quantizedProjectionWeight(
        _ weight: MLXArray,
        spec: MLXOQQuantizationSpec
    ) throws -> MLXArray {
        guard let mode = QuantizationMode(rawValue: spec.mode) else {
            throw MLXOQCalibrationSensitivityError.invalidQuantizationMode(spec.mode)
        }
        let quantizedWeight = MLX.quantized(
            weight,
            groupSize: spec.groupSize,
            bits: spec.bits,
            mode: mode
        )
        return MLX.dequantized(
            quantizedWeight.wq,
            scales: quantizedWeight.scales,
            biases: quantizedWeight.biases,
            groupSize: spec.groupSize,
            bits: spec.bits,
            mode: mode,
            dtype: weight.dtype
        )
    }

    private func validateProjectionShape(
        tensorName: String,
        inputs: MLXArray,
        weight: MLXArray
    ) throws {
        guard let inputWidth = inputs.shape.last,
            let weightWidth = weight.shape.last,
            inputWidth == weightWidth else {
            throw MLXOQCalibrationSensitivityError.incompatibleProjectionShape(tensorName)
        }
    }
}

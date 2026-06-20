import Foundation

/// Errors raised while computing oQ calibration sensitivity.
public enum MLXOQCalibrationSensitivityError: Error, Equatable, Sendable {
    case incompatibleProjectionShape(String)
    case invalidLayerIndex(String)
    case invalidQuantizationMode(String)
    case nonFiniteScore(String)
}

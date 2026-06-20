import Foundation
import MLX

/// Strategy for turning two calibration outputs into a scalar sensitivity score.
public protocol MLXOQSensitivityMetric {
    /// Returns a scalar sensitivity score for `candidate` relative to `reference`.
    func score(reference: MLXArray, candidate: MLXArray) throws -> Double
}

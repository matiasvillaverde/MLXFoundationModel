import Foundation
import MLX

/// oQ sensitivity metric: MSE(reference, candidate) / mean(reference^2).
public struct MLXOQRelativeMSESensitivityMetric: MLXOQSensitivityMetric {
    public let epsilon: Double

    public init(epsilon: Double = 1e-12) {
        self.epsilon = max(epsilon, .leastNonzeroMagnitude)
    }

    public func score(reference: MLXArray, candidate: MLXArray) throws -> Double {
        let reference = reference.asType(.float32)
        let candidate = candidate.asType(.float32)
        let error = MLX.mean(MLX.square(reference - candidate))
        let magnitude = MLX.mean(MLX.square(reference))
        MLX.eval(error, magnitude)
        return Double(error.item(Float.self)) / max(Double(magnitude.item(Float.self)), epsilon)
    }
}

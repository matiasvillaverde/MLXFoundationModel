import Foundation

/// Aggregation strategy for multiple tensor observations in the same layer.
public enum MLXOQSensitivityAggregation: Codable, Equatable, Sendable {
    case maximum
    case mean

    func aggregate(_ scores: [Double]) -> Double {
        guard !scores.isEmpty else {
            return 0
        }
        switch self {
        case .maximum:
            return scores.max() ?? 0

        case .mean:
            return scores.reduce(0, +) / Double(scores.count)
        }
    }
}

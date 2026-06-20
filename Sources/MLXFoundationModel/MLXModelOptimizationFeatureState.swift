import Foundation

/// API-visible support state for one detected optimization feature.
public struct MLXModelOptimizationFeatureState: Codable, Equatable, Hashable, Sendable {
    public let feature: MLXModelOptimizationFeature
    public let status: MLXModelOptimizationFeatureStatus
    public let detail: String

    public init(
        feature: MLXModelOptimizationFeature,
        status: MLXModelOptimizationFeatureStatus,
        detail: String
    ) {
        self.feature = feature
        self.status = status
        self.detail = detail
    }
}

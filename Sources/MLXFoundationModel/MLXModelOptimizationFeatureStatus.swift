import Foundation

/// Runtime support level for an optimization feature detected in a model profile.
public enum MLXModelOptimizationFeatureStatus: String, Codable, CaseIterable, Hashable, Sendable {
    case failClosed = "fail_closed"
    case implemented = "implemented"
    case pendingRuntime = "pending_runtime"
    case scalarFallback = "scalar_fallback"
}

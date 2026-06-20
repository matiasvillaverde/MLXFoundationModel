internal struct MLXGenerationExecutionPlan: Sendable, Equatable, Hashable {
    internal let capabilities: MLXGenerationRuntimeCapabilities
    internal let effectiveScheduling: MLXGenerationSchedulingConfiguration
    internal let reason: MLXGenerationExecutionPlanReason
    internal let requestedScheduling: MLXGenerationSchedulingConfiguration
    internal let requestedStrategy: MLXGenerationExecutionStrategy
    internal let selectedStrategy: MLXGenerationExecutionStrategy

    internal var downgradedToScalar: Bool {
        requestedStrategy == .continuousBatching && selectedStrategy == .scalar
    }
}

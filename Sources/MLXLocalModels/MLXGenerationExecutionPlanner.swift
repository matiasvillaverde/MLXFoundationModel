internal enum MLXGenerationExecutionPlanner {
    internal static func plan(
        preferences: ModelRuntimePreferences,
        capabilities: MLXGenerationRuntimeCapabilities
    ) throws -> MLXGenerationExecutionPlan {
        try preferences.optimization.validate(with: preferences.speculativeDecodingMode)

        let requestedScheduling = preferences.scheduling
        let effectiveScheduling = effectiveScheduling(
            requested: requestedScheduling,
            speculativeDecodingMode: preferences.speculativeDecodingMode,
            optimization: preferences.optimization,
            capabilities: capabilities
        )
        let requestedStrategy = strategy(for: requestedScheduling)
        let selectedStrategy = strategy(for: effectiveScheduling)
        let reason = reason(
            requested: requestedStrategy,
            selected: selectedStrategy,
            speculativeDecodingMode: preferences.speculativeDecodingMode,
            optimization: preferences.optimization,
            capabilities: capabilities
        )

        if selectedStrategy == .continuousBatching {
            try effectiveScheduling.validate(for: capabilities)
        }

        return MLXGenerationExecutionPlan(
            capabilities: capabilities,
            effectiveScheduling: effectiveScheduling,
            reason: reason,
            requestedScheduling: requestedScheduling,
            requestedStrategy: requestedStrategy,
            selectedStrategy: selectedStrategy
        )
    }

    private static func strategy(
        for scheduling: MLXGenerationSchedulingConfiguration
    ) -> MLXGenerationExecutionStrategy {
        switch scheduling.mode {
        case .continuousBatching:
            .continuousBatching

        case .serial:
            .scalar
        }
    }

    private static func effectiveScheduling(
        requested: MLXGenerationSchedulingConfiguration,
        speculativeDecodingMode: SpeculativeDecodingMode,
        optimization: MLXRuntimeOptimizationConfiguration,
        capabilities: MLXGenerationRuntimeCapabilities
    ) -> MLXGenerationSchedulingConfiguration {
        if speculativeDecodingMode != .off || optimization.requiresExclusiveSpeculativePath {
            return requested.scalarGenerationConfiguration
        }
        return requested.effectiveConfiguration(for: capabilities)
    }

    private static func reason(
        requested: MLXGenerationExecutionStrategy,
        selected: MLXGenerationExecutionStrategy,
        speculativeDecodingMode: SpeculativeDecodingMode,
        optimization: MLXRuntimeOptimizationConfiguration,
        capabilities: MLXGenerationRuntimeCapabilities
    ) -> MLXGenerationExecutionPlanReason {
        if requested == .scalar {
            return .scalarRequested
        }
        if speculativeDecodingMode != .off {
            return .speculativeDecodingRequiresScalar
        }
        if optimization.mode == .externalDraft {
            return .speculativeDecodingRequiresScalar
        }
        if optimization.mode == .nativeMTP {
            return .nativeMTPRequiresScalar
        }
        if optimization.mode == .vlmMTP {
            return .sharedKVMTPRequiresScalar
        }
        if optimization.mode == .specPrefill {
            return .specPrefillRequiresScalar
        }
        if selected == .continuousBatching,
            capabilities.supportsContinuousBatching {
            return .continuousBatchingSelected
        }
        return .continuousBatchingUnsupported
    }
}

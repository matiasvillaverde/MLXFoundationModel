import Foundation

extension MLXModelOptimizationProfile {
    /// Runtime support state for every optimization feature detected in the profile.
    public var featureStates: [MLXModelOptimizationFeatureState] {
        detectedFeatures
            .sorted { left, right in left.rawValue < right.rawValue }
            .map(featureState)
    }

    /// Runtime support status for a detected optimization feature.
    public func status(
        for feature: MLXModelOptimizationFeature
    ) -> MLXModelOptimizationFeatureStatus? {
        guard detectedFeatures.contains(feature) else {
            return nil
        }
        return featureState(feature).status
    }

    /// Runtime support state for a detected optimization feature.
    public func featureState(
        for feature: MLXModelOptimizationFeature
    ) -> MLXModelOptimizationFeatureState? {
        guard detectedFeatures.contains(feature) else {
            return nil
        }
        return featureState(feature)
    }

    private func featureState(
        _ feature: MLXModelOptimizationFeature
    ) -> MLXModelOptimizationFeatureState {
        if feature == .nativeMTP {
            return nativeMTPState(feature)
        }
        if feature == .speculativePrefill {
            return scalarFallbackState(feature)
        }
        if Self.implementedStateFeatures.contains(feature) {
            return implementedState(feature)
        }
        if Self.failClosedStateFeatures.contains(feature) {
            return failClosedState(feature)
        }
        return pendingRuntimeState(feature)
    }

    private static let implementedStateFeatures: Set<MLXModelOptimizationFeature> = [
        .fp8ScaleDequantization,
        .indexCache,
        .prefillStepPromptCacheReuse,
        .turboQuantKV
    ]

    private static let failClosedStateFeatures: Set<MLXModelOptimizationFeature> = [
        .dFlash,
        .vlmMTP
    ]

    private func implementedState(
        _ feature: MLXModelOptimizationFeature
    ) -> MLXModelOptimizationFeatureState {
        .init(
            feature: feature,
            status: .implemented,
            detail: "\(feature.rawValue) is implemented in the Swift runtime."
        )
    }

    private func nativeMTPState(
        _ feature: MLXModelOptimizationFeature
    ) -> MLXModelOptimizationFeatureState {
        nativeMTPRuntimeSupported ? .init(
            feature: feature,
            status: .implemented,
            detail: "Native MTP is implemented for this model family."
        ) : .init(
            feature: feature,
            status: .pendingRuntime,
            detail: "Native MTP weights were detected, but this family has no runtime path yet."
        )
    }

    private func scalarFallbackState(
        _ feature: MLXModelOptimizationFeature
    ) -> MLXModelOptimizationFeatureState {
        .init(
            feature: feature,
            status: .scalarFallback,
            detail: "SpecPrefill requests run through the scalar dense fallback."
        )
    }

    private func failClosedState(
        _ feature: MLXModelOptimizationFeature
    ) -> MLXModelOptimizationFeatureState {
        .init(
            feature: feature,
            status: .failClosed,
            detail: "\(feature.rawValue) is detected, but explicit runtime use is rejected."
        )
    }

    private func pendingRuntimeState(
        _ feature: MLXModelOptimizationFeature
    ) -> MLXModelOptimizationFeatureState {
        .init(
            feature: feature,
            status: .pendingRuntime,
            detail: "\(feature.rawValue) is detected but not executable by the runtime yet."
        )
    }
}

@testable import MLXFoundationModel
import Testing

@Suite("MLX optimization feature status")
struct MLXModelOptimizationFeatureStatusTests {
    @Test("reports implemented fallback fail-closed and pending feature states")
    func reportsFeatureRuntimeStates() {
        let optimization = MLXModelOptimizationProfile(
            isOQQuantized: true,
            oQLevel: "oQ3.5",
            requiresFP8ScaleDequantization: true,
            hasNativeMTPWeights: true,
            supportsNativeMTP: true,
            nativeMTPRuntimeSupported: false,
            supportsVLMMTP: true,
            supportsSpeculativePrefill: true,
            supportsDFlash: true,
            supportsIndexCache: true,
            supportsTurboQuantKV: true,
            promptCacheReuseAlignment: .prefillStep
        )

        let states = Dictionary(uniqueKeysWithValues: optimization.featureStates.map { state in
            (state.feature, state.status)
        })

        #expect(states[.fp8ScaleDequantization] == .implemented)
        #expect(states[.indexCache] == .implemented)
        #expect(states[.prefillStepPromptCacheReuse] == .implemented)
        #expect(states[.turboQuantKV] == .implemented)
        #expect(states[.nativeMTP] == .pendingRuntime)
        #expect(states[.oQQuantization] == .pendingRuntime)
        #expect(states[.speculativePrefill] == .scalarFallback)
        #expect(states[.dFlash] == .failClosed)
        #expect(states[.vlmMTP] == .implemented)
    }

    @Test("marks native MTP implemented only for supported runtime families")
    func marksNativeMTPImplementedOnlyForSupportedRuntimeFamilies() {
        let unsupported = MLXModelOptimizationProfile(
            supportsNativeMTP: true,
            nativeMTPRuntimeSupported: false
        )
        let supported = MLXModelOptimizationProfile(
            supportsNativeMTP: true,
            nativeMTPRuntimeSupported: true
        )

        #expect(unsupported.status(for: .nativeMTP) == .pendingRuntime)
        #expect(supported.status(for: .nativeMTP) == .implemented)
        #expect(supported.featureState(for: .nativeMTP)?.detail.isEmpty == false)
        #expect(supported.status(for: .oQQuantization) == nil)
    }
}

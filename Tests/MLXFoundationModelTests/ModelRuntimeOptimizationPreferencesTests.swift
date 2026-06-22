import Foundation
@testable import MLXLocalModels
import Testing

@Suite("Model runtime optimization preferences")
struct ModelRuntimeOptimizationPreferencesTests {
    @Test("rejects unimplemented DFlash runtime path")
    func rejectsUnimplementedDFlashRuntimePath() throws {
        let optimization = MLXRuntimeOptimizationConfiguration.dFlash(
            draftModelID: "qwen3.5-dflash-draft"
        )
        let preferences = ModelRuntimePreferences(optimization: optimization)

        do {
            try preferences.validate()
            Issue.record("Expected \(optimization.mode.rawValue) to fail closed")
        } catch LLMError.invalidConfiguration(let message) {
            #expect(message.contains("not implemented"))
            #expect(message.contains(optimization.mode.rawValue))
        }
    }

    @Test("accepts VLM MTP as a shared-KV draft runtime path")
    func acceptsVLMMTPAsSharedKVDraftRuntimePath() throws {
        let preferences = ModelRuntimePreferences(
            optimization: .vlmMTP(draftModelID: "gemma-4-assistant")
        )

        try preferences.validate()

        #expect(preferences.optimization.requiresExclusiveSpeculativePath)
    }

    @Test("accepts SpecPrefill as a scalar dense fallback runtime path")
    func acceptsSpecPrefillAsScalarDenseFallbackRuntimePath() throws {
        let preferences = ModelRuntimePreferences(
            optimization: .specPrefill(draftModelID: "qwen3.5-specprefill-draft")
        )

        try preferences.validate()

        #expect(preferences.optimization.requiresExclusiveSpeculativePath)
    }

    @Test("accepts native MTP as a scalar exclusive runtime path")
    func acceptsNativeMTPAsScalarExclusiveRuntimePath() throws {
        let preferences = ModelRuntimePreferences(optimization: .nativeMTP())

        try preferences.validate()

        #expect(preferences.optimization.requiresExclusiveSpeculativePath)
    }

    @Test("applies default IndexCache frequency without overriding explicit settings")
    func appliesDefaultIndexCacheFrequencyWithoutOverridingExplicitSettings() {
        let implicit = MLXRuntimeOptimizationConfiguration.off
            .applyingDefaultIndexCacheFrequency(2)
        let explicit = MLXRuntimeOptimizationConfiguration.indexCache(frequency: 4)
            .applyingDefaultIndexCacheFrequency(2)

        #expect(implicit.indexCacheFrequency == 2)
        #expect(implicit.mode == .off)
        #expect(explicit.indexCacheFrequency == 4)
    }

    @Test("TurboQuant runtime preferences configure generated KV parameters")
    func turboQuantRuntimePreferencesConfigureGeneratedKVParameters() async throws {
        let session = MLXSession()
        let preferences = ModelRuntimePreferences(
            optimization: .turboQuantKV(bits: 2.5, skipLastLayer: true)
        )

        let parameters = await session.createGenerateParameters(
            from: .deterministic,
            limits: ResourceLimits(maxTokens: 4),
            runtimePreferences: preferences
        )
        let recorded = try await MLXGenerationDiagnostics.withRecording {
            MLXGenerationDiagnostics.recordParameters(parameters)
        }
        let snapshot = try #require(Self.parameterSnapshots(from: recorded.events).last)

        #expect(parameters.kvBits == 3)
        #expect(parameters.quantizedKVSkipLastLayer)
        #expect(snapshot.kvBits == 3)
        #expect(snapshot.quantizedKVSkipLastLayer)
    }

    @Test("explicit KV limits override TurboQuant runtime preferences")
    func explicitKVLimitsOverrideTurboQuantRuntimePreferences() async {
        let session = MLXSession()
        let preferences = ModelRuntimePreferences(
            optimization: .turboQuantKV(bits: 2.5, skipLastLayer: true)
        )

        let parameters = await session.createGenerateParameters(
            from: .deterministic,
            limits: ResourceLimits(maxTokens: 4, kvCacheBits: 4),
            runtimePreferences: preferences
        )

        #expect(parameters.kvBits == 4)
        #expect(!parameters.quantizedKVSkipLastLayer)
    }

    private static func parameterSnapshots(
        from events: [MLXGenerationDiagnosticEvent]
    ) -> [MLXGenerationParameterSnapshot] {
        events.compactMap { event in
            guard case .parameters(let snapshot) = event else {
                return nil
            }
            return snapshot
        }
    }
}

import Foundation
@testable import MLXLocalModels
import Testing

@Suite("Model runtime preferences")
struct ModelRuntimePreferencesTests {
    @Test("decodes older persisted settings with cache budget defaults")
    func decodesOlderPersistedSettingsWithCacheBudgetDefaults() throws {
        let data = Data("""
        {
            "residencyPreference": "warm",
            "isPinned": false,
            "idleTTLSeconds": 300,
            "promptCachePolicy": "persistent",
            "promptCacheByteLimit": 134217728,
            "speculativeDecodingMode": "off",
            "speculativeDraftTokens": 2
        }
        """.utf8)

        let preferences = try JSONDecoder().decode(ModelRuntimePreferences.self, from: data)

        #expect(preferences.persistentPromptCacheTotalByteLimit == 1_073_741_824)
        #expect(preferences.persistentPromptCacheHotByteLimit == 67_108_864)
        #expect(preferences.memoryGuard == .balanced)
        #expect(preferences.scheduling == .serial)
        #expect(preferences.scheduling.mode == .serial)
        #expect(preferences.optimization == .off)
    }

    @Test("encodes scheduling preferences")
    func encodesSchedulingPreferences() throws {
        let preferences = ModelRuntimePreferences(
            scheduling: .init(maxConcurrentRequests: 2, maxQueuedRequests: 7, maxBatchSize: 2),
            optimization: .turboQuantKV(bits: 2.5, skipLastLayer: false)
        )

        let data = try JSONEncoder().encode(preferences)
        let decoded = try JSONDecoder().decode(ModelRuntimePreferences.self, from: data)

        #expect(decoded.scheduling.maxConcurrentRequests == 2)
        #expect(decoded.scheduling.maxQueuedRequests == 7)
        #expect(decoded.scheduling.maxBatchSize == 2)
        #expect(decoded.scheduling.mode == .serial)
        #expect(decoded.optimization.mode == .turboQuantKV)
        #expect(decoded.optimization.turboQuantKVBits == 2.5)
        #expect(!decoded.optimization.turboQuantSkipLastLayer)
    }

    @Test("decodes older scheduling preferences with batch defaults")
    func decodesOlderSchedulingPreferencesWithBatchDefaults() throws {
        let data = Data("""
        {"maxConcurrentRequests":3,"maxQueuedRequests":8}
        """.utf8)

        let configuration = try JSONDecoder().decode(
            MLXGenerationSchedulingConfiguration.self,
            from: data
        )

        #expect(configuration.maxConcurrentRequests == 3)
        #expect(configuration.maxQueuedRequests == 8)
        #expect(configuration.maxBatchSize == 3)
        #expect(configuration.mode == .serial)
        #expect(configuration.scalarGenerationConfiguration.maxConcurrentRequests == 1)
        #expect(configuration.scalarGenerationConfiguration.maxQueuedRequests == 8)
        #expect(configuration.scalarGenerationConfiguration.maxBatchSize == 1)
    }

    @Test("rejects continuous batching until the batched engine is active")
    func rejectsContinuousBatchingUntilTheBatchedEngineIsActive() throws {
        let preferences = ModelRuntimePreferences(
            scheduling: .init(
                mode: .continuousBatching,
                maxConcurrentRequests: 4,
                maxQueuedRequests: 8,
                maxBatchSize: 4
            )
        )

        #expect(throws: LLMError.self) {
            try preferences.validate()
        }
    }

    @Test("accepts continuous batching when runtime capabilities support it")
    func acceptsContinuousBatchingWhenRuntimeCapabilitiesSupportIt() throws {
        let preferences = ModelRuntimePreferences(
            scheduling: .init(
                mode: .continuousBatching,
                maxConcurrentRequests: 4,
                maxQueuedRequests: 8,
                maxBatchSize: 3
            )
        )

        try preferences.validate(for: .continuousBatching)

        let effectiveBatching = preferences.scheduling.effectiveConfiguration(for: .continuousBatching)
        let effectiveScalar = preferences.scheduling.effectiveConfiguration(for: .scalar)

        #expect(effectiveBatching.mode == .continuousBatching)
        #expect(effectiveBatching.maxBatchSize == 3)
        #expect(effectiveScalar.mode == .serial)
        #expect(effectiveScalar.maxConcurrentRequests == 1)
        #expect(effectiveScalar.maxQueuedRequests == 8)
        #expect(effectiveScalar.maxBatchSize == 1)
    }

    @Test("execution planner selects continuous batching when runtime supports it")
    func executionPlannerSelectsContinuousBatchingWhenRuntimeSupportsIt() throws {
        let preferences = ModelRuntimePreferences(
            scheduling: .init(
                mode: .continuousBatching,
                maxConcurrentRequests: 4,
                maxQueuedRequests: 8,
                maxBatchSize: 3
            )
        )

        let plan = try MLXGenerationExecutionPlanner.plan(
            preferences: preferences,
            capabilities: .continuousBatching
        )

        #expect(plan.requestedStrategy == .continuousBatching)
        #expect(plan.selectedStrategy == .continuousBatching)
        #expect(plan.reason == .continuousBatchingSelected)
        #expect(plan.effectiveScheduling.mode == .continuousBatching)
        #expect(plan.effectiveScheduling.maxConcurrentRequests == 4)
        #expect(plan.effectiveScheduling.maxBatchSize == 3)
    }

    @Test("scalar MLX session uses serial admission for persisted queue settings")
    func scalarMLXSessionUsesSerialAdmissionForPersistedQueueSettings() async throws {
        let preferences = ModelRuntimePreferences(
            scheduling: .init(maxConcurrentRequests: 3, maxQueuedRequests: 9, maxBatchSize: 3)
        )
        let session = MLXSession(configuration: ProviderConfiguration(
            location: URL(fileURLWithPath: "/tmp/scalar-scheduling-test-model"),
            modelName: "scalar-scheduling-test-model",
            runtime: preferences
        ))

        try await session.applyRuntimeConfiguration()

        let snapshot = await session.generationAdmission.snapshot()
        #expect(preferences.scheduling.maxConcurrentRequests == 3)
        #expect(snapshot.maxConcurrentRequests == 1)
        #expect(snapshot.maxQueuedRequests == 9)
        #expect(snapshot.maxBatchSize == 1)
    }

    @Test("scalar MLX session downgrades continuous batching preferences")
    func scalarMLXSessionDowngradesContinuousBatchingPreferences() async throws {
        let preferences = ModelRuntimePreferences(
            scheduling: .init(
                mode: .continuousBatching,
                maxConcurrentRequests: 5,
                maxQueuedRequests: 11,
                maxBatchSize: 4
            )
        )
        let session = MLXSession(configuration: ProviderConfiguration(
            location: URL(fileURLWithPath: "/tmp/scalar-continuous-scheduling-test-model"),
            modelName: "scalar-continuous-scheduling-test-model",
            runtime: preferences
        ))

        let recording = try await Self.recordRuntimeConfiguration(for: session)
        let runtimePreferences = await session.runtimePreferences

        #expect(runtimePreferences.scheduling.mode == .continuousBatching)
        Self.expectScalarDowngrade(recording, maxQueuedRequests: 11, requestedMaxBatchSize: 4)
    }

    @Test("rejects conflicting speculative optimization modes")
    func rejectsConflictingSpeculativeOptimizationModes() throws {
        let preferences = ModelRuntimePreferences(
            speculativeDecodingMode: .sameModelDraft,
            optimization: .nativeMTP()
        )

        #expect(throws: LLMError.self) {
            try preferences.validate()
        }
    }

    @Test("requires draft model identifiers for external draft modes")
    func requiresDraftModelIdentifiersForExternalDraftModes() throws {
        let preferences = ModelRuntimePreferences(
            optimization: MLXRuntimeOptimizationConfiguration(mode: .dFlash)
        )

        #expect(throws: LLMError.self) {
            try preferences.validate()
        }
    }

    @Test("accepts TurboQuant KV beside standard generation")
    func acceptsTurboQuantKVBesideStandardGeneration() throws {
        let preferences = ModelRuntimePreferences(
            optimization: .turboQuantKV(bits: 3.5)
        )

        try preferences.validate()

        #expect(preferences.optimization.kvCacheBitsForMemoryGuard == 3.5)
        #expect(!preferences.optimization.requiresExclusiveSpeculativePath)
    }

    @Test("encodes IndexCache preferences and cache signatures")
    func encodesIndexCachePreferencesAndCacheSignatures() throws {
        let preferences = ModelRuntimePreferences(optimization: .indexCache(frequency: 4))
        let data = try JSONEncoder().encode(preferences)
        let decoded = try JSONDecoder().decode(ModelRuntimePreferences.self, from: data)
        let standardSignature = PromptCacheSignature(parameters: GenerateParameters())
        let indexCacheSignature = PromptCacheSignature(
            parameters: GenerateParameters(indexCacheFrequency: 4)
        )

        #expect(decoded.optimization.mode == .off)
        #expect(decoded.optimization.indexCacheFrequency == 4)
        #expect(MLXRuntimeOptimizationConfiguration.indexCache(frequency: 1).indexCacheFrequency == nil)
        #expect(standardSignature != indexCacheSignature)
        #expect(indexCacheSignature.indexCacheFrequency == 4)
    }

    private static func executionPlanSnapshots(
        from events: [MLXGenerationDiagnosticEvent]
    ) -> [MLXGenerationExecutionPlanSnapshot] {
        events.compactMap { event in
            guard case .executionPlan(let snapshot) = event else {
                return nil
            }
            return snapshot
        }
    }

    private struct RuntimeConfigurationRecording {
        let admissionSnapshot: MLXGenerationAdmissionController.Snapshot
        let executionSnapshot: MLXGenerationExecutionPlanSnapshot
        let plan: MLXGenerationExecutionPlan
    }

    private static func recordRuntimeConfiguration(
        for session: MLXSession
    ) async throws -> RuntimeConfigurationRecording {
        let recorded = try await MLXGenerationDiagnostics.withRecording {
            try await session.applyRuntimeConfiguration()
            return (
                plan: await session.generationExecutionPlan,
                snapshot: await session.generationAdmission.snapshot()
            )
        }
        return RuntimeConfigurationRecording(
            admissionSnapshot: recorded.result.snapshot,
            executionSnapshot: try #require(Self.executionPlanSnapshots(from: recorded.events).last),
            plan: try #require(recorded.result.plan)
        )
    }

    private static func expectScalarDowngrade(
        _ recording: RuntimeConfigurationRecording,
        maxQueuedRequests: Int,
        requestedMaxBatchSize: Int
    ) {
        #expect(recording.plan.requestedStrategy == .continuousBatching)
        #expect(recording.plan.selectedStrategy == .scalar)
        #expect(recording.plan.reason == .continuousBatchingUnsupported)
        #expect(recording.admissionSnapshot.maxConcurrentRequests == 1)
        #expect(recording.admissionSnapshot.maxQueuedRequests == maxQueuedRequests)
        #expect(recording.admissionSnapshot.maxBatchSize == 1)
        #expect(recording.executionSnapshot.requestedStrategy == .continuousBatching)
        #expect(recording.executionSnapshot.selectedStrategy == .scalar)
        #expect(recording.executionSnapshot.reason == .continuousBatchingUnsupported)
        #expect(recording.executionSnapshot.requestedMaxBatchSize == requestedMaxBatchSize)
        #expect(recording.executionSnapshot.effectiveMaxBatchSize == 1)
    }
}

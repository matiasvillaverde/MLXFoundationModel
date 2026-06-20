import Foundation
@testable import MLXLocalModels
import Testing

@Suite(
    "MLX real-model speculative decoding",
    .serialized,
    .disabled(
        if: !MLXRealModelEnvironment.isEnabled,
        "Set MLX_RUN_REAL_MODEL_TESTS=1 and download models with scripts/download-test-models.sh"
    )
)
struct MLXRealModelSpeculativeDecodingTests {
    @Test("same-model speculative generation emits real tokens through scalar execution")
    func sameModelSpeculativeGenerationEmitsRealTokens() async throws {
        guard let model = try Self.selectedModel() else {
            return
        }
        let observed = try await MLXRealModelHarness.runWithDiagnostics(
            model: model,
            sampling: .deterministic,
            limits: ResourceLimits(maxTokens: 2, maxTime: .seconds(120), reusePromptCache: false),
            prompt: "Write two words about local inference.",
            runtime: Self.speculativeRuntime,
            runtimeCapabilities: .continuousBatching
        )

        MLXRealModelHarness.verifyGenerated(observed.result)
        MLXRealModelHarness.verifyGeneratedTokenDiagnostics(
            MLXRealModelHarness.generatedTokenSnapshots(from: observed.events),
            result: observed.result
        )
        try Self.verifySpeculativeScalarPlan(observed.events)
        try Self.verifySpeculativeDrafting(observed.events)
    }

    private static var speculativeRuntime: ModelRuntimePreferences {
        ModelRuntimePreferences(
            promptCachePolicy: .memory,
            speculativeDecodingMode: .sameModelDraft,
            speculativeDraftTokens: 2,
            scheduling: .init(mode: .continuousBatching, maxConcurrentRequests: 2, maxBatchSize: 2)
        )
    }

    private static func selectedModel() throws -> MLXRealModelCatalog.Model? {
        let models = try MLXRealModelCatalog.load()
        return try MLXRealModelHarness.selectedModel("llama-3.2-1b-instruct-4bit", in: models)
    }

    private static func verifySpeculativeScalarPlan(
        _ events: [MLXGenerationDiagnosticEvent]
    ) throws {
        let snapshot = try #require(Self.executionPlanSnapshots(from: events).last)

        #expect(snapshot.requestedStrategy == MLXGenerationExecutionStrategy.continuousBatching)
        #expect(snapshot.selectedStrategy == MLXGenerationExecutionStrategy.scalar)
        #expect(snapshot.reason == MLXGenerationExecutionPlanReason.speculativeDecodingRequiresScalar)
        #expect(snapshot.effectiveMaxBatchSize == 1)
    }

    private static func verifySpeculativeDrafting(
        _ events: [MLXGenerationDiagnosticEvent]
    ) throws {
        let snapshots = Self.speculativeSnapshots(from: events)

        #expect(try #require(snapshots.last).numDraftTokens == 2)
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

    private static func speculativeSnapshots(
        from events: [MLXGenerationDiagnosticEvent]
    ) -> [MLXSpeculativeDecodingSnapshot] {
        events.compactMap { event in
            guard case .speculativeDecoding(let snapshot) = event else {
                return nil
            }
            return snapshot
        }
    }
}

import Foundation
@testable import MLXLocalModels
import Testing

@Suite("Model runtime external draft planning")
struct ModelRuntimeExternalDraftPlanningTests {
    @Test("execution planner downgrades external draft decoding to scalar")
    func executionPlannerDowngradesExternalDraftDecodingToScalar() throws {
        let preferences = ModelRuntimePreferences(
            scheduling: .init(
                mode: .continuousBatching,
                maxConcurrentRequests: 4,
                maxQueuedRequests: 8,
                maxBatchSize: 3
            ),
            optimization: .externalDraft(draftModelID: "Qwen3-0.6B-4bit")
        )

        let plan = try MLXGenerationExecutionPlanner.plan(
            preferences: preferences,
            capabilities: .continuousBatching
        )

        #expect(plan.requestedStrategy == .continuousBatching)
        #expect(plan.selectedStrategy == .scalar)
        #expect(plan.reason == .speculativeDecodingRequiresScalar)
        #expect(plan.effectiveScheduling.mode == .serial)
        #expect(plan.effectiveScheduling.maxConcurrentRequests == 1)
        #expect(plan.effectiveScheduling.maxBatchSize == 1)
    }
}

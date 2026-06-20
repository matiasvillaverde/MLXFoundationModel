@testable import MLXLocalModels
import Testing

@Suite("Model runtime SpecPrefill preferences")
struct ModelRuntimeSpecPrefillPreferencesTests {
    @Test("execution planner selects scalar dense fallback for SpecPrefill")
    func executionPlannerSelectsScalarDenseFallbackForSpecPrefill() throws {
        let preferences = ModelRuntimePreferences(
            scheduling: .init(
                mode: .continuousBatching,
                maxConcurrentRequests: 4,
                maxQueuedRequests: 8,
                maxBatchSize: 3
            ),
            optimization: .specPrefill(draftModelID: "qwen3.5-specprefill-draft")
        )

        let plan = try MLXGenerationExecutionPlanner.plan(
            preferences: preferences,
            capabilities: .continuousBatching
        )

        #expect(plan.requestedStrategy == .continuousBatching)
        #expect(plan.selectedStrategy == .scalar)
        #expect(plan.reason == .specPrefillRequiresScalar)
        #expect(plan.effectiveScheduling.mode == .serial)
        #expect(plan.effectiveScheduling.maxConcurrentRequests == 1)
        #expect(plan.effectiveScheduling.maxBatchSize == 1)
    }
}

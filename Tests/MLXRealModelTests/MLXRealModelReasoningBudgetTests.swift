@testable import MLXLocalModels
import Testing

@Suite(
    "MLX real-model reasoning budget",
    .serialized,
    .disabled(
        if: !MLXRealModelEnvironment.isEnabled,
        "Set MLX_RUN_REAL_MODEL_TESTS=1 and download models with scripts/download-test-models.sh"
    )
)
struct MLXRealModelReasoningBudgetTests {
    @Test("Qwen3 reasoning budget is enforced by decoder logits")
    func qwen3ReasoningBudgetIsEnforcedByDecoderLogits() async throws {
        let models = try MLXRealModelCatalog.load()
        guard let model = try MLXRealModelHarness.selectedModel("qwen3-0.6b-4bit", in: models) else {
            return
        }
        let observed = try await MLXRealModelHarness.runWithDiagnostics(
            model: model,
            sampling: Self.oneTokenReasoningBudgetSampling,
            limits: ResourceLimits(maxTokens: 12, maxTime: .seconds(120), reusePromptCache: false),
            prompt: "Think briefly, then answer with exactly one word: blue."
        )
        let parameters = try MLXRealModelHarness.parameterSnapshot(from: observed.events)
        let tokenEvents = MLXRealModelHarness.generatedTokenSnapshots(from: observed.events)
        let reasoningEvents = MLXRealModelHarness.reasoningBudgetSnapshots(from: observed.events)

        MLXRealModelHarness.verifyGenerated(observed.result)
        MLXRealModelHarness.verifyGeneratedTokenDiagnostics(tokenEvents, result: observed.result)
        #expect(parameters.reasoningBudgetTokens == 1)
        #expect(parameters.reasoningEndTokenCount > 0)
        Self.verifyReasoningBudgetDiagnostics(reasoningEvents)
    }

    private static var oneTokenReasoningBudgetSampling: SamplingParameters {
        SamplingParameters(
            temperature: 0.0,
            topP: 1.0,
            topK: 1,
            seed: 42,
            advanced: AdvancedSamplingParameters(
                reasoningBudget: ReasoningBudgetConfiguration(maximumTokens: 1)
            )
        )
    }

    private static func verifyReasoningBudgetDiagnostics(
        _ snapshots: [MLXReasoningBudgetSnapshot]
    ) {
        let stages = snapshots.map(\.stage)
        let summary = Comment(rawValue: Self.summary(for: snapshots))

        #expect(!snapshots.isEmpty, summary)
        #expect(stages.contains(.budgetReached), summary)
        #expect(stages.contains(.maskApplied), summary)
        #expect(stages.contains(.forcingEndMarker) || stages.contains(.forcedClosed), summary)
        #expect(snapshots.contains { ($0.forcedTokenID ?? -1) >= 0 }, summary)
    }

    private static func summary(for snapshots: [MLXReasoningBudgetSnapshot]) -> String {
        snapshots
            .map { snapshot in
                [
                    "stage=\(snapshot.stage.rawValue)",
                    "count=\(snapshot.reasoningTokenCount)",
                    "forced=\(snapshot.forcedTokenID.map(String.init) ?? "nil")",
                    "message=\(snapshot.message ?? "")"
                ]
                .joined(separator: " ")
            }
            .joined(separator: "\n")
    }
}

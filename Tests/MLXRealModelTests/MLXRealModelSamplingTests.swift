@testable import MLXLocalModels
import Testing

@Suite(
    "MLX real-model sampling controls",
    .serialized,
    .disabled(
        if: !MLXRealModelEnvironment.isEnabled,
        "Set MLX_RUN_REAL_MODEL_TESTS=1 and download models with scripts/download-test-models.sh"
    )
)
struct MLXRealModelSamplingTests {
    @Test("Qwen3 generation applies sampling and logits controls")
    func qwen3GenerationAppliesSamplingAndLogitsControls() async throws {
        let models = try MLXRealModelCatalog.load()
        let model = try MLXRealModelHarness.requireModel("qwen3-0.6b-4bit", in: models)
        let observed = try await MLXRealModelHarness.runWithDiagnostics(
            model: model,
            sampling: Self.samplingControls,
            limits: ResourceLimits(maxTokens: 10, maxTime: .seconds(120), reusePromptCache: false)
        )
        let parameters = try MLXRealModelHarness.parameterSnapshot(from: observed.events)

        MLXRealModelHarness.verifyGenerated(observed.result)
        #expect(parameters.temperature == 0.7)
        #expect(parameters.topP == 0.85)
        #expect(parameters.topK == 8)
        #expect(parameters.minP == 0.01)
        #expect(parameters.repetitionPenalty == 1.05)
        #expect(parameters.repetitionContextSize == 16)
        #expect(parameters.presencePenalty == 0.10)
        #expect(parameters.frequencyPenalty == 0.10)
        #expect(parameters.seed == 1_234)
        #expect(parameters.logitBiasCount == 1)
    }

    private static var samplingControls: SamplingParameters {
        SamplingParameters(
            temperature: 0.7,
            topP: 0.85,
            topK: 8,
            repetitionPenalty: 1.05,
            frequencyPenalty: 0.10,
            presencePenalty: 0.10,
            repetitionPenaltyRange: 16,
            seed: 1_234,
            advanced: AdvancedSamplingParameters(
                minP: 0.01,
                logitBias: [0: -1.0]
            )
        )
    }
}

import MLXLocalModels
import Testing

@Suite(
    "MLX real-model generation",
    .serialized,
    .disabled(
        if: !MLXRealModelEnvironment.isEnabled,
        "Set MLX_RUN_REAL_MODEL_TESTS=1 and download models with scripts/download-test-models.sh"
    )
)
struct MLXRealModelGenerationTests {
    @Test("selected catalog models load and generate")
    func selectedCatalogModelsLoadAndGenerate() async throws {
        let models = try MLXRealModelCatalog.load()
        let selected = MLXRealModelEnvironment.selectedModels(from: models)
        let missing = selected.filter { !MLXRealModelEnvironment.hasModelFiles(for: $0) }

        #expect(!selected.isEmpty)
        #expect(
            missing.isEmpty,
            Comment(rawValue: MLXRealModelEnvironment.missingModelsMessage(missing))
        )
        guard missing.isEmpty else {
            return
        }

        for model in selected {
            let result = try await MLXRealModelHarness.run(model: model)
            MLXRealModelHarness.verifyGenerated(result, expectedTokens: model.expectedTokens)
        }
    }

    @Test("Qwen3 stops on configured stop sequence")
    func qwen3StopsOnConfiguredStopSequence() async throws {
        let models = try MLXRealModelCatalog.load()
        let model = try MLXRealModelHarness.requireModel("qwen3-0.6b-4bit", in: models)
        let result = try await MLXRealModelHarness.run(
            model: model,
            prompt: "Write exactly: alpha STOP beta",
            sampling: SamplingParameters(
                temperature: 0.0,
                topP: 1.0,
                topK: 1,
                seed: 42,
                stopSequences: ["STOP"]
            ),
            limits: ResourceLimits(
                maxTokens: 12,
                maxTime: .seconds(120),
                reusePromptCache: false
            )
        )

        MLXRealModelHarness.verifyGenerated(result)
        #expect(!result.text.contains("STOP"))
    }
}

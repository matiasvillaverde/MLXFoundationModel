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
    @Test("selected catalog models apply sampling and logits controls")
    func selectedCatalogModelsApplySamplingAndLogitsControls() async throws {
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

        var failures: [String] = []
        for model in selected {
            do {
                try await Self.verifySamplingAndLogitsControls(
                    model: model,
                    maxTokens: min(
                        model.maxTokens,
                        MLXRealModelEnvironment.architectureGenerationTokenLimit
                    )
                )
            } catch {
                failures.append("\(model.id): \(error)")
            }
        }
        #expect(failures.isEmpty, Comment(rawValue: failures.joined(separator: "\n")))
    }

    @Test("Qwen3 generation applies sampling and logits controls")
    func qwen3GenerationAppliesSamplingAndLogitsControls() async throws {
        let models = try MLXRealModelCatalog.load()
        guard let model = try MLXRealModelHarness.selectedModel("qwen3-0.6b-4bit", in: models) else {
            return
        }
        try await Self.verifySamplingAndLogitsControls(model: model, maxTokens: 10)
    }

    private static func verifySamplingAndLogitsControls(
        model: MLXRealModelCatalog.Model,
        maxTokens: Int
    ) async throws {
        let observed = try await MLXRealModelHarness.runWithDiagnostics(
            model: model,
            sampling: Self.samplingControls,
            limits: ResourceLimits(
                maxTokens: maxTokens,
                maxTime: .seconds(MLXRealModelEnvironment.architectureGenerationTimeoutSeconds),
                reusePromptCache: false
            )
        )
        let parameters = try MLXRealModelHarness.parameterSnapshot(from: observed.events)
        let tokenEvents = MLXRealModelHarness.generatedTokenSnapshots(from: observed.events)

        MLXRealModelHarness.verifyGenerated(observed.result)
        MLXRealModelHarness.verifyGeneratedTokenDiagnostics(tokenEvents, result: observed.result)
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

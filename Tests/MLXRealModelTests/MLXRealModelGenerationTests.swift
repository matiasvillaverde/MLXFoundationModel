@testable import MLXLocalModels
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

        var failures: [String] = []
        for model in selected {
            do {
                try await Self.verifyGeneration(for: model)
            } catch {
                failures.append("\(model.id): \(error)")
            }
        }
        #expect(failures.isEmpty, Comment(rawValue: failures.joined(separator: "\n")))
    }

    @Test("Qwen3 stops on configured stop sequence")
    func qwen3StopsOnConfiguredStopSequence() async throws {
        let models = try MLXRealModelCatalog.load()
        guard let model = try MLXRealModelHarness.selectedModel("qwen3-0.6b-4bit", in: models) else {
            return
        }
        let observed = try await MLXRealModelHarness.runWithDiagnostics(
            model: model,
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
            ),
            prompt: "Write exactly: alpha STOP beta"
        )
        let tokenEvents = MLXRealModelHarness.generatedTokenSnapshots(from: observed.events)

        MLXRealModelHarness.verifyGenerated(observed.result)
        MLXRealModelHarness.verifyGeneratedTokenDiagnostics(tokenEvents, result: observed.result)
        #expect(!observed.result.text.contains("STOP"))
    }

    private static func verifyGeneration(
        for model: MLXRealModelCatalog.Model
    ) async throws {
        let tokenLimit = min(model.maxTokens, MLXRealModelEnvironment.architectureGenerationTokenLimit)
        let observed = try await MLXRealModelHarness.runWithDiagnostics(
            model: model,
            sampling: .deterministic,
            limits: ResourceLimits(
                maxTokens: tokenLimit,
                maxTime: .seconds(MLXRealModelEnvironment.architectureGenerationTimeoutSeconds),
                reusePromptCache: false
            )
        )
        let tokenEvents = MLXRealModelHarness.generatedTokenSnapshots(from: observed.events)

        MLXRealModelHarness.verifyGenerated(observed.result)
        MLXRealModelHarness.verifyGeneratedTokenDiagnostics(tokenEvents, result: observed.result)
    }
}

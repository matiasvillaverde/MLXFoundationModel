import Foundation
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
        printBenchmarkSummary(model: model, result: observed.result)
    }

    private static func printBenchmarkSummary(
        model: MLXRealModelCatalog.Model,
        result: MLXRealModelHarness.GenerationResult
    ) {
        guard let metrics = result.metrics,
              let usage = metrics.usage,
              let timing = metrics.timing,
              usage.generatedTokens > 0 else {
            return
        }

        let totalSeconds = seconds(timing.totalTime)
        let promptSeconds = timing.promptProcessingTime.map(seconds) ?? 0
        let decodeSeconds = max(totalSeconds - promptSeconds, 0)
        let generated = Double(usage.generatedTokens)
        let promptTokens = usage.promptTokens ?? 0
        let decodeTPS = decodeSeconds > 0 ? generated / decodeSeconds : 0
        let endToEndTPS = totalSeconds > 0 ? generated / totalSeconds : 0

        print(
            String(
                format: """
                BENCH model=%@ architecture=%@ generated=%d prompt=%d \
                total_s=%.4f prompt_s=%.4f decode_s=%.4f decode_tps=%.2f e2e_tps=%.2f
                """,
                model.id,
                model.architecture,
                usage.generatedTokens,
                promptTokens,
                totalSeconds,
                promptSeconds,
                decodeSeconds,
                decodeTPS,
                endToEndTPS
            )
        )
    }

    private static func seconds(_ duration: Duration) -> Double {
        let components = duration.components
        return Double(components.seconds) + Double(components.attoseconds) / 1e18
    }
}

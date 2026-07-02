import Foundation
@testable import MLXLocalModels
import Testing

@Suite(
    "MLX real-model stress",
    .serialized,
    .disabled(
        if: !MLXRealModelEnvironment.isEnabled,
        "Set MLX_RUN_REAL_MODEL_TESTS=1 and download models with scripts/download-test-models.sh"
    )
)
struct MLXRealModelStressTests {
    @Test("selected models survive repeated generation")
    func selectedModelsSurviveRepeatedGeneration() async throws {
        let models = MLXRealModelEnvironment.selectedModels(from: try MLXRealModelCatalog.load())
        let missingModels = models.filter { !MLXRealModelEnvironment.hasModelFiles(for: $0) }

        #expect(!models.isEmpty)
        #expect(
            missingModels.isEmpty,
            Comment(rawValue: MLXRealModelEnvironment.missingModelsMessage(missingModels))
        )
        guard missingModels.isEmpty else {
            return
        }

        var failures: [String] = []
        for model in models {
            do {
                try await Self.verifyStressRun(for: model)
            } catch {
                failures.append("\(model.id): \(error)")
            }
        }
        #expect(failures.isEmpty, Comment(rawValue: failures.joined(separator: "\n")))
    }

    private static func verifyStressRun(
        for model: MLXRealModelCatalog.Model
    ) async throws {
        let session = MLXSessionFactory.create()
        do {
            try await MLXRealModelHarness.preload(session: session, model: model)
            for iteration in 1 ... MLXRealModelEnvironment.stressIterationCount {
                let observed = try await runWithDiagnostics(
                    session: session,
                    input: stressInput(for: model, iteration: iteration)
                )
                verifyStressOutput(observed)
                printStressSummary(model: model, iteration: iteration, result: observed.result)
            }
            await session.unload()
        } catch {
            await session.unload()
            throw error
        }
    }

    private static func stressInput(
        for model: MLXRealModelCatalog.Model,
        iteration: Int
    ) -> LLMInput {
        let prompt = """
        \(model.prompt)
        Reply in concise plain text. Include one specific technical detail.
        Stress iteration \(iteration).
        """
        return LLMInput(
            context: prompt,
            sampling: SamplingParameters(temperature: 0, topP: 1, topK: 1, seed: 1_000 + iteration),
            limits: ResourceLimits(
                maxTokens: MLXRealModelEnvironment.stressGenerationTokenLimit,
                maxTime: .seconds(MLXRealModelEnvironment.stressTimeoutSeconds),
                reusePromptCache: false
            )
        )
    }

    private static func verifyStressOutput(
        _ observed: (
            result: MLXRealModelHarness.GenerationResult,
            events: [MLXGenerationDiagnosticEvent]
        )
    ) {
        let tokenEvents = MLXRealModelHarness.generatedTokenSnapshots(from: observed.events)
        MLXRealModelHarness.verifyGenerated(observed.result)
        MLXRealModelHarness.verifyGeneratedTokenDiagnostics(tokenEvents, result: observed.result)
    }

    private static func runWithDiagnostics(
        session: any MLXGeneratingSession,
        input: LLMInput
    ) async throws -> (
        result: MLXRealModelHarness.GenerationResult,
        events: [MLXGenerationDiagnosticEvent]
    ) {
        try await MLXGenerationDiagnostics.withRecording {
            try await collectGeneration(from: await session.stream(input))
        }
    }

    private static func collectGeneration(
        from stream: AsyncThrowingStream<LLMStreamChunk, Error>
    ) async throws -> MLXRealModelHarness.GenerationResult {
        var text = ""
        var textChunkCount = 0
        var metrics: ChunkMetrics?
        var lifecycleEvents: [StreamLifecycleEvent] = []
        for try await chunk in stream {
            if case .text = chunk.event {
                text += chunk.text
                if !chunk.text.isEmpty {
                    textChunkCount += 1
                }
            }
            if case .lifecycle(let event) = chunk.event {
                lifecycleEvents.append(event)
            }
            metrics = chunk.metrics ?? metrics
        }
        return MLXRealModelHarness.GenerationResult(
            text: text,
            textChunkCount: textChunkCount,
            metrics: metrics,
            lifecycleEvents: lifecycleEvents
        )
    }

    private static func printStressSummary(
        model: MLXRealModelCatalog.Model,
        iteration: Int,
        result: MLXRealModelHarness.GenerationResult
    ) {
        guard let metrics = result.metrics,
              let summary = MLXRealModelBenchmarkSummary(metrics: metrics) else {
            return
        }

        print(summary.stressLine(model: model, iteration: iteration))
        print(summary.stressJSONLine(model: model, iteration: iteration))
    }
}

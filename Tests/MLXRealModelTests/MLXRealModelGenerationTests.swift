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

    @Test("Qwen3 runs rotating and quantized KV cache options")
    func qwen3RunsRuntimeKVCacheOptions() async throws {
        let models = try MLXRealModelCatalog.load()
        guard let model = try MLXRealModelHarness.selectedModel("qwen3-0.6b-4bit", in: models) else {
            return
        }

        try await Self.verifyRotatingKVCacheGeneration(model: model)
        try await Self.verifyQuantizedKVCacheGeneration(model: model)
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

    private static func verifyRotatingKVCacheGeneration(
        model: MLXRealModelCatalog.Model
    ) async throws {
        let maxKVSize = 16
        let observed = try await Self.runWithCacheSnapshots(
            model: model,
            limits: ResourceLimits(
                maxTokens: 8,
                maxTime: .seconds(120),
                reusePromptCache: false,
                maxKVSize: maxKVSize,
                prefillStepSize: 16
            )
        )
        let summary = Comment(rawValue: Self.cacheSummary(observed.events))

        MLXRealModelHarness.verifyGenerated(observed.result)
        #expect(Self.cacheSnapshots(from: observed.events).contains { snapshot in
            snapshot.entries.contains { entry in
                entry.maxSize == maxKVSize && entry.typeName.contains("Rotating")
            }
        }, summary)
    }

    private static func verifyQuantizedKVCacheGeneration(
        model: MLXRealModelCatalog.Model
    ) async throws {
        let observed = try await Self.runWithCacheSnapshots(
            model: model,
            limits: ResourceLimits(
                maxTokens: 8,
                maxTime: .seconds(120),
                reusePromptCache: false,
                kvCacheBits: 4,
                quantizedKVStart: 0
            )
        )
        let conversions = Self.quantizedKVConversions(from: observed.events)
        let summary = Comment(rawValue: Self.cacheSummary(observed.events))

        MLXRealModelHarness.verifyGenerated(observed.result)
        #expect(conversions.contains { conversion in
            conversion.kvBits == 4 && conversion.convertedCount > 0
        }, summary)
    }

    private static func runWithCacheSnapshots(
        model: MLXRealModelCatalog.Model,
        limits: ResourceLimits
    ) async throws -> (result: MLXRealModelHarness.GenerationResult, events: [MLXGenerationDiagnosticEvent]) {
        try await MLXGenerationDiagnostics.withRecording {
            try await MLXGenerationDiagnostics.withCacheSnapshotRecording {
                try await MLXRealModelHarness.run(
                    model: model,
                    prompt: Self.kvCachePrompt,
                    sampling: .deterministic,
                    limits: limits
                )
            }
        }
    }

    private static var kvCachePrompt: String {
        let sentence = """
        Runtime KV cache validation keeps local generation bounded, observable, and deterministic.
        """
        let body = Array(repeating: sentence, count: 8).joined(separator: " ")
        return "/no_think\n\(body)\nReply with two short words."
    }

    private static func cacheSnapshots(
        from events: [MLXGenerationDiagnosticEvent]
    ) -> [MLXCacheSnapshot] {
        events.compactMap { event in
            guard case .cacheSnapshot(let snapshot) = event else {
                return nil
            }
            return snapshot
        }
    }

    private static func quantizedKVConversions(
        from events: [MLXGenerationDiagnosticEvent]
    ) -> [MLXQuantizedKVConversionSnapshot] {
        events.compactMap { event in
            guard case .quantizedKVConversion(let snapshot) = event else {
                return nil
            }
            return snapshot
        }
    }

    private static func cacheSummary(_ events: [MLXGenerationDiagnosticEvent]) -> String {
        [
            "snapshots=\(Self.cacheSnapshots(from: events))",
            "quantizedKV=\(Self.quantizedKVConversions(from: events))"
        ].joined(separator: "\n")
    }

    private static func printBenchmarkSummary(
        model: MLXRealModelCatalog.Model,
        result: MLXRealModelHarness.GenerationResult
    ) {
        guard let metrics = result.metrics,
              let summary = MLXRealModelBenchmarkSummary(metrics: metrics) else {
            return
        }

        print(summary.benchmarkLine(model: model))
    }
}

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

    @Test("Qwen3 records redacted request summary observability")
    func qwen3RecordsRedactedRequestSummaryObservability() async throws {
        let models = try MLXRealModelCatalog.load()
        guard let model = try MLXRealModelHarness.selectedModel("qwen3-0.6b-4bit", in: models) else {
            return
        }
        let privateMarker = "private-observability-marker-1979"

        MLXObservability.reset()
        MLXObservability.configure(MLXObservabilityConfiguration(
            osLogEnabled: false,
            signpostsEnabled: false,
            minimumLogSeverity: .fault,
            keptRecentEventCount: 64,
            keptRecentRequestCount: 4
        ))
        defer { MLXObservability.reset() }

        let observed = try await MLXRealModelHarness.run(
            model: model,
            prompt: "\(privateMarker). Reply with one short word.",
            sampling: .deterministic,
            limits: ResourceLimits(
                maxTokens: 2,
                maxTime: .seconds(120),
                reusePromptCache: false
            )
        )

        MLXRealModelHarness.verifyGenerated(observed)

        let snapshot = MLXObservability.snapshot()
        let summary = try #require(snapshot.recentRequests.last)
        let event = try #require(Self.requestSummaryEvent(from: snapshot))
        let serializedEvent = Self.serializedEventPayload(event)

        #expect(summary.modelName == model.displayName)
        #expect(summary.promptTokens > 0)
        #expect(summary.generatedTokens > 0)
        #expect(summary.totalTokens == summary.promptTokens + summary.generatedTokens)
        #expect((summary.cachedPromptTokens ?? -1) >= 0)
        #expect((summary.kvCacheBytes ?? 0) > 0)
        #expect((summary.kvCacheEntries ?? 0) > 0)
        #expect(summary.promptTokensPerSecond != nil)
        #expect(summary.generationTokensPerSecond != nil)
        #expect(summary.totalTokensPerSecond != nil)
        #expect(event.attributes["model"] == model.displayName)
        #expect(event.measurements["prompt_tokens"] == Double(summary.promptTokens))
        #expect(event.measurements["generated_tokens"] == Double(summary.generatedTokens))
        #expect(event.measurements["cached_prompt_tokens"] == Double(summary.cachedPromptTokens ?? 0))
        #expect((event.measurements["kv_cache_bytes"] ?? 0) > 0)
        #expect((event.measurements["kv_cache_entries"] ?? 0) > 0)
        #expect(event.measurements["temperature"] == 0)
        #expect(event.measurements["top_p"] == 1)
        #expect(event.measurements["top_k"] == 1)
        #expect(!serializedEvent.contains(privateMarker))
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

    private static func requestSummaryEvent(
        from snapshot: MLXObservabilitySnapshot
    ) -> MLXObservabilityEvent? {
        snapshot.recentEvents.last { $0.name == "generation.request_summary" }
    }

    private static func serializedEventPayload(_ event: MLXObservabilityEvent) -> String {
        let attributeKeys = event.attributes.keys.sorted().joined(separator: " ")
        let attributeValues = event.attributes.values.sorted().joined(separator: " ")
        let measurementKeys = event.measurements.keys.sorted().joined(separator: " ")
        let measurementValues = event.measurements.values
            .map { String($0) }
            .sorted()
            .joined(separator: " ")

        let parts: [String] = [
            event.name,
            attributeKeys,
            attributeValues,
            measurementKeys,
            measurementValues
        ]
        return parts.joined(separator: " ")
    }
}

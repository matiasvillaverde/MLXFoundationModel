import Foundation
@testable import MLXLocalModels
import Testing

extension MLXRealModelGenerationTests {
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
        try Self.verifyRedactedObservabilitySnapshot(
            model: model,
            privateMarker: privateMarker
        )
    }

    private static func verifyRedactedObservabilitySnapshot(
        model: MLXRealModelCatalog.Model,
        privateMarker: String
    ) throws {
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

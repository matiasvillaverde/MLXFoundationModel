import Foundation
@testable import MLXLocalModels
import Testing

@Suite("MLX observability", .serialized)
struct MLXObservabilityTests {
    @Test("request summaries update counters, histograms, recent requests, and sinks")
    func requestSummariesUpdateSnapshotAndSink() throws {
        let sink = RecordingObservabilitySink()
        MLXObservability.reset()
        MLXObservability.configure(Self.testConfiguration, sink: sink)
        defer { MLXObservability.reset() }

        let summary = Self.makeRequestSummary()

        MLXObservability.recordRequestSummary(summary)

        let snapshot = MLXObservability.snapshot()
        #expect(snapshot.counters["generation.requests"] == 1)
        #expect(snapshot.counters["generation.tokens.prompt"] == 7)
        #expect(snapshot.counters["generation.tokens.generated"] == 5)
        #expect(snapshot.counters["prompt_cache.tokens.reused"] == 3)
        #expect(snapshot.histograms["generation.duration_seconds"]?.count == 1)
        #expect(snapshot.histograms["generation.tokens_per_second"]?.average == 3.3)
        #expect(snapshot.recentRequests == [summary])
        #expect(sink.requests() == [summary])
        #expect(sink.events().contains { $0.name == "generation.request_summary" })
    }

    @Test("diagnostic prompt cache plans map to central counters and histograms")
    func diagnosticPromptCachePlansMapToCentralMetrics() {
        MLXObservability.reset()
        MLXObservability.configure(Self.testConfiguration)
        MLXGenerationDiagnostics.resetPromptCacheObservability()
        defer {
            MLXGenerationDiagnostics.resetPromptCacheObservability()
            MLXObservability.reset()
        }

        MLXGenerationDiagnostics.recordPromptCachePlan(
            promptTokenCount: 11,
            reusedTokenCount: 5
        )

        let snapshot = MLXObservability.snapshot()
        #expect(snapshot.counters["prompt_cache.prefix.requests"] == 1)
        #expect(snapshot.counters["prompt_cache.prefix.hits"] == 1)
        #expect(snapshot.counters["prompt_cache.prefix.misses"] == nil)
        #expect(snapshot.counters["prompt_cache.tokens.requested"] == 10)
        #expect(snapshot.counters["prompt_cache.tokens.reused"] == 5)
        #expect(snapshot.histograms["prompt_cache.prefix.reused_tokens"]?.average == 5)
        #expect(snapshot.gauges["prompt_cache.prefix.hit_rate"] == 1)
    }

    @Test("generated token diagnostics are not exported to public observability")
    func generatedTokenDiagnosticsAreRedactedFromPublicObservability() async throws {
        let sink = RecordingObservabilitySink()
        let tokenID = 9_876_543
        let tokenText = "secret-model-output"
        MLXObservability.reset()
        MLXObservability.configure(Self.testConfiguration, sink: sink)
        defer { MLXObservability.reset() }

        _ = try await MLXGenerationDiagnostics.withRecording {
            MLXGenerationDiagnostics.recordGeneratedToken(
                tokenID: tokenID,
                tokenText: tokenText,
                index: 0
            )
        }

        let publicEvents = sink.events() + MLXObservability.snapshot().recentEvents
        var eventText = ""
        for event in publicEvents {
            eventText += event.name
            eventText += event.attributes.keys.joined()
            eventText += event.attributes.values.joined()
            eventText += event.measurements.keys.joined()
            for value in event.measurements.values {
                eventText += String(value)
            }
        }
        #expect(!eventText.contains(tokenText))
        #expect(!eventText.contains(String(tokenID)))
        #expect(!publicEvents.contains { $0.name.contains("generated_token") })
    }

    @Test("decode path diagnostics export structural performance metrics")
    func decodePathDiagnosticsExportStructuralPerformanceMetrics() async throws {
        let sink = RecordingObservabilitySink()
        MLXObservability.reset()
        MLXObservability.configure(Self.testConfiguration, sink: sink)
        defer { MLXObservability.reset() }

        try await Self.recordDecodePathSamples()

        let snapshot = MLXObservability.snapshot()
        let events = sink.events().filter { $0.name == "generation.decode_path.steps" }
        #expect(snapshot.counters["generation.decode_path.steps"] == 2)
        #expect(events.count == 2)
        #expect(events.contains(where: Self.isGreedyDecodePathEvent))
        #expect(events.contains(where: Self.isLogitsDecodePathEvent))
    }

    @Test("disabled observability does not update in-memory metrics")
    func disabledObservabilityDoesNotUpdateMetrics() {
        MLXObservability.reset()
        MLXObservability.configure(MLXObservabilityConfiguration(isEnabled: false))
        defer { MLXObservability.reset() }

        MLXObservability.incrementCounter("disabled.counter", category: .runtime)
        MLXObservability.setGauge("disabled.gauge", value: 1, category: .runtime)
        MLXObservability.recordHistogram("disabled.histogram", value: 1, category: .runtime)

        let snapshot = MLXObservability.snapshot()
        #expect(snapshot.counters.isEmpty)
        #expect(snapshot.gauges.isEmpty)
        #expect(snapshot.histograms.isEmpty)
        #expect(snapshot.recentEvents.isEmpty)
    }

    @Test("memory guard rejections record warning events and gauges")
    func memoryGuardRejectionsRecordWarningEventsAndGauges() {
        MLXObservability.reset()
        MLXObservability.configure(Self.testConfiguration)
        defer { MLXObservability.reset() }

        MLXGenerationDiagnostics.recordMemoryGuard(MLXMemoryGuardSnapshot(
            stage: .rejected,
            tier: .balanced,
            promptTokenCount: 128,
            cachedTokenCount: 32,
            newTokenCount: 96,
            maximumGeneratedTokenCount: 64,
            prefillStepSize: 32,
            currentMemoryBytes: 1_000,
            estimatedPeakBytes: 2_000,
            limitBytes: 1_500,
            limitSource: .hostAvailableMemory,
            message: "redacted"
        ))

        let snapshot = MLXObservability.snapshot()
        #expect(snapshot.counters["memory_guard.rejections"] == 1)
        #expect(snapshot.gauges["memory.current_bytes"] == 1_000)
        #expect(snapshot.gauges["memory.estimated_peak_bytes"] == 2_000)
        #expect(snapshot.gauges["memory.limit_bytes"] == 1_500)
        #expect(snapshot.recentEvents.contains { event in
            event.name == "memory_guard.rejected" && event.severity == .warning
        })
    }

    @Test("metrics data creates redacted request summaries")
    func metricsDataCreatesRedactedRequestSummaries() {
        let summary = Self.makeMetricsData().requestSummary(
            totalDuration: .seconds(3),
            modelName: "summary-model",
            strategy: .scalar
        )

        #expect(summary.modelName == "summary-model")
        #expect(summary.strategy == "scalar")
        #expect(summary.promptTokens == 4)
        #expect(summary.generatedTokens == 6)
        #expect(summary.totalTokens == 10)
        #expect(summary.cachedPromptTokens == 2)
        #expect(summary.stopReason == "max_tokens")
        #expect(abs(summary.temperature - 0.1) < 0.0001)
        #expect(abs(summary.topP - 0.8) < 0.0001)
        #expect(summary.topK == 5)
        #expect(summary.grammarKind == "builtinJSON")
    }

    private static var testConfiguration: MLXObservabilityConfiguration {
        MLXObservabilityConfiguration(
            osLogEnabled: false,
            signpostsEnabled: false,
            minimumLogSeverity: .fault,
            keptRecentEventCount: 64,
            keptRecentRequestCount: 8
        )
    }

    private static func makeRequestSummary() -> MLXRequestSummary {
        MLXRequestSummary(
            modelName: "test-model",
            strategy: "scalar",
            promptTokens: 7,
            generatedTokens: 5,
            totalTokens: 12,
            cachedPromptTokens: 3,
            totalDurationSeconds: 2,
            timeToFirstTokenSeconds: 0.25,
            generationTokensPerSecond: 3.3,
            stopReason: "max_tokens",
            temperature: 0.2,
            topP: 0.9
        )
    }

    private static func makeMetricsData() -> MetricsData {
        let clock = ContinuousClock()
        let now = clock.now
        return MetricsData(
            generationStartTime: now,
            promptStartTime: now,
            promptEndTime: now,
            firstTokenTime: now,
            promptTokenCount: 4,
            generatedTokenCount: 6,
            kvCacheBytes: 2_048,
            kvCacheEntries: 8,
            promptCacheReusedTokenCount: 2,
            stopReason: .maxTokens,
            parameters: GenerateParameters(
                maxTokens: 6,
                temperature: 0.1,
                topP: 0.8,
                topK: 5,
                grammar: .json()
            )
        )
    }

    private static func recordDecodePathSamples() async throws {
        _ = try await MLXGenerationDiagnostics.withRecording {
            MLXGenerationDiagnostics.recordDecodePath(MLXDecodePathSnapshot(
                path: .greedyToken,
                processorActive: false,
                argmaxSampler: true,
                greedyModelAvailable: true
            ))
            MLXGenerationDiagnostics.recordDecodePath(MLXDecodePathSnapshot(
                path: .logits,
                processorActive: true,
                argmaxSampler: true,
                greedyModelAvailable: true
            ))
        }
    }

    private static func isGreedyDecodePathEvent(_ event: MLXObservabilityEvent) -> Bool {
        isDecodePathEvent(event, path: "greedyToken", processorActive: false)
    }

    private static func isLogitsDecodePathEvent(_ event: MLXObservabilityEvent) -> Bool {
        isDecodePathEvent(event, path: "logits", processorActive: true)
    }

    private static func isDecodePathEvent(
        _ event: MLXObservabilityEvent,
        path: String,
        processorActive: Bool
    ) -> Bool {
        event.attributes["path"] == path
            && event.attributes["processor_active"] == String(processorActive)
            && event.attributes["argmax_sampler"] == "true"
            && event.attributes["greedy_model"] == "true"
    }
}

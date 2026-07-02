import Foundation
@testable import MLXLocalModels
import Testing

@Suite(
    "MLX real-model continuous-batch prompt cache",
    .serialized,
    .disabled(
        if: !MLXRealModelEnvironment.isEnabled,
        "Set MLX_RUN_REAL_MODEL_TESTS=1 and download models with scripts/download-test-models.sh"
    )
)
struct MLXRealModelBatchCacheTests {
    @Test("Qwen3 reuses memory prompt cache through continuous batching")
    func qwen3ReusesMemoryPromptCacheThroughContinuousBatching() async throws {
        let models = try MLXRealModelCatalog.load()
        guard let model = try MLXRealModelHarness.selectedModel("qwen3-0.6b-4bit", in: models) else {
            return
        }
        let identity = PromptCacheIdentity(stableFingerprint: "qwen3-continuous-cache-real-e2e-v1")
        let session = MLXSession(runtimeCapabilities: .continuousBatching)

        do {
            let observed = try await Self.runCachePair(
                session: session,
                configuration: Self.configuration(for: model),
                identity: identity
            )
            let selectedStrategy = await session.generationExecutionPlan?.selectedStrategy
            await session.unload()
            try Self.verify(observed, selectedStrategy: selectedStrategy)
        } catch {
            await session.unload()
            throw error
        }
    }

    private static func configuration(
        for model: MLXRealModelCatalog.Model
    ) -> ProviderConfiguration {
        ProviderConfiguration(
            location: MLXRealModelEnvironment.modelURL(for: model),
            authentication: .noAuth,
            modelName: "\(model.displayName) continuous-cache-e2e",
            compute: .small,
            runtime: runtime
        )
    }

    private static var runtime: ModelRuntimePreferences {
        ModelRuntimePreferences(
            promptCachePolicy: .memory,
            promptCacheByteLimit: 268_435_456,
            scheduling: .init(
                mode: .continuousBatching,
                maxConcurrentRequests: 2,
                maxQueuedRequests: 4,
                maxBatchSize: 2
            )
        )
    }

    private static func runCachePair(
        session: any MLXGeneratingSession,
        configuration: ProviderConfiguration,
        identity: PromptCacheIdentity
    ) async throws -> (
        result: CachePair,
        events: [MLXGenerationDiagnosticEvent]
    ) {
        try await MLXGenerationDiagnostics.withRecording {
            try await preload(session: session, configuration: configuration)
            let input = input(identity: identity)
            let first = try await collectGeneration(from: await session.stream(input))
            let second = try await collectGeneration(from: await session.stream(input))
            return CachePair(first: first, second: second)
        }
    }

    private static func verify(
        _ run: (result: CachePair, events: [MLXGenerationDiagnosticEvent]),
        selectedStrategy: MLXGenerationExecutionStrategy?
    ) throws {
        let reused = run.result.second.metrics?.usage?.promptCacheReusedTokenCount ?? 0
        let summary = Comment(rawValue: cacheSummary(run.events, result: run.result.second))

        MLXRealModelHarness.verifyGenerated(run.result.first)
        MLXRealModelHarness.verifyGenerated(run.result.second)
        #expect(selectedStrategy == .continuousBatching)
        #expect(promptCachePlans(from: run.events).contains { $0.reusedTokenCount > 0 }, summary)
        #expect(reused > 0, summary)
        try MLXRealModelHarness.verifyPromptCacheProgress(
            run.result.second,
            reusedTokenCount: reused,
            summary: summary
        )
    }

    private static func input(identity: PromptCacheIdentity) -> LLMInput {
        LLMInput(
            context: prompt,
            promptCacheIdentity: identity,
            sampling: .deterministic,
            limits: ResourceLimits(
                maxTokens: max(2, MLXRealModelEnvironment.architectureGenerationTokenLimit),
                maxTime: .seconds(120),
                reusePromptCache: true,
                maxPromptCacheBytes: 268_435_456
            )
        )
    }

    private static var prompt: String {
        let sentence = """
        Continuous batching should reuse the exact MLX prompt cache while still streaming real \
        Foundation Models style output.
        """
        let body = Array(repeating: sentence, count: 32).joined(separator: " ")
        return "/no_think\n\(body)\nReply with one short word."
    }

    private static func preload(
        session: any MLXGeneratingSession,
        configuration: ProviderConfiguration
    ) async throws {
        let progress = await session.preload(configuration: configuration)
        for try await _ in progress {
            // Consume preload progress before generation.
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
                textChunkCount += chunk.text.isEmpty ? 0 : 1
            }
            if case .lifecycle(let event) = chunk.event {
                lifecycleEvents.append(event)
            }
            metrics = chunk.metrics ?? metrics
        }
        return .init(
            text: text,
            textChunkCount: textChunkCount,
            metrics: metrics,
            lifecycleEvents: lifecycleEvents
        )
    }

    private static func promptCachePlans(
        from events: [MLXGenerationDiagnosticEvent]
    ) -> [MLXPromptCachePlanSnapshot] {
        events.compactMap { event in
            guard case .promptCachePlan(let snapshot) = event else {
                return nil
            }
            return snapshot
        }
    }

    private static func cacheSummary(
        _ events: [MLXGenerationDiagnosticEvent],
        result: MLXRealModelHarness.GenerationResult
    ) -> String {
        [
            "usageReused=\(result.metrics?.usage?.promptCacheReusedTokenCount ?? -1)",
            "plans=\(promptCachePlans(from: events))",
            "lifecycle=\(result.lifecycleEvents)"
        ].joined(separator: "\n")
    }

    private struct CachePair {
        let first: MLXRealModelHarness.GenerationResult
        let second: MLXRealModelHarness.GenerationResult
    }
}

import Foundation
@testable import MLXLocalModels
import Testing

@Suite(
    "MLX real-model persistent prompt cache",
    .serialized,
    .disabled(
        if: !MLXRealModelEnvironment.isEnabled,
        "Set MLX_RUN_REAL_MODEL_TESTS=1 and download models with scripts/download-test-models.sh"
    )
)
struct MLXRealModelPersistentPromptCacheTests {
    @Test("Qwen3 restores persistent prompt cache across sessions")
    func qwen3RestoresPersistentPromptCacheAcrossSessions() async throws {
        let models = try MLXRealModelCatalog.load()
        guard let model = try MLXRealModelHarness.selectedModel("qwen3-0.6b-4bit", in: models) else {
            return
        }
        let identity = Self.cacheIdentity
        let configuration = Self.configuration(for: model)

        Self.removePersistentArtifacts(identity: identity, configuration: configuration)
        defer { Self.removePersistentArtifacts(identity: identity, configuration: configuration) }
        MLXGenerationDiagnostics.resetPromptCacheObservability()

        let first = try await Self.runPersistentGeneration(
            model: model,
            configuration: configuration,
            identity: identity
        )
        try Self.verifyFirstPersistentRun(first)

        MLXPersistentPromptCacheBlockStore.clearHotCache()
        let second = try await Self.runPersistentGeneration(
            model: model,
            configuration: configuration,
            identity: identity
        )
        try Self.verifySecondPersistentRun(second)
    }

    private static var cacheIdentity: PromptCacheIdentity {
        PromptCacheIdentity(stableFingerprint: "qwen3-persistent-cache-real-e2e-v1")
    }

    private static func configuration(
        for model: MLXRealModelCatalog.Model
    ) -> ProviderConfiguration {
        ProviderConfiguration(
            location: MLXRealModelEnvironment.modelURL(for: model),
            authentication: .noAuth,
            modelName: "\(model.displayName) persistent-cache-e2e",
            compute: .small,
            runtime: persistentRuntime
        )
    }

    private static var persistentRuntime: ModelRuntimePreferences {
        ModelRuntimePreferences(
            promptCachePolicy: .persistent,
            promptCacheByteLimit: 268_435_456,
            persistentPromptCacheTotalByteLimit: 536_870_912,
            persistentPromptCacheHotByteLimit: 67_108_864
        )
    }

    private static func runPersistentGeneration(
        model: MLXRealModelCatalog.Model,
        configuration: ProviderConfiguration,
        identity: PromptCacheIdentity
    ) async throws -> (result: MLXRealModelHarness.GenerationResult, events: [MLXGenerationDiagnosticEvent]) {
        let session = MLXSessionFactory.create()
        do {
            try await preload(session: session, configuration: configuration)
            let result = try await MLXGenerationDiagnostics.withRecording {
                try await collectGeneration(from: await session.stream(input(identity: identity)))
            }
            await session.unload()
            return result
        } catch {
            await session.unload()
            throw error
        }
    }

    private static func verifyFirstPersistentRun(
        _ run: (result: MLXRealModelHarness.GenerationResult, events: [MLXGenerationDiagnosticEvent])
    ) throws {
        let plan = try #require(promptCachePlans(from: run.events).last)
        let counters = try lastPromptCacheCounters(from: run.events)
        let summary = Comment(rawValue: cacheSummary(run.events, result: run.result))

        MLXRealModelHarness.verifyGenerated(run.result)
        #expect(plan.promptTokenCount >= 256, summary)
        #expect(run.result.metrics?.usage?.promptCacheReusedTokenCount == 0, summary)
        #expect(counters.ssdSaves > 0, summary)
        try MLXRealModelHarness.verifyPromptCacheProgress(
            run.result,
            reusedTokenCount: 0,
            summary: summary
        )
    }

    private static func verifySecondPersistentRun(
        _ run: (result: MLXRealModelHarness.GenerationResult, events: [MLXGenerationDiagnosticEvent])
    ) throws {
        let reused = run.result.metrics?.usage?.promptCacheReusedTokenCount ?? 0
        let lookups = promptCacheLookups(from: run.events)
        let counters = try lastPromptCacheCounters(from: run.events)
        let summary = Comment(rawValue: cacheSummary(run.events, result: run.result))

        MLXRealModelHarness.verifyGenerated(run.result)
        #expect(reused >= 256, summary)
        #expect(lookups.contains { lookup in
            persistentStrategies.contains(lookup.strategy) && lookup.reusedTokenCount >= 256
        }, summary)
        #expect(counters.ssdDiskLoads > 0, summary)
        try MLXRealModelHarness.verifyPromptCacheProgress(
            run.result,
            reusedTokenCount: reused,
            summary: summary
        )
    }

    private static let persistentStrategies: Set<MLXPromptCacheLookupSnapshot.Strategy> = [
        .persistentSegments,
        .persistentSnapshot
    ]

    private static func input(identity: PromptCacheIdentity) -> LLMInput {
        LLMInput(
            context: longPrompt,
            promptCacheIdentity: identity,
            sampling: .deterministic,
            limits: ResourceLimits(
                maxTokens: 1,
                maxTime: .seconds(120),
                reusePromptCache: true,
                maxPromptCacheBytes: 268_435_456
            )
        )
    }

    private static var longPrompt: String {
        let sentence = """
        MLXFoundationModel persistent cache validation keeps local Apple silicon generation \
        fast deterministic private and compatible with Foundation Models style sessions.
        """
        let body = Array(repeating: sentence, count: 80).joined(separator: " ")
        return "/no_think\n\(body)\nReply with exactly one short word."
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

    private static func promptCacheLookups(
        from events: [MLXGenerationDiagnosticEvent]
    ) -> [MLXPromptCacheLookupSnapshot] {
        events.compactMap { event in
            guard case .promptCacheLookup(let snapshot) = event else {
                return nil
            }
            return snapshot
        }
    }

    private static func lastPromptCacheCounters(
        from events: [MLXGenerationDiagnosticEvent]
    ) throws -> MLXPromptCacheObservabilityCounters {
        try #require(events.compactMap { event in
            guard case .promptCacheObservability(let snapshot) = event else {
                return nil
            }
            return snapshot.counters
        }.last)
    }

    private static func cacheSummary(
        _ events: [MLXGenerationDiagnosticEvent],
        result: MLXRealModelHarness.GenerationResult
    ) -> String {
        [
            "usageReused=\(result.metrics?.usage?.promptCacheReusedTokenCount ?? -1)",
            "plans=\(promptCachePlans(from: events))",
            "lookups=\(promptCacheLookups(from: events))",
            "lifecycle=\(result.lifecycleEvents)"
        ].joined(separator: "\n")
    }

    private static func removePersistentArtifacts(
        identity: PromptCacheIdentity,
        configuration: ProviderConfiguration
    ) {
        try? FileManager.default.removeItem(at: MLXPersistentPromptCacheStore.url(for: configuration))
        removePersistentRecords(identity: identity, root: MLXPersistentPromptCacheBlockStore.rootURL())
        removePersistentRecords(identity: identity, root: MLXPersistentPromptCacheSegmentStore.rootURL())
        MLXPersistentPromptCacheBlockStore.clearHotCache()
    }

    private static func removePersistentRecords(
        identity: PromptCacheIdentity,
        root: URL
    ) {
        guard let records = try? MLXPersistentPromptCacheBlockStore.scan(rootURL: root) else {
            return
        }
        for record in records where record.signature.promptCacheIdentity == identity {
            try? FileManager.default.removeItem(
                at: MLXPersistentPromptCacheBlockStore.dataURL(for: record, rootURL: root)
            )
            try? FileManager.default.removeItem(
                at: MLXPersistentPromptCacheBlockStore.metadataURL(for: record, rootURL: root)
            )
        }
    }
}

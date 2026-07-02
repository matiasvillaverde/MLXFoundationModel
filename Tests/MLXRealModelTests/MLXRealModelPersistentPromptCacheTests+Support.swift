import Foundation
@testable import MLXLocalModels
import Testing

extension MLXRealModelPersistentPromptCacheTests {
    static func verifyPersistentCacheRestore(
        model: MLXRealModelCatalog.Model
    ) async throws {
        let identity = Self.cacheIdentity(for: model)
        let configuration = Self.configuration(for: model)
        let firstPrompt = Self.persistentPrompt
        let secondPrompt = Self.persistentPromptWithSuffix

        Self.removePersistentArtifacts(identity: identity, configuration: configuration)
        defer { Self.removePersistentArtifacts(identity: identity, configuration: configuration) }
        MLXGenerationDiagnostics.resetPromptCacheObservability()

        let first = try await Self.runPersistentGeneration(
            model: model,
            configuration: configuration,
            identity: identity,
            prompt: firstPrompt
        )
        try Self.verifyFirstPersistentRun(first, modelID: model.id)

        MLXPersistentPromptCacheBlockStore.clearHotCache()
        let second = try await Self.runPersistentGeneration(
            model: model,
            configuration: configuration,
            identity: identity,
            prompt: secondPrompt
        )
        try Self.verifySecondPersistentRun(second, modelID: model.id)
    }

    private static func cacheIdentity(
        for model: MLXRealModelCatalog.Model
    ) -> PromptCacheIdentity {
        PromptCacheIdentity(stableFingerprint: "\(model.id)-persistent-cache-real-e2e-v2")
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
            promptCacheByteLimit: 536_870_912,
            persistentPromptCacheTotalByteLimit: 1_073_741_824,
            persistentPromptCacheHotByteLimit: 536_870_912
        )
    }

    private static func runPersistentGeneration(
        model: MLXRealModelCatalog.Model,
        configuration: ProviderConfiguration,
        identity: PromptCacheIdentity,
        prompt: String
    ) async throws -> (result: MLXRealModelHarness.GenerationResult, events: [MLXGenerationDiagnosticEvent]) {
        let session = MLXSessionFactory.create()
        do {
            try await preload(session: session, configuration: configuration)
            let result = try await MLXGenerationDiagnostics.withRecording {
                try await collectGeneration(
                    from: await session.stream(input(identity: identity, prompt: prompt))
                )
            }
            await session.unload()
            return result
        } catch {
            await session.unload()
            throw error
        }
    }

    private static func verifyFirstPersistentRun(
        _ run: (result: MLXRealModelHarness.GenerationResult, events: [MLXGenerationDiagnosticEvent]),
        modelID: String
    ) throws {
        let plan = try #require(promptCachePlans(from: run.events).last)
        let counters = try lastPromptCacheCounters(from: run.events)
        let summary = Comment(rawValue: cacheSummary(
            run.events,
            result: run.result,
            modelID: modelID,
            phase: "first"
        ))

        try verifyGenerated(run.result, summary: summary)
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
        _ run: (result: MLXRealModelHarness.GenerationResult, events: [MLXGenerationDiagnosticEvent]),
        modelID: String
    ) throws {
        let reused = run.result.metrics?.usage?.promptCacheReusedTokenCount ?? 0
        let lookups = promptCacheLookups(from: run.events)
        let counters = try lastPromptCacheCounters(from: run.events)
        let summary = Comment(rawValue: cacheSummary(
            run.events,
            result: run.result,
            modelID: modelID,
            phase: "second"
        ))

        try verifyGenerated(run.result, summary: summary)
        #expect(reused >= 256, summary)
        #expect(hasPersistentRestore(lookups), summary)
        #expect(hasPlannerReuse(lookups), summary)
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

    private static func hasPersistentRestore(
        _ lookups: [MLXPromptCacheLookupSnapshot]
    ) -> Bool {
        lookups.contains { lookup in
            persistentStrategies.contains(lookup.strategy) && lookup.reusedTokenCount >= 256
        }
    }

    private static func hasPlannerReuse(
        _ lookups: [MLXPromptCacheLookupSnapshot]
    ) -> Bool {
        lookups.contains { lookup in
            lookup.strategy == .blockIndex && lookup.reusedTokenCount >= 256
        } || lookups.contains { lookup in
            lookup.strategy == .linear && lookup.reusedTokenCount >= 256
        }
    }

    private static func verifyGenerated(
        _ result: MLXRealModelHarness.GenerationResult,
        summary: Comment
    ) throws {
        try #require(!result.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty, summary)
        try #require(result.textChunkCount > 0, summary)
        try #require((result.metrics?.usage?.generatedTokens ?? 0) > 0, summary)
    }

    private static func input(identity: PromptCacheIdentity, prompt: String) -> LLMInput {
        LLMInput(
            context: prompt,
            promptCacheIdentity: identity,
            sampling: .deterministic,
            limits: ResourceLimits(
                maxTokens: max(8, MLXRealModelEnvironment.architectureGenerationTokenLimit),
                maxTime: .seconds(
                    MLXRealModelEnvironment.architectureGenerationTimeoutSeconds
                ),
                reusePromptCache: true,
                maxPromptCacheBytes: 536_870_912
            )
        )
    }

    private static var persistentPrompt: String {
        let sentence = """
        MLXFoundationModel persistent cache validation keeps local Apple silicon generation \
        fast deterministic private and compatible with Foundation Models style sessions.
        """
        let body = Array(repeating: sentence, count: 20).joined(separator: " ")
        return "\(body)\nReply with one short word."
    }

    private static var persistentPromptWithSuffix: String {
        "\(persistentPrompt)\nThen reply with one more short word."
    }
}

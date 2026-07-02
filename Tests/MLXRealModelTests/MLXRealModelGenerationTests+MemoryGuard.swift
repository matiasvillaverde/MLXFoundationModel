@testable import MLXLocalModels
import Testing

extension MLXRealModelGenerationTests {
    @Test("Qwen3 records memory guard admission decisions")
    func qwen3RecordsMemoryGuardAdmissionDecisions() async throws {
        let models = try MLXRealModelCatalog.load()
        guard let model = try MLXRealModelHarness.selectedModel("qwen3-0.6b-4bit", in: models) else {
            return
        }

        try await Self.verifyAllowedMemoryGuardGeneration(model: model)
        try await Self.verifyRejectedMemoryGuardModelLoad(model: model)
    }

    private static func verifyAllowedMemoryGuardGeneration(
        model: MLXRealModelCatalog.Model
    ) async throws {
        let runtime = ModelRuntimePreferences(promptCachePolicy: .memory, memoryGuard: .balanced)
        let observed = try await MLXRealModelHarness.runWithDiagnostics(
            model: model,
            sampling: .deterministic,
            limits: ResourceLimits(maxTokens: 4, maxTime: .seconds(120), reusePromptCache: false),
            prompt: Self.memoryGuardPrompt,
            runtime: runtime
        )
        let snapshots = Self.memoryGuardSnapshots(from: observed.events)
        let stages = snapshots.map(\.stage)
        let summary = Comment(rawValue: Self.memoryGuardSummary(snapshots))

        MLXRealModelHarness.verifyGenerated(observed.result)
        #expect(stages.contains(.modelLoadAllowed), summary)
        #expect(stages.contains(.allowed), summary)
    }

    private static func verifyRejectedMemoryGuardModelLoad(
        model: MLXRealModelCatalog.Model
    ) async throws {
        let observed = try await Self.preloadWithTooSmallMemoryGuard(model: model)
        let snapshots = Self.memoryGuardSnapshots(from: observed.events)
        let rejected = try #require(snapshots.last)
        let summary = Comment(rawValue: Self.memoryGuardSummary(snapshots))

        #expect(observed.error != nil, summary)
        #expect(rejected.stage == .modelLoadRejected, summary)
        #expect(rejected.tier == .custom, summary)
        #expect(rejected.limitBytes == 1, summary)
    }

    private static func preloadWithTooSmallMemoryGuard(
        model: MLXRealModelCatalog.Model
    ) async throws -> (error: Error?, events: [MLXGenerationDiagnosticEvent]) {
        let recorded = try await MLXGenerationDiagnostics.withRecording {
            await Self.preloadWithTooSmallMemoryGuardReturningError(model: model)
        }
        return (error: recorded.result, events: recorded.events)
    }

    private static func preloadWithTooSmallMemoryGuardReturningError(
        model: MLXRealModelCatalog.Model
    ) async -> Error? {
        let runtime = ModelRuntimePreferences(
            promptCachePolicy: .memory,
            memoryGuard: MLXMemoryGuardConfiguration(tier: .custom, customLimitBytes: 1)
        )
        let session = MLXSessionFactory.create()
        let progress = await session.preload(configuration: Self.configuration(for: model, runtime: runtime))
        do {
            for try await _ in progress {
                // Drain preload progress until the guard rejects or loading finishes.
            }
            await session.unload()
            return nil
        } catch {
            await session.unload()
            return error
        }
    }

    private static var memoryGuardPrompt: String {
        "/no_think\nReply with one short word about bounded local inference."
    }

    private static func configuration(
        for model: MLXRealModelCatalog.Model,
        runtime: ModelRuntimePreferences
    ) -> ProviderConfiguration {
        ProviderConfiguration(
            location: MLXRealModelEnvironment.modelURL(for: model),
            authentication: .noAuth,
            modelName: "\(model.displayName) memory-guard-e2e",
            compute: .small,
            runtime: runtime
        )
    }

    private static func memoryGuardSnapshots(
        from events: [MLXGenerationDiagnosticEvent]
    ) -> [MLXMemoryGuardSnapshot] {
        events.compactMap { event in
            guard case .memoryGuard(let snapshot) = event else {
                return nil
            }
            return snapshot
        }
    }

    private static func memoryGuardSummary(_ snapshots: [MLXMemoryGuardSnapshot]) -> String {
        "memoryGuard=\(snapshots)"
    }
}

import Foundation
@testable import MLXLocalModels
import Testing

@Suite(
    "MLX real-model Gemma 4 shared-KV MTP",
    .serialized,
    .disabled(
        if: !MLXRealModelEnvironment.isEnabled,
        "Set MLX_RUN_REAL_MODEL_TESTS=1 and download models with scripts/download-test-models.sh"
    )
)
struct MLXRealModelGemma4MTPTests {
    @Test("Gemma 4 E2B shared-KV MTP matches greedy target output")
    func gemma4E2BSharedKVMTPEndToEnd() async throws {
        try await Self.verifySharedKVMTP(
            modelID: "gemma-4-e2b-it-4bit",
            assistantDirectory: "gemma-4-E2B-it-assistant-bf16"
        )
    }

    @Test("Gemma 4 E4B shared-KV MTP matches greedy target output")
    func gemma4E4BSharedKVMTPEndToEnd() async throws {
        try await Self.verifySharedKVMTP(
            modelID: "gemma-4-e4b-it-4bit",
            assistantDirectory: "gemma-4-E4B-it-assistant-bf16"
        )
    }

    private static func verifySharedKVMTP(
        modelID: String,
        assistantDirectory: String
    ) async throws {
        guard let setup = try Self.sharedKVMTPSetup(
            modelID: modelID,
            assistantDirectory: assistantDirectory
        ) else {
            return
        }

        let prompt = "Write one concise sentence about local inference performance."
        let limits = ResourceLimits(maxTokens: 16, maxTime: .seconds(180), reusePromptCache: false)
        let baseline = try await Self.runBaseline(model: setup.model, prompt: prompt, limits: limits)
        let accelerated = try await Self.runSharedKVMTP(
            model: setup.model,
            assistantURL: setup.assistantURL,
            prompt: prompt,
            limits: limits
        )

        try Self.verifySharedKVMTPResult(
            baseline: baseline.result,
            accelerated: accelerated.result,
            events: accelerated.events
        )
        Self.printBenchmarkSummary(
            modelID: modelID,
            baseline: baseline.result,
            mtp: accelerated.result,
            events: accelerated.events
        )
    }

    private static func verifySharedKVMTPResult(
        baseline: MLXRealModelHarness.GenerationResult,
        accelerated: MLXRealModelHarness.GenerationResult,
        events: [MLXGenerationDiagnosticEvent]
    ) throws {
        MLXRealModelHarness.verifyGenerated(baseline)
        MLXRealModelHarness.verifyGenerated(accelerated)
        MLXRealModelHarness.verifyGeneratedTokenDiagnostics(
            MLXRealModelHarness.generatedTokenSnapshots(from: events),
            result: accelerated
        )
        #expect(
            accelerated.text == baseline.text,
            Comment(rawValue: """
            Shared-KV MTP must preserve greedy target output.
            baseline=\(baseline.text.debugDescription)
            mtp=\(accelerated.text.debugDescription)
            """)
        )

        try Self.verifySharedKVMTPPlan(events)
        try Self.verifySharedKVMTPDrafting(events)
    }

    private static func sharedKVMTPSetup(
        modelID: String,
        assistantDirectory: String
    ) throws -> (model: MLXRealModelCatalog.Model, assistantURL: URL)? {
        let models = try MLXRealModelCatalog.load()
        guard let model = try MLXRealModelHarness.selectedModel(modelID, in: models) else {
            return nil
        }
        let assistantURL = MLXRealModelEnvironment.modelRoot.appendingPathComponent(
            assistantDirectory,
            isDirectory: true
        )
        #expect(
            MLXRealModelEnvironment.hasModelFiles(at: assistantURL),
            Comment(rawValue: "Missing Gemma 4 assistant drafter at \(assistantURL.path).")
        )
        guard MLXRealModelEnvironment.hasModelFiles(at: assistantURL) else {
            return nil
        }
        return (model, assistantURL)
    }

    private static func runBaseline(
        model: MLXRealModelCatalog.Model,
        prompt: String,
        limits: ResourceLimits
    ) async throws -> (result: MLXRealModelHarness.GenerationResult, events: [MLXGenerationDiagnosticEvent]) {
        try await MLXRealModelHarness.runWithDiagnostics(
            model: model,
            sampling: .deterministic,
            limits: limits,
            prompt: prompt,
            runtime: ModelRuntimePreferences(promptCachePolicy: .memory)
        )
    }

    private static func runSharedKVMTP(
        model: MLXRealModelCatalog.Model,
        assistantURL: URL,
        prompt: String,
        limits: ResourceLimits
    ) async throws -> (result: MLXRealModelHarness.GenerationResult, events: [MLXGenerationDiagnosticEvent]) {
        try await MLXRealModelHarness.runWithDiagnostics(
            model: model,
            sampling: .deterministic,
            limits: limits,
            prompt: prompt,
            runtime: Self.sharedKVMTPRuntime(assistantURL: assistantURL),
            runtimeCapabilities: .continuousBatching
        )
    }

    private static func sharedKVMTPRuntime(assistantURL: URL) -> ModelRuntimePreferences {
        ModelRuntimePreferences(
            promptCachePolicy: .memory,
            scheduling: .init(mode: .continuousBatching, maxConcurrentRequests: 2, maxBatchSize: 2),
            optimization: .vlmMTP(draftModelID: assistantURL.path)
        )
    }

    private static func verifySharedKVMTPPlan(
        _ events: [MLXGenerationDiagnosticEvent]
    ) throws {
        let snapshot = try #require(Self.executionPlanSnapshots(from: events).last)

        #expect(snapshot.requestedStrategy == MLXGenerationExecutionStrategy.continuousBatching)
        #expect(snapshot.selectedStrategy == MLXGenerationExecutionStrategy.scalar)
        #expect(snapshot.reason == MLXGenerationExecutionPlanReason.sharedKVMTPRequiresScalar)
        #expect(snapshot.effectiveMaxBatchSize == 1)
    }

    private static func verifySharedKVMTPDrafting(
        _ events: [MLXGenerationDiagnosticEvent]
    ) throws {
        let snapshots = Self.speculativeSnapshots(from: events)

        #expect(snapshots.contains { $0.numDraftTokens == 3 })
    }

    private static func printBenchmarkSummary(
        modelID: String,
        baseline: MLXRealModelHarness.GenerationResult,
        mtp: MLXRealModelHarness.GenerationResult,
        events: [MLXGenerationDiagnosticEvent]
    ) {
        guard let baselineTPS = tokensPerSecond(for: baseline),
              let mtpTPS = tokensPerSecond(for: mtp) else {
            return
        }
        let speedup = mtpTPS / max(baselineTPS, .leastNonzeroMagnitude)
        let acceptance = acceptanceSummary(from: events)
        print(
            String(
                format: "Gemma4 MTP %@ baseline %.2f tok/s, mtp %.2f tok/s, %.2fx, accepted %.1f%%",
                modelID,
                baselineTPS,
                mtpTPS,
                speedup,
                acceptance
            )
        )
    }

    private static func acceptanceSummary(from events: [MLXGenerationDiagnosticEvent]) -> Double {
        let snapshots = speculativeSnapshots(from: events)
        let accepted = snapshots.compactMap(\.acceptedDraftTokens).reduce(0, +)
        let rejected = snapshots.compactMap(\.rejectedDraftTokens).reduce(0, +)
        let total = accepted + rejected
        guard total > 0 else {
            return 0
        }
        return 100 * Double(accepted) / Double(total)
    }

    private static func tokensPerSecond(for result: MLXRealModelHarness.GenerationResult) -> Double? {
        guard let metrics = result.metrics,
              let timing = metrics.timing,
              let generatedTokens = metrics.usage?.generatedTokens,
              generatedTokens > 0 else {
            return nil
        }
        let totalSeconds = seconds(timing.totalTime)
        let promptSeconds = timing.promptProcessingTime.map(seconds) ?? 0
        let generationSeconds = max(totalSeconds - promptSeconds, 0)
        guard generationSeconds > 0 else {
            return nil
        }
        return Double(generatedTokens) / generationSeconds
    }

    private static func seconds(_ duration: Duration) -> Double {
        let components = duration.components
        return Double(components.seconds) + Double(components.attoseconds) / 1e18
    }

    private static func executionPlanSnapshots(
        from events: [MLXGenerationDiagnosticEvent]
    ) -> [MLXGenerationExecutionPlanSnapshot] {
        events.compactMap { event in
            guard case .executionPlan(let snapshot) = event else {
                return nil
            }
            return snapshot
        }
    }

    private static func speculativeSnapshots(
        from events: [MLXGenerationDiagnosticEvent]
    ) -> [MLXSpeculativeDecodingSnapshot] {
        events.compactMap { event in
            guard case .speculativeDecoding(let snapshot) = event else {
                return nil
            }
            return snapshot
        }
    }
}

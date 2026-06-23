import Foundation
@testable import MLXLocalModels
import Testing

@Suite(
    "MLX real-model speculative decoding",
    .serialized,
    .disabled(
        if: !MLXRealModelEnvironment.isEnabled,
        "Set MLX_RUN_REAL_MODEL_TESTS=1 and download models with scripts/download-test-models.sh"
    )
)
struct MLXRealModelSpeculativeDecodingTests {
    @Test("same-model speculative generation emits real tokens through scalar execution")
    func sameModelSpeculativeGenerationEmitsRealTokens() async throws {
        guard let model = try Self.selectedModel() else {
            return
        }
        let observed = try await MLXRealModelHarness.runWithDiagnostics(
            model: model,
            sampling: .deterministic,
            limits: ResourceLimits(maxTokens: 2, maxTime: .seconds(120), reusePromptCache: false),
            prompt: "Write two words about local inference.",
            runtime: Self.speculativeRuntime,
            runtimeCapabilities: .continuousBatching
        )

        MLXRealModelHarness.verifyGenerated(observed.result)
        MLXRealModelHarness.verifyGeneratedTokenDiagnostics(
            MLXRealModelHarness.generatedTokenSnapshots(from: observed.events),
            result: observed.result
        )
        try Self.verifySpeculativeScalarPlan(observed.events)
        try Self.verifySpeculativeDrafting(observed.events)
    }

    @Test("Qwen external draft speculative decoding preserves greedy target output")
    func qwenExternalDraftSpeculativeDecodingPreservesGreedyTargetOutput() async throws {
        guard let setup = try Self.qwenExternalDraftSetup() else {
            return
        }

        let run = try await Self.qwenExternalDraftRun(setup: setup)

        try Self.verifyExternalDraftRun(run)
        Self.printExternalDraftBenchmark(
            targetID: setup.target.id,
            draftID: setup.draft.id,
            baseline: run.baseline.result,
            accelerated: run.accelerated.result,
            events: run.accelerated.events
        )
    }

    private static func qwenExternalDraftRun(
        setup: QwenExternalDraftFixtures.Setup
    ) async throws -> QwenExternalDraftFixtures.Run {
        let prompt = "/no_think\nContinue the comma-separated list: item001, item002, item003,"
        let limits = ResourceLimits(
            maxTokens: Self.externalDraftGenerationTokenLimit,
            maxTime: .seconds(180),
            reusePromptCache: false
        )
        let baseline = try await MLXRealModelHarness.runWithDiagnostics(
            model: setup.target,
            sampling: .deterministic,
            limits: limits,
            prompt: prompt,
            runtime: ModelRuntimePreferences(promptCachePolicy: .memory)
        )
        let accelerated = try await MLXRealModelHarness.runWithDiagnostics(
            model: setup.target,
            sampling: .deterministic,
            limits: limits,
            prompt: prompt,
            runtime: Self.externalDraftRuntime(draftURL: setup.draftURL),
            runtimeCapabilities: .continuousBatching
        )

        return QwenExternalDraftFixtures.Run(baseline: baseline, accelerated: accelerated)
    }

    private static var speculativeRuntime: ModelRuntimePreferences {
        ModelRuntimePreferences(
            promptCachePolicy: .memory,
            speculativeDecodingMode: .sameModelDraft,
            speculativeDraftTokens: 2,
            scheduling: .init(mode: .continuousBatching, maxConcurrentRequests: 2, maxBatchSize: 2)
        )
    }

    private static func selectedModel() throws -> MLXRealModelCatalog.Model? {
        let models = try MLXRealModelCatalog.load()
        return try MLXRealModelHarness.selectedModel("llama-3.2-1b-instruct-4bit", in: models)
    }

    private static func qwenExternalDraftSetup() throws -> QwenExternalDraftFixtures.Setup? {
        let models = try MLXRealModelCatalog.load()
        let targetID = ProcessInfo.processInfo.environment["MLX_EXTERNAL_DRAFT_TARGET_ID"]
            ?? "qwen3-4b-4bit"
        let draftID = ProcessInfo.processInfo.environment["MLX_EXTERNAL_DRAFT_MODEL_ID"]
            ?? "qwen3-0.6b-4bit"
        guard let target = try MLXRealModelHarness.selectedModel(targetID, in: models) else {
            return nil
        }
        let draft = try #require(models.first { $0.id == draftID })
        let draftURL = MLXRealModelEnvironment.modelURL(for: draft)
        #expect(
            MLXRealModelEnvironment.hasModelFiles(at: draftURL),
            Comment(rawValue: "Missing Qwen external draft model at \(draftURL.path).")
        )
        guard MLXRealModelEnvironment.hasModelFiles(at: draftURL) else {
            return nil
        }
        return QwenExternalDraftFixtures.Setup(target: target, draft: draft, draftURL: draftURL)
    }

    private static func externalDraftRuntime(draftURL: URL) -> ModelRuntimePreferences {
        ModelRuntimePreferences(
            promptCachePolicy: .memory,
            speculativeDraftTokens: Self.externalDraftTokenCount,
            scheduling: .init(mode: .continuousBatching, maxConcurrentRequests: 2, maxBatchSize: 2),
            optimization: .externalDraft(draftModelID: draftURL.path)
        )
    }

    private static var externalDraftTokenCount: Int {
        guard let rawValue = ProcessInfo.processInfo.environment["MLX_EXTERNAL_DRAFT_TOKENS"],
              let value = Int(rawValue) else {
            return 3
        }
        return max(1, value)
    }

    private static var externalDraftGenerationTokenLimit: Int {
        guard let rawValue = ProcessInfo.processInfo.environment["MLX_EXTERNAL_DRAFT_GENERATION_TOKENS"],
              let value = Int(rawValue) else {
            return 24
        }
        return max(1, value)
    }

    private static func verifySpeculativeScalarPlan(
        _ events: [MLXGenerationDiagnosticEvent]
    ) throws {
        let snapshot = try #require(Self.executionPlanSnapshots(from: events).last)

        #expect(snapshot.requestedStrategy == MLXGenerationExecutionStrategy.continuousBatching)
        #expect(snapshot.selectedStrategy == MLXGenerationExecutionStrategy.scalar)
        #expect(snapshot.reason == MLXGenerationExecutionPlanReason.speculativeDecodingRequiresScalar)
        #expect(snapshot.effectiveMaxBatchSize == 1)
    }

    private static func verifySpeculativeDrafting(
        _ events: [MLXGenerationDiagnosticEvent]
    ) throws {
        let snapshots = Self.speculativeSnapshots(from: events)

        #expect(try #require(snapshots.last).numDraftTokens == 2)
    }

    private static func verifySpeculativeAcceptanceDiagnostics(
        _ events: [MLXGenerationDiagnosticEvent]
    ) throws {
        let snapshots = Self.speculativeSnapshots(from: events)
        let roundSnapshots = snapshots.filter { $0.acceptedDraftTokens != nil }

        #expect(try #require(snapshots.first).numDraftTokens == Self.externalDraftTokenCount)
        #expect(!roundSnapshots.isEmpty)
        #expect(roundSnapshots.allSatisfy { $0.emittedTokens != nil })
    }

    private static func verifyExternalDraftRun(_ run: QwenExternalDraftFixtures.Run) throws {
        MLXRealModelHarness.verifyGenerated(run.baseline.result)
        MLXRealModelHarness.verifyGenerated(run.accelerated.result)
        MLXRealModelHarness.verifyGeneratedTokenDiagnostics(
            MLXRealModelHarness.generatedTokenSnapshots(from: run.accelerated.events),
            result: run.accelerated.result
        )
        #expect(
            run.accelerated.result.text == run.baseline.result.text,
            Comment(rawValue: """
            External draft speculative decoding must preserve greedy target output.
            baseline=\(run.baseline.result.text.debugDescription)
            accelerated=\(run.accelerated.result.text.debugDescription)
            """)
        )
        try Self.verifySpeculativeScalarPlan(run.accelerated.events)
        try Self.verifySpeculativeAcceptanceDiagnostics(run.accelerated.events)
    }

    private static func printExternalDraftBenchmark(
        targetID: String,
        draftID: String,
        baseline: MLXRealModelHarness.GenerationResult,
        accelerated: MLXRealModelHarness.GenerationResult,
        events: [MLXGenerationDiagnosticEvent]
    ) {
        guard let baselineTPS = tokensPerSecond(for: baseline),
              let acceleratedTPS = tokensPerSecond(for: accelerated) else {
            return
        }
        let speedup = acceleratedTPS / max(baselineTPS, .leastNonzeroMagnitude)
        let acceptance = acceptanceSummary(from: events)
        let format = "ExternalDraft target=%@ draft=%@ generated=%d/%d " +
            "baseline %.2f tok/s, accelerated %.2f tok/s, %.2fx, accepted %.1f%%"
        print(
            String(
                format: format,
                targetID,
                draftID,
                baseline.metrics?.usage?.generatedTokens ?? 0,
                accelerated.metrics?.usage?.generatedTokens ?? 0,
                baselineTPS,
                acceleratedTPS,
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

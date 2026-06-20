import Foundation
@testable import MLXLocalModels
import Testing

@Suite("Model runtime continuous batching")
struct ModelRuntimeContinuousBatchingTests {
    @Test("continuous-capable MLX session selects batched admission")
    func continuousCapableMLXSessionSelectsBatchedAdmission() async throws {
        let preferences = ModelRuntimePreferences(
            scheduling: .init(
                mode: .continuousBatching,
                maxConcurrentRequests: 5,
                maxQueuedRequests: 13,
                maxBatchSize: 4
            )
        )
        let session = MLXSession(
            configuration: ProviderConfiguration(
                location: URL(fileURLWithPath: "/tmp/continuous-scheduling-test-model"),
                modelName: "continuous-scheduling-test-model",
                runtime: preferences
            ),
            runtimeCapabilities: .continuousBatching
        )

        let recording = try await Self.recordRuntimeConfiguration(for: session)

        #expect(recording.plan.requestedStrategy == .continuousBatching)
        #expect(recording.plan.selectedStrategy == .continuousBatching)
        #expect(recording.plan.reason == .continuousBatchingSelected)
        #expect(recording.admissionSnapshot.maxConcurrentRequests == 5)
        #expect(recording.admissionSnapshot.maxQueuedRequests == 13)
        #expect(recording.admissionSnapshot.maxBatchSize == 4)
        #expect(recording.executionSnapshot.requestedStrategy == .continuousBatching)
        #expect(recording.executionSnapshot.selectedStrategy == .continuousBatching)
        #expect(recording.executionSnapshot.reason == .continuousBatchingSelected)
        #expect(recording.executionSnapshot.requestedMaxBatchSize == 4)
        #expect(recording.executionSnapshot.effectiveMaxBatchSize == 4)
    }

    @Test("continuous batching configures a live stream engine")
    func continuousBatchingConfiguresLiveStreamEngine() async throws {
        let session = Self.continuousBatchSession()

        try await session.applyRuntimeConfiguration()

        #expect(await session.continuousBatchEngine != nil)
    }

    @Test("same-model speculative decoding uses scalar admission even when batching is requested")
    func sameModelSpeculativeDecodingUsesScalarAdmission() async throws {
        let session = Self.speculativeContinuousBatchSession()
        let reason = MLXGenerationExecutionPlanReason.speculativeDecodingRequiresScalar

        let recording = try await Self.recordRuntimeConfiguration(for: session)

        #expect(recording.plan.requestedStrategy == MLXGenerationExecutionStrategy.continuousBatching)
        #expect(recording.plan.selectedStrategy == MLXGenerationExecutionStrategy.scalar)
        #expect(recording.plan.reason == reason)
        #expect(recording.admissionSnapshot.maxConcurrentRequests == 1)
        #expect(recording.admissionSnapshot.maxQueuedRequests == 13)
        #expect(recording.admissionSnapshot.maxBatchSize == 1)
        #expect(
            recording.executionSnapshot.requestedStrategy == MLXGenerationExecutionStrategy.continuousBatching
        )
        #expect(recording.executionSnapshot.selectedStrategy == MLXGenerationExecutionStrategy.scalar)
        #expect(recording.executionSnapshot.reason == reason)
        #expect(recording.executionSnapshot.requestedMaxBatchSize == 4)
        #expect(recording.executionSnapshot.effectiveMaxBatchSize == 1)
        #expect(await session.continuousBatchEngine == nil)
    }

    @Test("native MTP uses scalar admission even when batching is requested")
    func nativeMTPUsesScalarAdmission() async throws {
        let session = Self.nativeMTPContinuousBatchSession()
        let reason = MLXGenerationExecutionPlanReason.nativeMTPRequiresScalar

        let recording = try await Self.recordRuntimeConfiguration(for: session)

        #expect(recording.plan.requestedStrategy == MLXGenerationExecutionStrategy.continuousBatching)
        #expect(recording.plan.selectedStrategy == MLXGenerationExecutionStrategy.scalar)
        #expect(recording.plan.reason == reason)
        #expect(recording.admissionSnapshot.maxConcurrentRequests == 1)
        #expect(recording.executionSnapshot.reason == reason)
        #expect(recording.executionSnapshot.effectiveMaxBatchSize == 1)
        #expect(await session.continuousBatchEngine == nil)
    }

    @Test(
        "continuous-capable session streams through the live batched engine",
        .disabled(
            if: ProcessInfo.processInfo.environment["MLX_RUN_ARRAY_UNIT_TESTS"] != "1",
            "Set MLX_RUN_ARRAY_UNIT_TESTS=1 on a launcher with MLX Metal resources available"
        )
    )
    func continuousCapableSessionStreamsThroughLiveBatchedEngine() async throws {
        let session = Self.continuousBatchSession()
        try await session.applyRuntimeConfiguration()

        let recorded = try await MLXGenerationDiagnostics.withRecording {
            async let first = Self.chunks(from: await session.stream(Self.input()))
            async let second = Self.chunks(from: await session.stream(Self.input()))
            return try await (first, second)
        }

        #expect(recorded.result.0.contains { $0.text == "A" })
        #expect(recorded.result.0.contains(where: Self.isFinished))
        #expect(recorded.result.1.contains(where: Self.isFinished))
        #expect(Self.continuousBatchSnapshots(from: recorded.events).contains { snapshot in
            snapshot.rowCount == 2
        })
    }

    private struct RuntimeConfigurationRecording {
        let admissionSnapshot: MLXGenerationAdmissionController.Snapshot
        let executionSnapshot: MLXGenerationExecutionPlanSnapshot
        let plan: MLXGenerationExecutionPlan
    }

    private static func continuousBatchSession() -> MLXSession {
        MLXSession(
            configuration: ProviderConfiguration(
                location: URL(fileURLWithPath: "/tmp/continuous-stream-test-model"),
                modelName: "continuous-stream-test-model",
                runtime: ModelRuntimePreferences(
                    scheduling: .init(
                        mode: .continuousBatching,
                        maxConcurrentRequests: 2,
                        maxQueuedRequests: 4,
                        maxBatchSize: 2
                    )
                )
            ),
            modelContainer: ModelContainer(context: modelContext()),
            runtimeCapabilities: .continuousBatching
        )
    }

    private static func speculativeContinuousBatchSession() -> MLXSession {
        MLXSession(
            configuration: ProviderConfiguration(
                location: URL(fileURLWithPath: "/tmp/speculative-scheduling-test-model"),
                modelName: "speculative-scheduling-test-model",
                runtime: ModelRuntimePreferences(
                    speculativeDecodingMode: .sameModelDraft,
                    scheduling: .init(
                        mode: .continuousBatching,
                        maxConcurrentRequests: 5,
                        maxQueuedRequests: 13,
                        maxBatchSize: 4
                    )
                )
            ),
            modelContainer: ModelContainer(context: Self.modelContext()),
            runtimeCapabilities: .continuousBatching
        )
    }

    private static func nativeMTPContinuousBatchSession() -> MLXSession {
        MLXSession(
            configuration: ProviderConfiguration(
                location: URL(fileURLWithPath: "/tmp/native-mtp-scheduling-test-model"),
                modelName: "native-mtp-scheduling-test-model",
                runtime: ModelRuntimePreferences(
                    scheduling: .init(
                        mode: .continuousBatching,
                        maxConcurrentRequests: 5,
                        maxQueuedRequests: 13,
                        maxBatchSize: 4
                    ),
                    optimization: .nativeMTP()
                )
            ),
            modelContainer: ModelContainer(context: Self.modelContext()),
            runtimeCapabilities: .continuousBatching
        )
    }

    private static func modelContext() -> ModelContext {
        ModelContext(
            configuration: ModelConfiguration(
                id: "test/continuous-batch",
                eosTokenIds: [99]
            ),
            model: MLXEchoBatchLanguageModel(),
            tokenizer: PreparedGenerationTokenizer()
        )
    }

    private static func input() -> LLMInput {
        LLMInput(
            context: "hello world",
            promptMetadata: PromptRenderMetadata(rendererID: "continuous-batch-test"),
            promptCacheIdentity: PromptCacheIdentity(stableFingerprint: "continuous-batch-test"),
            sampling: .deterministic,
            limits: ResourceLimits(maxTokens: 2, reusePromptCache: false)
        )
    }

    private static func chunks(
        from stream: AsyncThrowingStream<LLMStreamChunk, any Error>
    ) async throws -> [LLMStreamChunk] {
        var chunks: [LLMStreamChunk] = []
        for try await chunk in stream {
            chunks.append(chunk)
        }
        return chunks
    }

    private static func isFinished(_ chunk: LLMStreamChunk) -> Bool {
        if case .finished = chunk.event {
            return true
        }
        return false
    }

    private static func recordRuntimeConfiguration(
        for session: MLXSession
    ) async throws -> RuntimeConfigurationRecording {
        let recorded = try await MLXGenerationDiagnostics.withRecording {
            try await session.applyRuntimeConfiguration()
            return (
                plan: await session.generationExecutionPlan,
                snapshot: await session.generationAdmission.snapshot()
            )
        }
        return RuntimeConfigurationRecording(
            admissionSnapshot: recorded.result.snapshot,
            executionSnapshot: try #require(Self.executionPlanSnapshots(from: recorded.events).last),
            plan: try #require(recorded.result.plan)
        )
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

    private static func continuousBatchSnapshots(
        from events: [MLXGenerationDiagnosticEvent]
    ) -> [MLXContinuousBatchLogitsSnapshot] {
        events.compactMap { event in
            guard case .continuousBatchLogits(let snapshot) = event else {
                return nil
            }
            return snapshot
        }
    }
}

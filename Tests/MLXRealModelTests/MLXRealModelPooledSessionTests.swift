import Foundation
import MLXFoundationModel
@testable import MLXLocalModels
import Testing

@Suite(
    "MLX real-model pooled session",
    .serialized,
    .disabled(
        if: !MLXRealModelEnvironment.isEnabled,
        "Set MLX_RUN_REAL_MODEL_TESTS=1 and download models with scripts/download-test-models.sh"
    )
)
struct MLXRealModelPooledSessionTests {
    @Test("Qwen3 pooled session streams real tokens and unloads cold residency")
    func qwen3PooledSessionStreamsRealTokensAndUnloadsColdResidency() async throws {
        let models = try MLXRealModelCatalog.load()
        guard let model = try MLXRealModelHarness.selectedModel("qwen3-0.6b-4bit", in: models) else {
            return
        }
        let pool = MLXModelPool(configuration: .init(maxResidentModels: 1))
        let session = MLXPooledSession(model: Self.languageModel(for: model), pool: pool)

        let observed = try await MLXGenerationDiagnostics.withRecording {
            try await Self.preload(session, model: model)
            return try await Self.collectGeneration(from: await session.stream(Self.input))
        }
        let snapshot = await pool.snapshot()
        let tokenEvents = MLXRealModelHarness.generatedTokenSnapshots(from: observed.events)

        MLXRealModelHarness.verifyGenerated(observed.result)
        MLXRealModelHarness.verifyGeneratedTokenDiagnostics(tokenEvents, result: observed.result)
        #expect(snapshot.residentModelIDs.isEmpty)
    }

    private static var input: LLMInput {
        LLMInput(
            context: "/no_think\nReply with one short word about local MLX sessions.",
            sampling: .deterministic,
            limits: ResourceLimits(maxTokens: 4, maxTime: .seconds(120), reusePromptCache: false)
        )
    }

    private static func languageModel(
        for model: MLXRealModelCatalog.Model
    ) -> MLXLanguageModel {
        MLXLanguageModel(
            model: MLXModel(
                id: model.displayName,
                location: MLXRealModelEnvironment.modelURL(for: model)
            ),
            compute: .small,
            runtime: ModelRuntimePreferences(
                residencyPreference: .cold,
                promptCachePolicy: .memory
            ),
            sampling: .deterministic,
            maximumResponseTokens: model.maxTokens
        )
    }

    private static func preload(
        _ session: MLXPooledSession,
        model: MLXRealModelCatalog.Model
    ) async throws {
        let progress = await session.preload(configuration: providerConfiguration(for: model))
        for try await _ in progress {
            // Drain preload progress before streaming.
        }
    }

    private static func providerConfiguration(
        for model: MLXRealModelCatalog.Model
    ) -> ProviderConfiguration {
        ProviderConfiguration(
            location: MLXRealModelEnvironment.modelURL(for: model),
            authentication: .noAuth,
            modelName: model.displayName,
            compute: .small,
            runtime: ModelRuntimePreferences(
                residencyPreference: .cold,
                promptCachePolicy: .memory
            )
        )
    }

    private static func collectGeneration(
        from stream: AsyncThrowingStream<LLMStreamChunk, any Error>
    ) async throws -> MLXRealModelHarness.GenerationResult {
        var text = ""
        var textChunkCount = 0
        var metrics: ChunkMetrics?
        for try await chunk in stream {
            if case .text = chunk.event {
                text += chunk.text
                textChunkCount += chunk.text.isEmpty ? 0 : 1
            }
            metrics = chunk.metrics ?? metrics
        }
        return .init(text: text, textChunkCount: textChunkCount, metrics: metrics)
    }
}

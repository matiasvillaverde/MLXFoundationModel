#if FOUNDATION_MODELS_PROVIDER_API && canImport(FoundationModels)
import Foundation
import FoundationModels
import MLXFoundationModel
import MLXLocalModels
import Testing

@Suite(
    "MLX real-model Foundation Models provider",
    .serialized,
    .disabled(
        if: !MLXRealModelEnvironment.isEnabled,
        "Set MLX_RUN_REAL_MODEL_TESTS=1 and download models with scripts/download-test-models.sh"
    )
)
struct MLXRealModelFoundationModelsProviderTests {
    @Test("LanguageModelSession streams text through a real MLX model")
    func languageModelSessionStreamsTextThroughRealMLXModel() async throws {
        guard #available(macOS 27.0, iOS 27.0, visionOS 27.0, *) else {
            return
        }

        let models = try MLXRealModelCatalog.load()
        guard let model = try MLXRealModelHarness.selectedModel("qwen3-0.6b-4bit", in: models) else {
            return
        }
        let languageModel = try MLXLanguageModel(
            id: model.id,
            location: MLXRealModelEnvironment.modelURL(for: model),
            compute: .small,
            runtime: ModelRuntimePreferences(promptCachePolicy: .memory),
            sampling: .deterministic,
            maximumResponseTokens: 4
        )
        let session = LanguageModelSession(
            model: languageModel,
            instructions: "Answer in short plain text."
        )
        let stream = session.streamResponse(
            to: "/no_think\nWrite one word about local inference.",
            options: GenerationOptions(
                samplingMode: .greedy,
                maximumResponseTokens: 4
            ),
            contextOptions: ContextOptions()
        )

        var snapshotCount = 0
        var finalText = ""
        var finalUsage: LanguageModelSession.Usage?
        for try await snapshot in stream {
            snapshotCount += 1
            finalText = snapshot.content
            finalUsage = snapshot.usage
        }

        #expect(snapshotCount > 0)
        #expect(!finalText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        let usage = try #require(finalUsage)
        #expect(usage.output.totalTokenCount > 0)
    }
}
#endif

import MLXFoundationModel
import MLXLocalModels
import Testing

@Suite(
    "MLX real-model Foundation Models-style interface",
    .serialized,
    .disabled(
        if: !MLXRealModelEnvironment.isEnabled,
        "Set MLX_RUN_REAL_MODEL_TESTS=1 and download models with scripts/download-test-models.sh"
    )
)
struct MLXRealModelInterfaceTests {
    @Test("selected models run rendered session-style requests")
    func selectedModelsRunRenderedSessionStyleRequests() async throws {
        let models = try MLXRealModelCatalog.load()
        let selected = MLXRealModelEnvironment
            .selectedModels(from: models)
            .filter { !$0.tags.contains("native-template-only") }
        let missing = selected.filter { !MLXRealModelEnvironment.hasModelFiles(for: $0) }

        #expect(!selected.isEmpty)
        #expect(
            missing.isEmpty,
            Comment(rawValue: MLXRealModelEnvironment.missingModelsMessage(missing))
        )
        guard missing.isEmpty else {
            return
        }

        var failures: [String] = []
        for model in selected {
            do {
                try await Self.verifySessionStyleRequest(on: model)
            } catch {
                failures.append("\(model.id): \(error)")
            }
        }
        #expect(failures.isEmpty, Comment(rawValue: failures.joined(separator: "\n")))
    }

    @Test("Qwen3 streams multiple chunks from a rendered text request")
    func qwen3StreamsMultipleChunksFromRenderedTextRequest() async throws {
        let models = try MLXRealModelCatalog.load()
        guard let model = try MLXRealModelHarness.selectedModel("qwen3-0.6b-4bit", in: models) else {
            return
        }
        let result = try await MLXRealModelHarness.runRenderedRequest(
            model: model,
            request: Self.textStreamingRequest,
            limits: ResourceLimits(maxTokens: 24, maxTime: .seconds(120), reusePromptCache: false),
            style: .chatML
        )

        MLXRealModelHarness.verifyGenerated(result)
        #expect(result.textChunkCount > 1)
    }

    private static var sessionStyleRequest: MLXBridgeRequest {
        MLXBridgeRequest(
            messages: [
                MLXBridgeMessage(
                    role: .user,
                    content: "/no_think\nProject: MLXFoundationModel. Reply briefly."
                )
            ],
            instructions: "Answer in short plain text."
        )
    }

    private static func verifySessionStyleRequest(
        on model: MLXRealModelCatalog.Model
    ) async throws {
        let result = try await MLXRealModelHarness.runRenderedRequest(
            model: model,
            request: Self.sessionStyleRequest,
            limits: ResourceLimits(
                maxTokens: min(8, MLXRealModelEnvironment.architectureGenerationTokenLimit),
                maxTime: .seconds(MLXRealModelEnvironment.architectureGenerationTimeoutSeconds),
                reusePromptCache: false
            )
        )
        MLXRealModelHarness.verifyGenerated(result)
    }

    private static var textStreamingRequest: MLXBridgeRequest {
        MLXBridgeRequest(
            messages: [
                MLXBridgeMessage(
                    role: .user,
                    content: "/no_think\nWrite two short clauses about local model adapters."
                )
            ],
            instructions: "Answer in plain text. Do not include Markdown."
        )
    }
}

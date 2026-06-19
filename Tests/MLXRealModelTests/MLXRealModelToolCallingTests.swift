import MLXFoundationModel
@testable import MLXLocalModels
import Testing

@Suite(
    "MLX real-model tool calling",
    .serialized,
    .disabled(
        if: !MLXRealModelEnvironment.isEnabled,
        "Set MLX_RUN_REAL_MODEL_TESTS=1 and download models with scripts/download-test-models.sh"
    )
)
struct MLXRealModelToolCallingTests {
    @Test("Qwen3 emits a parseable tool call when tools are rendered")
    func qwen3EmitsParseableToolCall() async throws {
        let models = try MLXRealModelCatalog.load()
        let model = try MLXRealModelHarness.requireModel("qwen3-0.6b-4bit", in: models)
        let rendered = MLXPromptRenderer.render(Self.toolRequest, style: .chatML)
        let observed = try await MLXGenerationDiagnostics.withRecording {
            try await MLXRealModelHarness.run(
                model: model,
                prompt: rendered.prompt,
                sampling: .deterministic,
                limits: ResourceLimits(maxTokens: 160, maxTime: .seconds(120), reusePromptCache: false)
            )
        }
        let result = observed.result
        let tokenEvents = MLXRealModelHarness.generatedTokenSnapshots(from: observed.events)

        MLXRealModelHarness.verifyGenerated(result)
        MLXRealModelHarness.verifyGeneratedTokenDiagnostics(tokenEvents, result: result)
        let call = try #require(MLXToolCallExtractor.extract(from: result.text))
        #expect(call.name == "weather")
        #expect(call.argumentsJSON.contains("Berlin"))
    }

    private static var toolRequest: MLXBridgeRequest {
        MLXBridgeRequest(
            messages: [
                MLXBridgeMessage(role: .user, content: "/no_think\nCall the weather tool for Berlin.")
            ],
            instructions: """
            You are a tool router. Do not think aloud.
            Return exactly one JSON object and no prose:
            {"tool_name":"weather","arguments":{"city":"Berlin"}}
            """,
            tools: [weatherTool]
        )
    }

    private static var weatherTool: MLXBridgeToolDefinition {
        MLXBridgeToolDefinition(
            name: "weather",
            description: "Read local weather for a city",
            parametersJSONSchema: """
            {"type":"object","required":["city"],"properties":{"city":{"type":"string"}}}
            """
        )
    }
}

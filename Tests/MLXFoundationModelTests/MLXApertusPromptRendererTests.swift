import Foundation
import MLXFoundationModel
import Testing

@Suite("MLX Apertus prompt renderer")
struct MLXApertusPromptRendererTests {
    @Test("renders native role markers")
    func rendersNativeRoleMarkers() {
        let rendered = MLXPromptRenderer.render(Self.requestWithInstructions, style: .apertus)

        #expect(rendered.rendererID == "mlx.apertus.v1")
        #expect(rendered.prompt.hasPrefix("<s><|system_start|>"))
        #expect(rendered.prompt.contains("Available tools:"))
        #expect(rendered.prompt.contains("<|developer_start|>Deliberation: disabled"))
        #expect(rendered.prompt.contains("<|developer_end|><|user_start|>What is the weather?<|user_end|>"))
        #expect(rendered.prompt.hasSuffix("<|assistant_start|>"))
    }

    private static var requestWithInstructions: MLXBridgeRequest {
        MLXBridgeRequest(
            messages: [
                MLXBridgeMessage(role: .user, content: "What is the weather?")
            ],
            instructions: "Be concise.",
            tools: [weatherTool]
        )
    }

    private static var weatherTool: MLXBridgeToolDefinition {
        MLXBridgeToolDefinition(
            name: "weather",
            description: "Read local weather",
            parametersJSONSchema: """
            {"type":"object","properties":{"city":{"type":"string"},"count":{"type":"integer"}}}
            """
        )
    }
}

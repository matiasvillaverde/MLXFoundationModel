import Foundation
import MLXFoundationModel
import Testing

@Suite("MLX Harmony prompt renderer")
struct MLXHarmonyPromptRendererTests {
    @Test("renders native tool contract")
    func rendersNativeToolContract() {
        let rendered = MLXPromptRenderer.render(Self.request, style: .harmony)

        #expect(rendered.prompt.contains("You may call tools only from the commentary channel."))
        #expect(rendered.prompt.contains("functions.weather"))
        #expect(rendered.prompt.contains("<|start|>assistant to=functions.weather<|channel|>commentary"))
        #expect(rendered.prompt.contains("<|message|>{\"city\":\"value\",\"count\":1}<|call|>"))
        #expect(rendered.prompt.contains(#""type":"object""#))
        #expect(!rendered.prompt.contains("Available tools:"))
    }

    private static var request: MLXBridgeRequest {
        MLXBridgeRequest(
            messages: [
                MLXBridgeMessage(role: .user, content: "What is the weather?")
            ],
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

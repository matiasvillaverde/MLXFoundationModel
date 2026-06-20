import Foundation
import MLXFoundationModel
import Testing

@Suite("MLX FunctionGemma prompt renderer")
struct MLXFunctionGemmaPromptRendererTests {
    @Test("replays tool-call history with legacy markers")
    func replaysToolCallHistoryWithLegacyMarkers() {
        let request = MLXBridgeRequest(
            messages: [
                MLXBridgeMessage(
                    role: .assistant,
                    content: #"{"tool_name":"weather","arguments":{"city":"Berlin"}}"#
                )
            ],
            tools: [Self.weatherTool]
        )

        let rendered = MLXPromptRenderer.render(request, style: .functionGemma)

        #expect(rendered.prompt.contains("<start_function_call>"))
        #expect(rendered.prompt.contains("call:weather{city:Berlin}"))
        #expect(rendered.prompt.contains("<end_function_call>"))
        #expect(!rendered.prompt.contains("<|tool_call>"))
    }

    private static var weatherTool: MLXBridgeToolDefinition {
        MLXBridgeToolDefinition(
            name: "weather",
            description: "Read local weather",
            parametersJSONSchema: #"{"type":"object","properties":{"city":{"type":"string"}}}"#
        )
    }
}

import Foundation
import MLXFoundationModel
import Testing

@Suite("MLX Qwen prompt renderer")
struct MLXQwenPromptRendererTests {
    @Test("renders tool-call and tool-result history")
    func rendersToolCallAndToolResultHistory() {
        let request = MLXBridgeRequest(
            messages: [
                MLXBridgeMessage(
                    role: .assistant,
                    content: #"{"tool_name":"weather","arguments":{"city":"Berlin"}}"#
                ),
                MLXBridgeMessage(role: .tool, content: #"{"temperature":18}"#, name: "weather")
            ],
            tools: [Self.weatherTool]
        )

        let rendered = MLXPromptRenderer.render(request, style: .qwenXML)

        #expect(rendered.prompt.contains(#"<tool_call><function=weather>"#))
        #expect(rendered.prompt.contains(#"<parameter=city>Berlin</parameter>"#))
        #expect(rendered.prompt.contains("<|im_start|>user\n<tool_response>"))
        #expect(rendered.prompt.contains(#"{"temperature":18}"#))
        #expect(rendered.prompt.contains("</tool_response><|im_end|>"))
        #expect(!rendered.prompt.contains("<|im_start|>tool"))
        #expect(rendered.prompt.hasSuffix("<|im_start|>assistant\n"))
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

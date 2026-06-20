import Foundation
import MLXFoundationModel
import Testing

@Suite("MLX GLM prompt renderer")
struct MLXGLMPromptRendererTests {
    @Test("replays tool-call and observation history with native GLM markers")
    func replaysToolCallAndObservationHistoryWithNativeGLMMarkers() {
        let rendered = MLXPromptRenderer.render(Self.toolHistoryRequest, style: .glmXML)

        #expect(rendered.prompt.hasPrefix("[gMASK]<sop><|system|>"))
        #expect(rendered.prompt.contains(#"<tool_call>weather<arg_key>city</arg_key>"#))
        #expect(rendered.prompt.contains(#"<arg_value>Berlin</arg_value>"#))
        #expect(rendered.prompt.contains(#"<arg_key>count</arg_key><arg_value>2</arg_value>"#))
        #expect(rendered.prompt.contains(#"<|observation|>"#))
        #expect(rendered.prompt.contains(#"{"temperature":18}"#))
        #expect(rendered.prompt.hasSuffix("<|assistant|>"))
        #expect(!rendered.prompt.contains("<|im_start|>"))
        #expect(!rendered.prompt.contains("Available tools:"))
    }

    @Test("renders GLM tool instructions with typed XML arguments")
    func rendersGLMToolInstructionsWithTypedXMLArguments() {
        let rendered = MLXPromptRenderer.render(Self.userRequest, style: .glmXML)

        #expect(rendered.prompt.contains("To call a tool, emit only GLM XML:"))
        #expect(rendered.prompt.contains(#"<tool_call>weather"#))
        #expect(rendered.prompt.contains(#"<arg_key>city</arg_key><arg_value>value</arg_value>"#))
        #expect(rendered.prompt.contains(#"<arg_key>count</arg_key><arg_value>1</arg_value>"#))
        #expect(rendered.prompt.contains(#""type":"function""#))
        #expect(rendered.prompt.contains(#"<|user|>"#))
        #expect(rendered.prompt.hasSuffix("<|assistant|>"))
    }

    private static var toolHistoryRequest: MLXBridgeRequest {
        MLXBridgeRequest(
            messages: [
                MLXBridgeMessage(
                    role: .assistant,
                    content: #"{"tool_name":"weather","arguments":{"city":"Berlin","count":2}}"#
                ),
                MLXBridgeMessage(role: .tool, content: #"{"temperature":18}"#, name: "weather")
            ],
            instructions: "Use tools when needed.",
            tools: [weatherTool]
        )
    }

    private static var userRequest: MLXBridgeRequest {
        MLXBridgeRequest(
            messages: [
                MLXBridgeMessage(role: .user, content: "Weather for Berlin?")
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

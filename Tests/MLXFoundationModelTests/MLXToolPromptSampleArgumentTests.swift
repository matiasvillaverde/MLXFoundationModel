import Foundation
import MLXFoundationModel
import Testing

@Suite("MLX tool prompt sample arguments")
struct MLXToolPromptSampleArgumentTests {
    @Test("renders schema-aware native examples for composed schemas")
    func rendersSchemaAwareNativeExamplesForComposedSchemas() {
        let rendered = MLXPromptRenderer.render(Self.request, style: .qwenXML)

        #expect(rendered.prompt.contains("<tool_call><function=weather>"))
        #expect(rendered.prompt.contains("<parameter=city>value</parameter>"))
        #expect(rendered.prompt.contains("<parameter=count>1</parameter>"))
        #expect(rendered.prompt.contains("<parameter=flags>[true,\"value\"]</parameter>"))
        #expect(rendered.prompt.contains(
            #"<parameter=payload>{"limit":7,"unit":"metric"}</parameter>"#
        ))
    }

    @Test("renders nested native XML examples for MiniMax M3")
    func rendersNestedNativeXMLExamplesForMiniMaxM3() {
        let rendered = MLXPromptRenderer.render(Self.request, style: .minimaxM3)

        #expect(rendered.prompt.contains(#"]<]minimax[>[<invoke name="weather">"#))
        #expect(rendered.prompt.contains(#"]<]minimax[>[<count>1]<]minimax[>[</count>"#))
        #expect(rendered.prompt.contains(#"]<]minimax[>[<limit>7]<]minimax[>[</limit>"#))
        #expect(rendered.prompt.contains(#"]<]minimax[>[<unit>metric]<]minimax[>[</unit>"#))
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
            description: "Read weather",
            parametersJSONSchema: Self.composedSchema
        )
    }

    private static var composedSchema: String {
        [
            #"{"type":"object","$defs":{"#,
            #""Payload":{"type":"object","properties":{"#,
            #""limit":{"const":7},"unit":{"enum":["metric","imperial"]}}}},"#,
            #""properties":{"city":{"type":"string"},"#,
            ##""payload":{"allOf":[{"$ref":"#/$defs/Payload"}]}},"##,
            #""allOf":[{"properties":{"count":{"type":"integer"},"#,
            #""flags":{"type":"array","prefixItems":[{"type":"boolean"},{"type":"string"}]}}}]}"#
        ].joined()
    }
}

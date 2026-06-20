import Foundation
import MLXFoundationModel
import Testing

@Suite("MLX Kimi prompt renderer")
struct MLXKimiPromptRendererTests {
    @Test("groups multiple assistant tool calls into one native section")
    func groupsMultipleAssistantToolCallsIntoOneNativeSection() {
        let request = MLXBridgeRequest(
            messages: [
                MLXBridgeMessage(role: .assistant, content: Self.assistantToolCalls)
            ],
            tools: [
                Self.tool(named: "weather", property: "city"),
                Self.tool(named: "search", property: "query")
            ]
        )

        let prompt = MLXPromptRenderer.render(request, style: .kimiK2).prompt

        #expect(Self.occurrences(of: "<|tool_calls_section_begin|>", in: prompt) == 1)
        #expect(Self.occurrences(of: "<|tool_calls_section_end|>", in: prompt) == 1)
        #expect(prompt.contains(#"<|tool_call_begin|>functions.weather:0"#))
        #expect(prompt.contains(#"<|tool_call_begin|>functions.search:1"#))
        #expect(prompt.contains(#"<|tool_call_argument_begin|>{"city":"Berlin"}"#))
        #expect(prompt.contains(#"<|tool_call_argument_begin|>{"query":"MLX"}"#))
    }

    private static let assistantToolCalls = """
    {"tool_calls":[\
    {"function":{"name":"weather","arguments":{"city":"Berlin"}}},\
    {"function":{"name":"search","arguments":{"query":"MLX"}}}\
    ]}
    """

    private static func tool(named name: String, property: String) -> MLXBridgeToolDefinition {
        MLXBridgeToolDefinition(
            name: name,
            description: "\(name) tool",
            parametersJSONSchema: #"{"type":"object","properties":{"\#(property)":{"type":"string"}}}"#
        )
    }

    private static func occurrences(of marker: String, in text: String) -> Int {
        text.components(separatedBy: marker).count - 1
    }
}

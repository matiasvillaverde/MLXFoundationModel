import Foundation
import MLXFoundationModel
import Testing

@Suite("MLX thinking tool call extractor")
struct MLXThinkingToolCallExtractorTests {
    @Test("prefers regular tool calls over thinking tool calls")
    func prefersRegularToolCallsOverThinkingToolCalls() {
        let calls = MLXToolCallExtractor.extractAll(
            from: """
            <think><tool_call>{"name":"weather","arguments":{"city":"Berlin"}}</tool_call></think>\
            <tool_call>{"name":"search","arguments":{"query":"MLX"}}</tool_call>
            """,
            tools: [Self.weatherTool, Self.searchTool]
        )

        #expect(calls.map(\.name) == ["search"])
        #expect(calls.map(\.argumentsJSON) == [#"{"query":"MLX"}"#])
    }

    @Test("promotes valid tool calls from thinking content")
    func promotesValidToolCallsFromThinkingContent() throws {
        let call = try #require(MLXToolCallExtractor.extract(
            from: """
            <think>I should call a tool. \
            <tool_call>{"name":"weather","arguments":{"city":"Berlin"}}</tool_call></think>\
            I will check.
            """,
            tools: [Self.weatherTool]
        ))

        #expect(call.name == "weather")
        #expect(call.argumentsJSON == #"{"city":"Berlin"}"#)
    }

    @Test("promotes valid tool calls from LongCat thinking content")
    func promotesValidToolCallsFromLongCatThinkingContent() throws {
        let call = try #require(MLXToolCallExtractor.extract(
            from: """
            <longcat_think>I should call a tool. \
            <longcat_tool_call>{"name":"weather","arguments":{"city":"Berlin"}}</longcat_tool_call>\
            </longcat_think>I will check.
            """,
            tools: [Self.weatherTool]
        ))

        #expect(call.name == "weather")
        #expect(call.argumentsJSON == #"{"city":"Berlin"}"#)
    }

    @Test("drops unknown tool calls from thinking content")
    func dropsUnknownToolCallsFromThinkingContent() {
        let calls = MLXToolCallExtractor.extractAll(
            from: """
            <think><tool_call>{"name":"weather","arguments":{"city":"Berlin"}}</tool_call></think>\
            I can answer directly.
            """,
            tools: [Self.searchTool]
        )

        #expect(calls.isEmpty)
    }

    private static var weatherTool: MLXBridgeToolDefinition {
        MLXBridgeToolDefinition(
            name: "weather",
            description: "Read weather",
            parametersJSONSchema: #"{"type":"object","properties":{"city":{"type":"string"}}}"#
        )
    }

    private static var searchTool: MLXBridgeToolDefinition {
        MLXBridgeToolDefinition(
            name: "search",
            description: "Search",
            parametersJSONSchema: #"{"type":"object","properties":{"query":{"type":"string"}}}"#
        )
    }
}

import Foundation
import MLXFoundationModel
import Testing

@Suite("MLX upstream tool call extractor")
struct MLXUpstreamToolCallExtractorTests {
    @Test("extracts namespaced Qwen function XML tool call")
    func extractsNamespacedQwenFunctionXMLToolCall() throws {
        let call = try #require(MLXToolCallExtractor.extract(
            from: """
            <tool_call>\
            <function=functions.weather:current><parameter=city>"Berlin"</parameter></function>\
            </tool_call>
            """
        ))

        #expect(call.name == "functions.weather:current")
        #expect(call.argumentsJSON == #"{"city":"Berlin"}"#)
    }

    @Test("extracts Kimi K2 section tool calls")
    func extractsKimiK2SectionToolCalls() {
        let calls = MLXToolCallExtractor.extractAll(
            from: """
            <|tool_calls_section_begin|>\
            <|tool_call_begin|>functions.weather:0<|tool_call_argument_begin|>{"city":"Berlin"}\
            <|tool_call_end|>\
            <|tool_call_begin|>functions.search:1<|tool_call_argument_begin|>{"query":"MLX"}\
            <|tool_call_end|>\
            <|tool_calls_section_end|>
            """
        )

        #expect(calls.map(\.name) == ["weather", "search"])
        #expect(calls.map(\.argumentsJSON) == [
            #"{"city":"Berlin"}"#,
            #"{"query":"MLX"}"#
        ])
    }

    @Test("extracts LongCat tool call")
    func extractsLongCatToolCall() throws {
        let call = try #require(MLXToolCallExtractor.extract(
            from: """
            <longcat_tool_call>weather\
            <longcat_arg_key>city</longcat_arg_key>\
            <longcat_arg_value>"Berlin"</longcat_arg_value>\
            <longcat_arg_key>count</longcat_arg_key>\
            <longcat_arg_value>2</longcat_arg_value>\
            </longcat_tool_call>
            """
        ))
        let arguments = try Self.jsonObject(from: call.argumentsJSON)

        #expect(call.name == "weather")
        #expect(arguments["city"] as? String == "Berlin")
        #expect(arguments["count"] as? Int == 2)
    }

    @Test("extracts LongCat JSON tool call")
    func extractsLongCatJSONToolCall() throws {
        let call = try #require(MLXToolCallExtractor.extract(
            from: """
            <longcat_tool_call>\
            {"name":"weather","arguments":{"city":"Berlin","count":2}}\
            </longcat_tool_call>
            """
        ))
        let arguments = try Self.jsonObject(from: call.argumentsJSON)

        #expect(call.name == "weather")
        #expect(arguments["city"] as? String == "Berlin")
        #expect(arguments["count"] as? Int == 2)
    }

    @Test("extracts Cohere action tool call")
    func extractsCohereActionToolCall() throws {
        let call = try #require(MLXToolCallExtractor.extract(
            from: """
            <|START_ACTION|>\
            {"tool_name":"weather","parameters":{"city":"Berlin","count":2}}\
            <|END_ACTION|>
            """
        ))
        let arguments = try Self.jsonObject(from: call.argumentsJSON)

        #expect(call.name == "weather")
        #expect(arguments["city"] as? String == "Berlin")
        #expect(arguments["count"] as? Int == 2)
    }

    @Test("extracts Mistral args marker tool call")
    func extractsMistralArgumentsMarkerToolCall() throws {
        let call = try #require(MLXToolCallExtractor.extract(
            from: #"[TOOL_CALLS]weather[ARGS]{"city":"Berlin","count":2}"#
        ))
        let arguments = try Self.jsonObject(from: call.argumentsJSON)

        #expect(call.name == "weather")
        #expect(arguments["city"] as? String == "Berlin")
        #expect(arguments["count"] as? Int == 2)
    }

    @Test("extracts FunctionGemma legacy tool call")
    func extractsFunctionGemmaLegacyToolCall() throws {
        let call = try #require(MLXToolCallExtractor.extract(
            from: """
            <start_function_call>\
            call:weather{city:<escape>Berlin<escape>,count:2}\
            <end_function_call>
            """
        ))
        let arguments = try Self.jsonObject(from: call.argumentsJSON)

        #expect(call.name == "weather")
        #expect(arguments["city"] as? String == "Berlin")
        #expect(arguments["count"] as? Int == 2)
    }

    private static func jsonObject(from text: String) throws -> [String: Any] {
        let data = try #require(text.data(using: .utf8))
        return try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
    }
}

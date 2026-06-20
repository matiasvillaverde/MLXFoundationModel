import Foundation
import MLXFoundationModel
import Testing

@Suite("MLX tool call extractor")
struct MLXToolCallExtractorTests {
    @Test("extracts flat JSON tool call")
    func extractsFlatJSONToolCall() throws {
        let call = try #require(MLXToolCallExtractor.extract(
            from: #"{"tool_name":"weather","arguments":{"city":"Berlin"}}"#
        ))

        #expect(call.name == "weather")
        #expect(call.argumentsJSON == #"{"city":"Berlin"}"#)
    }

    @Test("extracts fenced JSON tool call")
    func extractsFencedJSONToolCall() throws {
        let call = try #require(MLXToolCallExtractor.extract(
            from: """
            ```json
            {"name":"search","arguments":{"query":"MLX Swift"}}
            ```
            """
        ))

        #expect(call.name == "search")
        #expect(call.argumentsJSON == #"{"query":"MLX Swift"}"#)
    }

    @Test("extracts embedded JSON tool call")
    func extractsEmbeddedJSONToolCall() throws {
        let call = try #require(MLXToolCallExtractor.extract(
            from: """
            <think>
            The user needs current weather, so I should call a tool.
            </think>

            {"tool_name":"weather","arguments":{"city":"Berlin"}}
            """
        ))

        #expect(call.name == "weather")
        #expect(call.argumentsJSON == #"{"city":"Berlin"}"#)
    }

    @Test("extracts XML JSON tool call")
    func extractsXMLJSONToolCall() throws {
        let call = try #require(MLXToolCallExtractor.extract(
            from: #"<tool_call>{"name":"weather","arguments":{"city":"Berlin"}}</tool_call>"#
        ))

        #expect(call.name == "weather")
        #expect(call.argumentsJSON == #"{"city":"Berlin"}"#)
    }

    @Test("extracts Qwen function XML tool call")
    func extractsQwenFunctionXMLToolCall() throws {
        let call = try #require(MLXToolCallExtractor.extract(
            from: """
            <tool_call><function=weather><parameter=city>"Berlin"</parameter></function></tool_call>
            """
        ))

        #expect(call.name == "weather")
        #expect(call.argumentsJSON == #"{"city":"Berlin"}"#)
    }

    @Test("extracts GLM arg key XML tool call")
    func extractsGLMArgKeyXMLToolCall() throws {
        let call = try #require(MLXToolCallExtractor.extract(
            from: """
            <tool_call>weather<arg_key>city</arg_key><arg_value>"Berlin"</arg_value></tool_call>
            """
        ))

        #expect(call.name == "weather")
        #expect(call.argumentsJSON == #"{"city":"Berlin"}"#)
    }

    @Test("extracts bare GLM arg key XML tool call")
    func extractsBareGLMArgKeyXMLToolCall() throws {
        let call = try #require(MLXToolCallExtractor.extract(
            from: """
            weather<arg_key>city</arg_key><arg_value>"Berlin"</arg_value>\
            <arg_key>count</arg_key><arg_value>2</arg_value>
            """
        ))

        #expect(call.name == "weather")
        #expect(call.argumentsJSON == #"{"city":"Berlin","count":2}"#)
    }

    @Test("extracts namespaced invoke tool call")
    func extractsNamespacedInvokeToolCall() throws {
        let call = try #require(MLXToolCallExtractor.extract(
            from: """
            <minimax:tool_call><invoke name="weather">\
            <parameter name="city">"Berlin"</parameter></invoke></minimax:tool_call>
            """
        ))

        #expect(call.name == "weather")
        #expect(call.argumentsJSON == #"{"city":"Berlin"}"#)
    }

    @Test("extracts MiniMax M3 namespace-token tool call")
    func extractsMiniMaxM3NamespaceTokenToolCall() throws {
        let call = try #require(MLXToolCallExtractor.extract(
            from: """
            ]<]minimax[>[<tool_call>
            ]<]minimax[>[<invoke name="weather">\
            ]<]minimax[>[<city>Berlin]<]minimax[>[</city>\
            ]<]minimax[>[<count>2]<]minimax[>[</count>\
            ]<]minimax[>[<options>\
            ]<]minimax[>[<item>metric]<]minimax[>[</item>\
            ]<]minimax[>[<item>brief]<]minimax[>[</item>\
            ]<]minimax[>[</options>\
            ]<]minimax[>[</invoke>
            ]<]minimax[>[</tool_call>
            """
        ))
        let arguments = try Self.jsonObject(from: call.argumentsJSON)
        let options = try #require(arguments["options"] as? [String])

        #expect(call.name == "weather")
        #expect(arguments["city"] as? String == "Berlin")
        #expect(arguments["count"] as? Int == 2)
        #expect(options == ["metric", "brief"])
    }

    @Test("extracts Hermes JSON tool call")
    func extractsHermesJSONToolCall() throws {
        let call = try #require(MLXToolCallExtractor.extract(
            from: """
            <|tool_call_start|>{"name":"weather","arguments":{"city":"Berlin"}}<|tool_call_end|>
            """
        ))

        #expect(call.name == "weather")
        #expect(call.argumentsJSON == #"{"city":"Berlin"}"#)
    }

    @Test("extracts Harmony commentary tool call")
    func extractsHarmonyCommentaryToolCall() throws {
        let call = try #require(MLXToolCallExtractor.extract(
            from: """
            <|channel|>analysis<|message|>thinking<|end|>\
            <|start|>assistant to=functions.weather<|channel|>commentary\
            <|message|>{"city":"Berlin","count":2}<|call|>
            """
        ))

        #expect(call.name == "weather")
        #expect(call.argumentsJSON == #"{"city":"Berlin","count":2}"#)
    }

    @Test("extracts Harmony tool call with recipient in channel descriptor")
    func extractsHarmonyToolCallWithRecipientInChannelDescriptor() throws {
        let call = try #require(MLXToolCallExtractor.extract(
            from: #"<|channel|>commentary to=functions.weather<|message|>{"city":"Seoul"}<|call|>"#
        ))

        #expect(call.name == "weather")
        #expect(call.argumentsJSON == #"{"city":"Seoul"}"#)
    }

    @Test("extracts Mistral tool calls marker")
    func extractsMistralToolCallsMarker() throws {
        let call = try #require(MLXToolCallExtractor.extract(
            from: #"[TOOL_CALLS][{"name":"weather","arguments":{"city":"Berlin"}}]"#
        ))

        #expect(call.name == "weather")
        #expect(call.argumentsJSON == #"{"city":"Berlin"}"#)
    }

    @Test("extracts all Mistral tool calls")
    func extractsAllMistralToolCalls() {
        let calls = MLXToolCallExtractor.extractAll(
            from: """
            [TOOL_CALLS][\
            {"name":"weather","arguments":{"city":"Berlin"}},\
            {"name":"search","arguments":{"query":"MLX"}}\
            ]
            """
        )

        #expect(calls.map(\.name) == ["weather", "search"])
        #expect(calls.map(\.argumentsJSON) == [
            #"{"city":"Berlin"}"#,
            #"{"query":"MLX"}"#
        ])
    }

    @Test("extracts bracket tool call")
    func extractsBracketToolCall() throws {
        let call = try #require(MLXToolCallExtractor.extract(
            from: #"[Tool call: weather({"city":"Berlin"})]"#
        ))

        #expect(call.name == "weather")
        #expect(call.argumentsJSON == #"{"city":"Berlin"}"#)
    }

    @Test("extracts Gemma call object tool call")
    func extractsGemmaCallObjectToolCall() throws {
        let call = try #require(MLXToolCallExtractor.extract(
            from: #"<|tool_call>call:weather{"city":"Berlin"}<tool_call|>"#
        ))

        #expect(call.name == "weather")
        #expect(call.argumentsJSON == #"{"city":"Berlin"}"#)
    }

    @Test("extracts Gemma parenthesized tool call")
    func extractsGemmaParenthesizedToolCall() throws {
        let call = try #require(MLXToolCallExtractor.extract(
            from: #"<|tool_call>call:weather(city="Berlin")<tool_call|>"#
        ))

        #expect(call.name == "weather")
        #expect(call.argumentsJSON == #"{"city":"Berlin"}"#)
    }

    @Test("extracts all Gemma tool calls from one marker block")
    func extractsAllGemmaToolCallsFromOneMarkerBlock() {
        let calls = MLXToolCallExtractor.extractAll(
            from: #"<|tool_call>call:weather{"city":"Berlin"} call:search(query="MLX")<tool_call|>"#
        )

        #expect(calls.map(\.name) == ["weather", "search"])
        #expect(calls.map(\.argumentsJSON) == [
            #"{"city":"Berlin"}"#,
            #"{"query":"MLX"}"#
        ])
    }

    @Test("extracts Gemma single quoted arguments")
    func extractsGemmaSingleQuotedArguments() throws {
        let call = try #require(MLXToolCallExtractor.extract(
            from: #"<|tool_call>call:weather{'city':'Berlin'}<tool_call|>"#
        ))

        #expect(call.name == "weather")
        #expect(call.argumentsJSON == #"{"city":"Berlin"}"#)
    }

    @Test("extracts Hermes list payload with Python keyword arguments")
    func extractsHermesListPayloadWithPythonKeywordArguments() throws {
        let calls = MLXToolCallExtractor.extractAll(
            from: """
            <|tool_call_start|>\
            [execute_code(command='python3 a.py', timeout=400),\
            execute_code(command='python3 b.py', timeout=401)]\
            <|tool_call_end|>
            """
        )

        #expect(calls.map(\.name) == ["execute_code", "execute_code"])
        let first = try Self.jsonObject(from: try #require(calls.first?.argumentsJSON))
        let second = try Self.jsonObject(from: try #require(calls.last?.argumentsJSON))
        #expect(first["command"] as? String == "python3 a.py")
        #expect(first["timeout"] as? Int == 400)
        #expect(second["command"] as? String == "python3 b.py")
        #expect(second["timeout"] as? Int == 401)
    }

    @Test("extracts raw control characters from JSON tool arguments")
    func extractsRawControlCharactersFromJSONToolArguments() throws {
        let oldString = "\tindent\nnext line"
        let call = try #require(MLXToolCallExtractor.extract(
            from: """
            <tool_call>{"name":"edit","arguments":{"old_string":"\(oldString)","new_string":"x"}}</tool_call>
            """
        ))
        let arguments = try Self.jsonObject(from: call.argumentsJSON)

        #expect(call.name == "edit")
        #expect(arguments["old_string"] as? String == oldString)
        #expect(arguments["new_string"] as? String == "x")
    }

    @Test("ignores normal assistant prose")
    func ignoresNormalAssistantProse() {
        let call = MLXToolCallExtractor.extract(from: "I can answer without tools.")

        #expect(call == nil)
    }

    private static func jsonObject(from text: String) throws -> [String: Any] {
        let data = try #require(text.data(using: .utf8))
        return try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
    }
}

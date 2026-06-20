import Foundation
@testable import MLXFoundationModel
import Testing

@Suite("MLX DeepSeek DSML dialect")
struct MLXDeepSeekDSMLTests {
    @Test("renders native DeepSeek DSML prompt and tool declarations")
    func rendersNativeDeepSeekDSMLPromptAndToolDeclarations() {
        let rendered = MLXPromptRenderer.render(Self.request, style: .deepSeekDSML)

        #expect(rendered.rendererID == "mlx.deepSeekDSML.v1")
        #expect(rendered.prompt.contains("<｜begin▁of▁sentence｜>"))
        #expect(rendered.prompt.contains("<｜User｜>What is the weather?<｜Assistant｜>"))
        #expect(rendered.prompt.contains("## Tools"))
        #expect(rendered.prompt.contains(#""type":"function""#))
        #expect(!rendered.prompt.contains("Available tools:"))
        #expect(rendered.prompt.contains("<\(Self.dsmlToken)tool_calls>"))
    }

    @Test("replays DeepSeek DSML tool-call history with typed parameters")
    func replaysDeepSeekDSMLToolCallHistory() {
        let request = MLXBridgeRequest(
            messages: [
                MLXBridgeMessage(
                    role: .assistant,
                    content: #"{"tool_name":"weather","arguments":{"city":"Berlin","count":2}}"#
                )
            ],
            tools: [Self.weatherTool]
        )

        let rendered = MLXPromptRenderer.render(request, style: .deepSeekDSML)

        #expect(rendered.prompt.contains(#"<｜DSML｜invoke name="weather">"#))
        #expect(rendered.prompt.contains(#"<｜DSML｜parameter name="city" string="true">Berlin"#))
        #expect(rendered.prompt.contains(#"<｜DSML｜parameter name="count" string="false">2"#))
    }

    @Test("groups consecutive DeepSeek DSML tool results")
    func groupsConsecutiveDeepSeekDSMLToolResults() {
        let rendered = MLXPromptRenderer.render(
            Self.toolResultRequest(reasoningEnabled: true),
            style: .deepSeekDSML
        )

        #expect(Self.occurrences(of: "<function_results>", in: rendered.prompt) == 1)
        #expect(Self.occurrences(of: "</function_results>", in: rendered.prompt) == 1)
        #expect(rendered.prompt.contains("""
        <result>{"temperature":18}</result>
        <result>{"results":["MLX Swift"]}</result>
        """))
        #expect(rendered.prompt.hasSuffix("<think>\n"))
        #expect(!rendered.prompt.hasSuffix("<｜Assistant｜><think>\n"))
    }

    @Test("closes DeepSeek DSML thinking after tool results when reasoning is off")
    func closesDeepSeekDSMLThinkingAfterToolResultsWhenReasoningIsOff() {
        let rendered = MLXPromptRenderer.render(
            Self.toolResultRequest(reasoningEnabled: false),
            style: .deepSeekDSML
        )

        #expect(rendered.prompt.hasSuffix("</think>"))
        #expect(!rendered.prompt.contains("</function_results><｜Assistant｜>"))
    }

    @Test("extracts DeepSeek DSML tool calls with typed parameters")
    func extractsDeepSeekDSMLToolCallsWithTypedParameters() throws {
        let calls = MLXToolCallExtractor.extractAll(from: Self.multiCallText)
        let weather = try #require(calls.first)
        let search = try #require(calls.last)
        let weatherArguments = try Self.jsonObject(from: weather.argumentsJSON)
        let searchArguments = try Self.jsonObject(from: search.argumentsJSON)
        let options = try #require(weatherArguments["options"] as? [String: String])

        #expect(calls.map(\.name) == ["weather", "search"])
        #expect(weatherArguments["city"] as? String == "Berlin")
        #expect(weatherArguments["count"] as? Int == 2)
        #expect(weatherArguments["metric"] as? Bool == true)
        #expect(options == ["unit": "celsius"])
        #expect(searchArguments["query"] as? String == "MLX Swift")
    }

    @Test("extracts DeepSeek DSML Python literal fallback values")
    func extractsDeepSeekDSMLPythonLiteralFallbackValues() throws {
        let call = try #require(MLXToolCallExtractor.extract(from: Self.pythonLiteralText))
        let arguments = try Self.jsonObject(from: call.argumentsJSON)
        let metadata = try #require(arguments["metadata"] as? [String: Any])

        #expect(call.name == "weather")
        #expect(arguments["enabled"] as? Bool == false)
        #expect(arguments["attempts"] as? [Int] == [1, 2])
        #expect(metadata["fresh"] as? Bool == true)
        #expect(metadata["limit"] is NSNull)
    }

    @Test("suppresses split DeepSeek DSML tool envelopes while streaming")
    func suppressesSplitDeepSeekDSMLToolEnvelopes() {
        var filter = MLXToolCallStreamFilter()
        let output = [
            filter.feed("before <\(Self.dsmlToken)tool"),
            filter.feed(#"_calls><｜DSML｜invoke name="weather">"#),
            filter.feed(#"<｜DSML｜parameter name="city" string="true">Berlin</｜DSML｜parameter>"#),
            filter.feed("</\(Self.dsmlToken)invoke></\(Self.dsmlToken)tool_calls> after"),
            filter.finish()
        ].joined()

        #expect(output == "before  after")
    }

    @Test("infers DeepSeek DSML prompt style")
    func infersDeepSeekDSMLPromptStyle() {
        let profile = MLXModelProfile.make(
            config: [
                "model_type": "deepseek_v4",
                "architectures": ["DeepseekV4ForCausalLM"]
            ],
            tokenizerConfig: [
                "chat_template": "<\(Self.dsmlToken)tool_calls><\(Self.dsmlToken)invoke name=\"weather\">"
            ],
            id: "deepseek-v4-flash"
        )

        #expect(profile.promptStyle == .deepSeekDSML)
        #expect(profile.capabilities.reasoning)
    }

    private static let dsmlToken = MLXToolPromptDialect.deepSeekDSMLToken

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

    private static var searchTool: MLXBridgeToolDefinition {
        MLXBridgeToolDefinition(
            name: "search",
            description: "Search local documents",
            parametersJSONSchema: #"{"type":"object","properties":{"query":{"type":"string"}}}"#
        )
    }

    private static func toolResultRequest(reasoningEnabled: Bool) -> MLXBridgeRequest {
        MLXBridgeRequest(
            messages: [
                MLXBridgeMessage(role: .user, content: "Use both tools."),
                MLXBridgeMessage(
                    role: .assistant,
                    content: """
                    [{"tool_name":"weather","arguments":{"city":"Berlin"}},\
                    {"tool_name":"search","arguments":{"query":"MLX Swift"}}]
                    """
                ),
                MLXBridgeMessage(role: .tool, content: #"{"temperature":18}"#, name: "weather"),
                MLXBridgeMessage(role: .tool, content: #"{"results":["MLX Swift"]}"#, name: "search")
            ],
            reasoningEnabled: reasoningEnabled,
            tools: [weatherTool, searchTool]
        )
    }

    private static var multiCallText: String {
        """
        <\(dsmlToken)tool_calls>
        <\(dsmlToken)invoke name="weather">
        <\(dsmlToken)parameter name="city" string="true">Berlin</\(dsmlToken)parameter>
        <\(dsmlToken)parameter name="count" string="false">2</\(dsmlToken)parameter>
        <\(dsmlToken)parameter name="metric" string="false">true</\(dsmlToken)parameter>
        <\(dsmlToken)parameter name="options" string="false">{"unit":"celsius"}</\(dsmlToken)parameter>
        </\(dsmlToken)invoke>
        <\(dsmlToken)invoke name="search">
        <\(dsmlToken)parameter name="query" string="true">
        MLX Swift
        </\(dsmlToken)parameter>
        </\(dsmlToken)invoke>
        </\(dsmlToken)tool_calls>
        """
    }

    private static var pythonLiteralText: String {
        """
        <\(dsmlToken)tool_calls>
        <\(dsmlToken)invoke name="weather">
        <\(dsmlToken)parameter name="enabled" string="false">False</\(dsmlToken)parameter>
        <\(dsmlToken)parameter name="attempts" string="false">[1, 2]</\(dsmlToken)parameter>
        <\(dsmlToken)parameter name="metadata" string="false">\
        {'fresh': True, 'limit': None}</\(dsmlToken)parameter>
        </\(dsmlToken)invoke>
        </\(dsmlToken)tool_calls>
        """
    }

    private static func jsonObject(from text: String) throws -> [String: Any] {
        let data = try #require(text.data(using: .utf8))
        return try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
    }

    private static func occurrences(of needle: String, in text: String) -> Int {
        text.components(separatedBy: needle).count - 1
    }
}

import Foundation
@testable import MLXFoundationModel
import MLXLocalModels
import Testing

@Suite("MLX tool-aware stream reducer")
struct MLXToolAwareStreamReducerTests {
    @Test("streams visible text before final tool call")
    func streamsVisibleTextBeforeFinalToolCall() throws {
        var reducer = MLXToolAwareStreamReducer(tools: [Self.weatherTool])
        var actions: [MLXToolAwareStreamReducer.Action] = []

        let firstActions = reducer.consume(Self.textChunk("visible before ", tokenCount: 3))
        actions.append(contentsOf: firstActions)
        actions.append(contentsOf: reducer.consume(Self.textChunk("<too", tokenCount: 1)))
        actions.append(contentsOf: reducer.consume(Self.textChunk("""
        l_call><function=weather><parameter=city>Berlin</parameter>\
        <parameter=count>"2"</parameter></function></tool_call> visible after
        """, tokenCount: 6)))
        actions.append(contentsOf: reducer.consume(Self.metricsChunk()))
        actions.append(contentsOf: reducer.finish())

        let call = try #require(Self.toolCalls(in: actions).first)
        let arguments = try Self.jsonObject(from: call.argumentsJSON)

        #expect(Self.responseTexts(in: firstActions) == ["visible before "])
        #expect(Self.responseTexts(in: actions).joined() == "visible before  visible after")
        #expect(Self.responseTokenCounts(in: actions) == [3, 7])
        #expect(!Self.responseTexts(in: actions).joined().contains("tool_call"))
        #expect(call.name == "weather")
        #expect(arguments["city"] as? String == "Berlin")
        #expect(arguments["count"] as? Int == 2)
        #expect(Self.actionKinds(in: actions) == ["responseText", "responseText", "toolCall", "toolUsage"])
    }

    @Test("streams plain chunks and records response usage without tools")
    func streamsPlainChunksAndRecordsResponseUsageWithoutTools() {
        var reducer = MLXToolAwareStreamReducer(tools: [])
        var actions: [MLXToolAwareStreamReducer.Action] = []

        actions.append(contentsOf: reducer.consume(Self.textChunk("one ", tokenCount: 2)))
        actions.append(contentsOf: reducer.consume(Self.textChunk("two", tokenCount: 3)))
        actions.append(contentsOf: reducer.consume(Self.metricsChunk()))
        actions.append(contentsOf: reducer.finish())

        #expect(Self.responseTexts(in: actions) == ["one ", "two"])
        #expect(Self.responseTokenCounts(in: actions) == [2, 3])
        #expect(Self.actionKinds(in: actions) == ["responseText", "responseText", "responseUsage"])
    }

    @Test("records response usage when tool-enabled stream has no calls")
    func recordsResponseUsageWhenToolEnabledStreamHasNoCalls() {
        var reducer = MLXToolAwareStreamReducer(tools: [Self.weatherTool])
        var actions: [MLXToolAwareStreamReducer.Action] = []

        actions.append(contentsOf: reducer.consume(Self.textChunk("No tool needed.")))
        actions.append(contentsOf: reducer.consume(Self.metricsChunk()))
        actions.append(contentsOf: reducer.finish())

        #expect(Self.responseTexts(in: actions) == ["No tool needed."])
        #expect(Self.actionKinds(in: actions) == ["responseText", "responseUsage"])
    }

    @Test("streams Harmony analysis as think block")
    func streamsHarmonyAnalysisAsThinkBlock() {
        var reducer = MLXToolAwareStreamReducer(tools: [Self.weatherTool])
        var actions: [MLXToolAwareStreamReducer.Action] = []

        actions.append(contentsOf: reducer.consume(Self.textChunk(
            "<|channel|>analysis<|message|>thinking<|end|>"
        )))
        actions.append(contentsOf: reducer.consume(Self.textChunk(
            "<|start|>assistant<|channel|>final<|message|>Answer<|return|>"
        )))
        actions.append(contentsOf: reducer.consume(Self.metricsChunk()))
        actions.append(contentsOf: reducer.finish())

        #expect(Self.responseTexts(in: actions) == ["<think>\nthinking</think>\n", "Answer"])
        #expect(Self.actionKinds(in: actions) == ["responseText", "responseText", "responseUsage"])
    }

    @Test("emits Harmony commentary tool call without streaming protocol text")
    func emitsHarmonyCommentaryToolCallWithoutStreamingProtocolText() throws {
        var reducer = MLXToolAwareStreamReducer(tools: [Self.weatherTool])
        var actions: [MLXToolAwareStreamReducer.Action] = []

        actions.append(contentsOf: reducer.consume(Self.textChunk(
            #" to=functions.weather<|channel|>commentary<|message|>{"city":"Berlin"}<|call|>"#,
            tokenCount: 5
        )))
        actions.append(contentsOf: reducer.consume(Self.metricsChunk()))
        actions.append(contentsOf: reducer.finish())

        let call = try #require(Self.toolCalls(in: actions).first)

        #expect(Self.responseTexts(in: actions).isEmpty)
        #expect(call.name == "weather")
        #expect(call.argumentsJSON == #"{"city":"Berlin"}"#)
        #expect(Self.toolTokenCounts(in: actions) == [5])
        #expect(Self.actionKinds(in: actions) == ["toolCall", "toolUsage"])
    }

    @Test("streams Gemma 4 thought channel as think block")
    func streamsGemma4ThoughtChannelAsThinkBlock() {
        var reducer = MLXToolAwareStreamReducer(tools: [Self.weatherTool])
        var actions: [MLXToolAwareStreamReducer.Action] = []

        actions.append(contentsOf: reducer.consume(Self.textChunk("<|channel>thought\nreasoning")))
        actions.append(contentsOf: reducer.consume(Self.textChunk("<channel|>Answer<turn|>")))
        actions.append(contentsOf: reducer.consume(Self.metricsChunk()))
        actions.append(contentsOf: reducer.finish())

        #expect(Self.responseTexts(in: actions).joined() == "<think>\nreasoning</think>\nAnswer")
        #expect(Self.actionKinds(in: actions) == ["responseText", "responseText", "responseUsage"])
    }

    @Test("suppresses Gemma 4 tool call after protocol filtering")
    func suppressesGemma4ToolCallAfterProtocolFiltering() throws {
        var reducer = MLXToolAwareStreamReducer(tools: [Self.weatherTool])
        var actions: [MLXToolAwareStreamReducer.Action] = []

        actions.append(contentsOf: reducer.consume(Self.textChunk(
            #"<|tool_call>call:weather{"city":"Berlin"}<tool_call|>"#,
            tokenCount: 4
        )))
        actions.append(contentsOf: reducer.consume(Self.metricsChunk()))
        actions.append(contentsOf: reducer.finish())

        let call = try #require(Self.toolCalls(in: actions).first)

        #expect(Self.responseTexts(in: actions).isEmpty)
        #expect(call.name == "weather")
        #expect(call.argumentsJSON == #"{"city":"Berlin"}"#)
        #expect(Self.toolTokenCounts(in: actions) == [4])
        #expect(Self.actionKinds(in: actions) == ["toolCall", "toolUsage"])
    }

    @Test("keeps text after empty Qwen tool block")
    func keepsTextAfterEmptyQwenToolBlock() {
        var reducer = MLXToolAwareStreamReducer(tools: [Self.weatherTool])
        var actions: [MLXToolAwareStreamReducer.Action] = []

        actions.append(contentsOf: reducer.consume(Self.textChunk("<tool_call>")))
        actions.append(contentsOf: reducer.consume(Self.textChunk("</tool_call>Visible answer.")))
        actions.append(contentsOf: reducer.consume(Self.metricsChunk()))
        actions.append(contentsOf: reducer.finish())

        #expect(Self.toolCalls(in: actions).isEmpty)
        #expect(Self.responseTexts(in: actions).joined() == "Visible answer.")
        #expect(Self.actionKinds(in: actions) == ["responseText", "responseUsage"])
    }

    @Test("keeps split Qwen XML parameter values containing angle brackets")
    func keepsSplitQwenXMLParameterValuesContainingAngleBrackets() throws {
        var reducer = MLXToolAwareStreamReducer(tools: [Self.weatherTool])
        var actions: [MLXToolAwareStreamReducer.Action] = []

        actions.append(contentsOf: reducer.consume(Self.textChunk(
            "<tool_call><function=functions.weather><parameter=condition>temperature "
        )))
        actions.append(contentsOf: reducer.consume(Self.textChunk(
            "< 10</parameter></function></tool_call>Visible answer."
        )))
        actions.append(contentsOf: reducer.consume(Self.metricsChunk()))
        actions.append(contentsOf: reducer.finish())

        let call = try #require(Self.toolCalls(in: actions).first)
        let arguments = try Self.jsonObject(from: call.argumentsJSON)

        #expect(call.name == "weather")
        #expect(arguments["condition"] as? String == "temperature < 10")
        #expect(Self.responseTexts(in: actions).joined() == "Visible answer.")
        #expect(Self.actionKinds(in: actions) == ["responseText", "toolCall", "toolUsage"])
    }

    @Test("keeps text after empty Gemma 4 tool block")
    func keepsTextAfterEmptyGemma4ToolBlock() {
        var reducer = MLXToolAwareStreamReducer(tools: [Self.weatherTool])
        var actions: [MLXToolAwareStreamReducer.Action] = []

        actions.append(contentsOf: reducer.consume(Self.textChunk("<|tool_call>")))
        actions.append(contentsOf: reducer.consume(Self.textChunk("<tool_call|>Visible answer.")))
        actions.append(contentsOf: reducer.consume(Self.metricsChunk()))
        actions.append(contentsOf: reducer.finish())

        #expect(Self.toolCalls(in: actions).isEmpty)
        #expect(Self.responseTexts(in: actions).joined() == "Visible answer.")
        #expect(Self.actionKinds(in: actions) == ["responseText", "responseUsage"])
    }

    @Test("drops unknown thinking tool calls and keeps response usage")
    func dropsUnknownThinkingToolCallsAndKeepsResponseUsage() {
        var reducer = MLXToolAwareStreamReducer(tools: [Self.searchTool])
        var actions: [MLXToolAwareStreamReducer.Action] = []

        actions.append(contentsOf: reducer.consume(Self.textChunk(
            """
            <think><tool_call>{"name":"weather","arguments":{"city":"Berlin"}}</tool_call></think>\
            I can answer directly.
            """,
            tokenCount: 6
        )))
        actions.append(contentsOf: reducer.consume(Self.metricsChunk()))
        actions.append(contentsOf: reducer.finish())

        #expect(Self.toolCalls(in: actions).isEmpty)
        #expect(Self.responseTexts(in: actions).joined().contains("I can answer directly."))
        #expect(Self.actionKinds(in: actions) == ["responseText", "responseUsage"])
    }

    @Test("streams MiniMax M3 thinking markers as think block")
    func streamsMiniMaxM3ThinkingMarkersAsThinkBlock() {
        var reducer = MLXToolAwareStreamReducer(tools: [Self.weatherTool])
        var actions: [MLXToolAwareStreamReducer.Action] = []

        actions.append(contentsOf: reducer.consume(Self.textChunk("<mm:think>reasoning")))
        actions.append(contentsOf: reducer.consume(Self.textChunk("</mm:think>[e~[Answer")))
        actions.append(contentsOf: reducer.consume(Self.metricsChunk()))
        actions.append(contentsOf: reducer.finish())

        #expect(Self.responseTexts(in: actions).joined() == "<think>reasoning</think>Answer")
        #expect(Self.actionKinds(in: actions) == ["responseText", "responseText", "responseUsage"])
    }

    @Test("suppresses MiniMax M3 tool call after protocol filtering")
    func suppressesMiniMaxM3ToolCallAfterProtocolFiltering() throws {
        var reducer = MLXToolAwareStreamReducer(tools: [Self.weatherTool])
        var actions: [MLXToolAwareStreamReducer.Action] = []

        actions.append(contentsOf: reducer.consume(Self.textChunk("before ]<]minimax[>[<tool")))
        actions.append(contentsOf: reducer.consume(Self.textChunk(
            #"_call>]<]minimax[>[<invoke name="weather">"#
        )))
        actions.append(contentsOf: reducer.consume(Self.textChunk(
            "]<]minimax[>[<city>Berlin]<]minimax[>[</city>"
        )))
        actions.append(contentsOf: reducer.consume(Self.textChunk(
            "]<]minimax[>[</invoke>]<]minimax[>[</tool_call> after"
        )))
        actions.append(contentsOf: reducer.consume(Self.metricsChunk()))
        actions.append(contentsOf: reducer.finish())

        let call = try #require(Self.toolCalls(in: actions).first)

        #expect(Self.responseTexts(in: actions).joined() == "before  after")
        #expect(call.name == "weather")
        #expect(call.argumentsJSON == #"{"city":"Berlin"}"#)
        #expect(Self.actionKinds(in: actions) == ["responseText", "responseText", "toolCall", "toolUsage"])
    }

    private static var weatherTool: MLXBridgeToolDefinition {
        MLXBridgeToolDefinition(
            name: "weather",
            description: "Read local weather",
            parametersJSONSchema: """
            {
              "type":"object",
              "properties":{
                "city":{"type":"string"},
                "condition":{"type":"string"},
                "count":{"type":"integer"}
              }
            }
            """
        )
    }

    private static var searchTool: MLXBridgeToolDefinition {
        MLXBridgeToolDefinition(
            name: "search",
            description: "Search",
            parametersJSONSchema: """
            {"type":"object","properties":{"query":{"type":"string"}}}
            """
        )
    }

    private static func textChunk(_ text: String, tokenCount: Int = 1) -> LLMStreamChunk {
        LLMStreamChunk(text: text, event: .text, tokenCount: tokenCount)
    }

    private static func metricsChunk() -> LLMStreamChunk {
        LLMStreamChunk(
            text: "",
            event: .finished,
            metrics: ChunkMetrics(usage: UsageMetrics(
                generatedTokens: 8,
                totalTokens: 13,
                promptTokens: 5,
                promptCacheReusedTokenCount: 2
            ))
        )
    }

    private static func responseTexts(
        in actions: [MLXToolAwareStreamReducer.Action]
    ) -> [String] {
        actions.compactMap { action in
            if case .responseText(let text, _) = action {
                return text
            }
            return nil
        }
    }

    private static func responseTokenCounts(
        in actions: [MLXToolAwareStreamReducer.Action]
    ) -> [Int] {
        actions.compactMap { action in
            if case .responseText(_, let tokenCount) = action {
                return tokenCount
            }
            return nil
        }
    }

    private static func toolCalls(
        in actions: [MLXToolAwareStreamReducer.Action]
    ) -> [MLXExtractedToolCall] {
        actions.compactMap { action in
            if case .toolCall(let call, _) = action {
                return call
            }
            return nil
        }
    }

    private static func toolTokenCounts(
        in actions: [MLXToolAwareStreamReducer.Action]
    ) -> [Int] {
        actions.compactMap { action in
            if case .toolCall(_, let tokenCount) = action {
                return tokenCount
            }
            return nil
        }
    }

    private static func actionKinds(
        in actions: [MLXToolAwareStreamReducer.Action]
    ) -> [String] {
        actions.map { action in
            switch action {
            case .responseText:
                return "responseText"

            case .responseUsage:
                return "responseUsage"

            case .toolCall:
                return "toolCall"

            case .toolUsage:
                return "toolUsage"
            }
        }
    }

    private static func jsonObject(from text: String) throws -> [String: Any] {
        let data = try #require(text.data(using: .utf8))
        return try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
    }
}

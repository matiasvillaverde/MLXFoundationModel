import Foundation
@testable import MLXFoundationModel
import MLXLocalModels
import Testing

@Suite("MLX LongCat stream reducer")
struct MLXLongCatStreamReducerTests {
    @Test("drops replacement characters at LongCat think marker boundaries")
    func dropsReplacementCharactersAtLongCatThinkMarkerBoundaries() {
        var filter = MLXLongCatStreamFilter()
        let output = [
            filter.feed("<longcat_think>reasoning\u{FFFD}</long"),
            filter.feed("cat_think>"),
            filter.feed("\u{FFFD}Answer"),
            filter.finish()
        ].joined()

        #expect(output == "<think>reasoning</think>Answer")
    }

    @Test("suppresses JSON tool call while streaming visible text")
    func suppressesJSONToolCallWhileStreamingVisibleText() throws {
        var reducer = MLXToolAwareStreamReducer(tools: [Self.weatherTool])
        var actions: [MLXToolAwareStreamReducer.Action] = []

        actions.append(contentsOf: reducer.consume(Self.textChunk("before <longcat_tool")))
        actions.append(contentsOf: reducer.consume(Self.textChunk(
            #"_call>{"name":"weather","arguments":{"city":"Berlin","count":2}}"#
        )))
        actions.append(contentsOf: reducer.consume(Self.textChunk("</longcat_tool_call> after")))
        actions.append(contentsOf: reducer.consume(Self.metricsChunk()))
        actions.append(contentsOf: reducer.finish())

        let call = try #require(Self.toolCalls(in: actions).first)
        let arguments = try Self.jsonObject(from: call.argumentsJSON)

        #expect(Self.responseTexts(in: actions).joined() == "before  after")
        #expect(call.name == "weather")
        #expect(arguments["city"] as? String == "Berlin")
        #expect(arguments["count"] as? Int == 2)
        #expect(Self.actionKinds(in: actions) == [
            "responseText",
            "responseText",
            "toolCall",
            "toolUsage"
        ])
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

    private static func textChunk(_ text: String) -> LLMStreamChunk {
        LLMStreamChunk(text: text, event: .text, tokenCount: 1)
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

    private static func actionKinds(
        in actions: [MLXToolAwareStreamReducer.Action]
    ) -> [String] {
        actions.map { action in
            switch action {
            case .responseText:
                "responseText"

            case .responseUsage:
                "responseUsage"

            case .toolCall:
                "toolCall"

            case .toolUsage:
                "toolUsage"
            }
        }
    }

    private static func jsonObject(from text: String) throws -> [String: Any] {
        let data = try #require(text.data(using: .utf8))
        return try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
    }
}

import Foundation
@testable import MLXFoundationModel
import MLXLocalModels
import Testing

@Suite("MLX Cohere stream reducer")
struct MLXCohereStreamReducerTests {
    @Test("streams Cohere Command thinking markers as think block")
    func streamsCohereCommandThinkingMarkersAsThinkBlock() {
        var reducer = MLXToolAwareStreamReducer(tools: [], promptStyle: .cohereAction)
        var actions: [MLXToolAwareStreamReducer.Action] = []

        actions.append(contentsOf: reducer.consume(Self.textChunk("<|START_THINK")))
        actions.append(contentsOf: reducer.consume(Self.textChunk("ING|>reasoning<|END_THINKING|>")))
        actions.append(contentsOf: reducer.consume(Self.textChunk("<|START_TEXT|>Answer<|END_TEXT|>")))
        actions.append(contentsOf: reducer.consume(Self.metricsChunk()))
        actions.append(contentsOf: reducer.finish())

        #expect(Self.responseTexts(in: actions).joined() == "<think>\nreasoning</think>\nAnswer")
        #expect(Self.actionKinds(in: actions) == ["responseText", "responseText", "responseUsage"])
    }

    @Test("suppresses Cohere action marker tool call while streaming")
    func suppressesCohereActionMarkerToolCallWhileStreaming() throws {
        var reducer = MLXToolAwareStreamReducer(tools: [Self.weatherTool])
        var actions: [MLXToolAwareStreamReducer.Action] = []

        actions.append(contentsOf: reducer.consume(Self.textChunk("before <|START_ACT")))
        actions.append(contentsOf: reducer.consume(Self.textChunk(
            #"ION|>{"tool_name":"weather","parameters":{"city":"Berlin"}}<|END_ACTION|> after"#
        )))
        actions.append(contentsOf: reducer.consume(Self.metricsChunk()))
        actions.append(contentsOf: reducer.finish())

        let call = try #require(Self.toolCalls(in: actions).first)

        #expect(Self.responseTexts(in: actions).joined() == "before  after")
        #expect(call.name == "weather")
        #expect(call.argumentsJSON == #"{"city":"Berlin"}"#)
        #expect(Self.actionKinds(in: actions) == [
            "responseText",
            "responseText",
            "toolCall",
            "toolUsage"
        ])
    }

    @Test("suppresses Cohere array action tool call while streaming")
    func suppressesCohereArrayActionToolCallWhileStreaming() throws {
        var reducer = MLXToolAwareStreamReducer(tools: [Self.weatherTool])
        var actions: [MLXToolAwareStreamReducer.Action] = []

        actions.append(contentsOf: reducer.consume(Self.textChunk("before <|START_ACTION|>[")))
        actions.append(contentsOf: reducer.consume(Self.textChunk(
            #"{"tool_call_id":"0","tool_name":"weather","parameters":{"city":"Berlin"}}"#
        )))
        actions.append(contentsOf: reducer.consume(Self.textChunk("]<|END_ACTION|> after")))
        actions.append(contentsOf: reducer.consume(Self.metricsChunk()))
        actions.append(contentsOf: reducer.finish())

        let call = try #require(Self.toolCalls(in: actions).first)

        #expect(Self.responseTexts(in: actions).joined() == "before  after")
        #expect(call.name == "weather")
        #expect(call.argumentsJSON == #"{"city":"Berlin"}"#)
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
            {"type":"object","properties":{"city":{"type":"string"}}}
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
}

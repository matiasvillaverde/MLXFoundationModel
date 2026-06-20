import Foundation
@testable import MLXFoundationModel
import MLXLocalModels
import Testing

@Suite("MLX Kimi stream reducer")
struct MLXKimiStreamReducerTests {
    @Test("suppresses bare Kimi tool calls while streaming visible text")
    func suppressesBareKimiToolCallsWhileStreamingVisibleText() throws {
        var reducer = MLXToolAwareStreamReducer(tools: [Self.weatherTool])
        var actions: [MLXToolAwareStreamReducer.Action] = []

        actions.append(contentsOf: reducer.consume(Self.textChunk("before <|tool_call_beg")))
        actions.append(contentsOf: reducer.consume(Self.textChunk(
            #"in|>functions.weather:0<|tool_call_argument_begin|>{"city":"Berlin","count":2}"#
        )))
        actions.append(contentsOf: reducer.consume(Self.textChunk("<|tool_call_end|> after")))
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

import Foundation
@testable import MLXFoundationModel
import MLXLocalModels
import Testing

@Suite("MLX tool-aware stream reducer early tools")
struct MLXToolAwareStreamReducerEarlyToolTests {
    @Test("emits completed tool calls before stream finish")
    func emitsCompletedToolCallsBeforeStreamFinish() throws {
        var reducer = MLXToolAwareStreamReducer(tools: [Self.weatherTool])
        let earlyActions = reducer.consume(Self.textChunk(Self.weatherToolCall, tokenCount: 5))
        var actions = earlyActions

        let call = try #require(Self.toolCalls(in: earlyActions).first)
        actions.append(contentsOf: reducer.consume(Self.metricsChunk()))
        actions.append(contentsOf: reducer.finish())

        #expect(call.name == "weather")
        #expect(call.argumentsJSON == #"{"city":"Berlin"}"#)
        #expect(Self.toolCalls(in: actions).count == 1)
        #expect(Self.actionKinds(in: actions) == ["toolCall", "toolUsage"])
    }

    private static var weatherTool: MLXBridgeToolDefinition {
        MLXBridgeToolDefinition(
            name: "weather",
            description: "Read local weather",
            parametersJSONSchema: #"{"type":"object","properties":{"city":{"type":"string"}}}"#
        )
    }

    private static let weatherToolCall = """
    <tool_call><function=weather><parameter=city>Berlin</parameter></function></tool_call>
    """

    private static func textChunk(_ text: String, tokenCount: Int) -> LLMStreamChunk {
        LLMStreamChunk(text: text, event: .text, tokenCount: tokenCount)
    }

    private static func metricsChunk() -> LLMStreamChunk {
        LLMStreamChunk(
            text: "",
            event: .finished,
            metrics: ChunkMetrics(usage: UsageMetrics(
                generatedTokens: 5,
                totalTokens: 9,
                promptTokens: 4
            ))
        )
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
}

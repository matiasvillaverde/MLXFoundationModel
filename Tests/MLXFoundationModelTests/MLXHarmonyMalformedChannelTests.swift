@testable import MLXFoundationModel
import MLXLocalModels
import Testing

@Suite("MLX Harmony malformed channel hardening")
struct MLXHarmonyMalformedChannelTests {
    @Test("stream filter preserves malformed tool channels as visible text")
    func streamFilterPreservesMalformedToolChannelsAsVisibleText() {
        var filter = MLXHarmonyStreamFilter()
        let output = [
            filter.feed(#" to=functions.weather<|channel|>comentary<|message|>{"city":"Berlin"}"#),
            filter.feed("<|call|>"),
            filter.finish()
        ].joined()

        #expect(output == #"{"city":"Berlin"}"#)
    }

    @Test("extractor ignores malformed Harmony tool channel")
    func extractorIgnoresMalformedHarmonyToolChannel() {
        let call = MLXToolCallExtractor.extract(
            from: """
            <|start|>assistant to=functions.weather<|channel|>comentary\
            <|message|>{"city":"Berlin"}<|call|>
            """
        )

        #expect(call == nil)
    }

    @Test("stream reducer does not execute malformed Harmony tool channel")
    func streamReducerDoesNotExecuteMalformedHarmonyToolChannel() {
        var reducer = MLXToolAwareStreamReducer(tools: [Self.weatherTool])
        var actions: [MLXToolAwareStreamReducer.Action] = []

        actions.append(contentsOf: reducer.consume(Self.textChunk(
            #" to=functions.weather<|channel|>comentary<|message|>{"city":"Berlin"}"#,
            tokenCount: 5
        )))
        actions.append(contentsOf: reducer.consume(Self.textChunk("<|call|>")))
        actions.append(contentsOf: reducer.consume(Self.metricsChunk()))
        actions.append(contentsOf: reducer.finish())

        #expect(Self.toolCalls(in: actions).isEmpty)
        #expect(Self.responseTexts(in: actions).joined() == #"{"city":"Berlin"}"#)
        #expect(Self.actionKinds(in: actions) == ["responseText", "responseUsage"])
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

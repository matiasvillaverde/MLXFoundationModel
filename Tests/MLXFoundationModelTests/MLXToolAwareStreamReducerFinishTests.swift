import Foundation
@testable import MLXFoundationModel
import MLXLocalModels
import Testing

@Suite("MLX tool-aware stream reducer finish")
struct MLXToolAwareStreamReducerFinishTests {
    @Test("flushes retained protocol suffix when no-tool stream finishes")
    func flushesRetainedProtocolSuffixWhenNoToolStreamFinishes() {
        var reducer = MLXToolAwareStreamReducer(tools: [])
        var actions: [MLXToolAwareStreamReducer.Action] = []

        actions.append(contentsOf: reducer.consume(Self.textChunk("literal <|chan", tokenCount: 4)))
        actions.append(contentsOf: reducer.consume(Self.metricsChunk()))
        actions.append(contentsOf: reducer.finish())

        #expect(Self.responseTexts(in: actions) == ["literal ", "<|chan"])
        #expect(Self.actionKinds(in: actions) == ["responseText", "responseText", "responseUsage"])
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

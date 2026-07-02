import Foundation
@testable import MLXFoundationModel
import MLXLocalModels
import Testing

@Suite("MLX stream lifecycle translation")
struct MLXStreamLifecycleTranslatorTests {
    @Test("ignores lifecycle metadata chunks")
    func ignoresLifecycleMetadataChunks() async throws {
        let sink = RecordingStreamEventSink()

        try await MLXStreamEventTranslator().translate(
            Self.stream([
                Self.lifecycleChunk(.init(phase: .request, state: .started)),
                Self.lifecycleChunk(.init(phase: .promptProcessing, state: .started)),
                Self.lifecycleChunk(.init(
                    phase: .promptProcessing,
                    state: .ended,
                    totalUnitCount: 5,
                    cachedUnitCount: 2
                )),
                Self.lifecycleChunk(.init(phase: .decode, state: .started)),
                Self.textChunk("ok", tokenCount: 1),
                Self.metricsChunk()
            ]),
            into: sink,
            tools: []
        )

        let events = await sink.snapshot()

        #expect(Self.responseTexts(in: events) == ["ok"])
        #expect(Self.eventKinds(in: events) == ["responseText", "responseUsage"])
    }

    private static func stream(
        _ chunks: [LLMStreamChunk]
    ) -> AsyncThrowingStream<LLMStreamChunk, any Error> {
        AsyncThrowingStream { continuation in
            for chunk in chunks {
                continuation.yield(chunk)
            }
            continuation.finish()
        }
    }

    private static func lifecycleChunk(_ event: StreamLifecycleEvent) -> LLMStreamChunk {
        LLMStreamChunk(text: "", event: .lifecycle(event))
    }

    private static func textChunk(_ text: String, tokenCount: Int = 1) -> LLMStreamChunk {
        LLMStreamChunk(text: text, event: .text, tokenCount: tokenCount)
    }

    private static func metricsChunk() -> LLMStreamChunk {
        LLMStreamChunk(
            text: "",
            event: .finished,
            metrics: ChunkMetrics(usage: UsageMetrics(
                generatedTokens: 1,
                totalTokens: 6,
                promptTokens: 5,
                promptCacheReusedTokenCount: 2
            ))
        )
    }

    private static func responseTexts(
        in events: [MLXTranslatedStreamEvent]
    ) -> [String] {
        events.compactMap { event in
            if case .responseText(let text, _) = event {
                return text
            }
            return nil
        }
    }

    private static func eventKinds(
        in events: [MLXTranslatedStreamEvent]
    ) -> [String] {
        events.map { event in
            switch event {
            case .responseText:
                return "responseText"

            case .reasoningText:
                return "reasoningText"

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

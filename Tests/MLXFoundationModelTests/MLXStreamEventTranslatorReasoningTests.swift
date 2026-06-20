@testable import MLXFoundationModel
import MLXLocalModels
import Testing

@Suite("MLX stream event translator reasoning")
struct MLXStreamEventTranslatorReasoningTests {
    @Test("splits think tags into typed reasoning events")
    func splitsThinkTagsIntoTypedReasoningEvents() async throws {
        let sink = RecordingStreamEventSink()

        try await MLXStreamEventTranslator().translate(
            Self.stream([
                Self.textChunk("visible <thi", tokenCount: 2),
                Self.textChunk("nk>\nprivate</think> final", tokenCount: 6),
                Self.metricsChunk()
            ]),
            into: sink,
            tools: []
        )

        let events = await sink.snapshot()
        let allResponseText = Self.responseTexts(in: events).joined()
        let allReasoningText = Self.reasoningTexts(in: events).joined()

        #expect(Self.responseTexts(in: events) == ["visible ", " final"])
        #expect(Self.reasoningTexts(in: events) == ["private"])
        #expect(!allResponseText.contains("<think>"))
        #expect(!allResponseText.contains("</think>"))
        #expect(!allReasoningText.contains("<think>"))
        #expect(Self.eventKinds(in: events) == [
            "responseText",
            "reasoningText",
            "responseText",
            "responseUsage"
        ])
    }

    @Test("splits LongCat native thinking markers into typed reasoning events")
    func splitsLongCatNativeThinkingMarkersIntoTypedReasoningEvents() async throws {
        let sink = RecordingStreamEventSink()

        try await MLXStreamEventTranslator().translate(
            Self.stream([
                Self.textChunk("visible <longcat_", tokenCount: 2),
                Self.textChunk("think>\nprivate</longcat_think> final", tokenCount: 6),
                Self.metricsChunk()
            ]),
            into: sink,
            tools: [],
            promptStyle: .longCat
        )

        let events = await sink.snapshot()
        let allResponseText = Self.responseTexts(in: events).joined()
        let allReasoningText = Self.reasoningTexts(in: events).joined()

        #expect(Self.responseTexts(in: events) == ["visible ", " final"])
        #expect(Self.reasoningTexts(in: events) == ["private"])
        #expect(!allResponseText.contains("longcat_think"))
        #expect(!allReasoningText.contains("longcat_think"))
        #expect(Self.eventKinds(in: events) == [
            "responseText",
            "reasoningText",
            "responseText",
            "responseUsage"
        ])
    }

    @Test("flushes partial think marker before translated tool events")
    func flushesPartialThinkMarkerBeforeTranslatedToolEvents() async throws {
        let source = LLMChunkStreamSource()
        let sink = RecordingStreamEventSink()
        let translation = makeTranslationTask(source: source, sink: sink)

        source.yield(Self.textChunk("visible <thi", tokenCount: 2))
        let firstEvent = await sink.event(at: 0)
        source.yield(Self.textChunk(Self.weatherToolCall, tokenCount: 5))
        source.yield(Self.metricsChunk())
        source.finish()
        try await translation.value

        let events = await sink.snapshot()
        Self.expectResponseText(firstEvent, text: "visible ", tokenCount: 2)
        #expect(Self.responseTexts(in: events) == ["visible ", "<thi"])
        #expect(Self.eventKinds(in: events) == [
            "responseText",
            "responseText",
            "toolCall",
            "toolUsage"
        ])
    }

    @Test("recovers unclosed thinking as response when stream has no answer")
    func recoversUnclosedThinkingAsResponseWhenStreamHasNoAnswer() async throws {
        let sink = RecordingStreamEventSink()

        try await MLXStreamEventTranslator().translate(
            Self.stream([
                Self.textChunk("<think>\nbody without close", tokenCount: 5),
                Self.metricsChunk()
            ]),
            into: sink,
            tools: []
        )

        let events = await sink.snapshot()

        #expect(Self.reasoningTexts(in: events) == ["body without close"])
        #expect(Self.responseTexts(in: events) == ["body without close"])
        #expect(Self.eventKinds(in: events) == [
            "reasoningText",
            "responseText",
            "responseUsage"
        ])
    }

    @Test("classifies prompt-opened thinking without streamed open marker")
    func classifiesPromptOpenedThinkingWithoutStreamedOpenMarker() async throws {
        let sink = RecordingStreamEventSink()

        try await MLXStreamEventTranslator().translate(
            Self.stream([
                Self.textChunk("reasoning</think>\nAnswer", tokenCount: 5),
                Self.metricsChunk()
            ]),
            into: sink,
            tools: [],
            reasoningStartsOpen: true
        )

        let events = await sink.snapshot()

        #expect(Self.reasoningTexts(in: events) == ["reasoning"])
        #expect(Self.responseTexts(in: events) == ["Answer"])
        #expect(Self.eventKinds(in: events) == [
            "reasoningText",
            "responseText",
            "responseUsage"
        ])
    }

    @Test("classifies prompt-opened Cohere Command thinking")
    func classifiesPromptOpenedCohereCommandThinking() async throws {
        let sink = RecordingStreamEventSink()

        try await MLXStreamEventTranslator().translate(
            Self.stream([
                Self.textChunk("reasoning<|END_THINKING|><|START_TEXT|>Answer", tokenCount: 5),
                Self.metricsChunk()
            ]),
            into: sink,
            tools: [],
            promptStyle: .cohereAction,
            reasoningStartsOpen: true
        )

        let events = await sink.snapshot()

        #expect(Self.reasoningTexts(in: events) == ["reasoning"])
        #expect(Self.responseTexts(in: events) == ["Answer"])
        #expect(Self.eventKinds(in: events) == [
            "reasoningText",
            "responseText",
            "responseUsage"
        ])
    }

    private func makeTranslationTask(
        source: LLMChunkStreamSource,
        sink: RecordingStreamEventSink
    ) -> Task<Void, any Error> {
        let stream = source.stream()
        return Task {
            try await MLXStreamEventTranslator().translate(
                stream,
                into: sink,
                tools: [Self.weatherTool]
            )
        }
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

    private static func textChunk(_ text: String, tokenCount: Int) -> LLMStreamChunk {
        LLMStreamChunk(text: text, event: .text, tokenCount: tokenCount)
    }

    private static func metricsChunk() -> LLMStreamChunk {
        LLMStreamChunk(
            text: "",
            event: .finished,
            metrics: ChunkMetrics(usage: UsageMetrics(
                generatedTokens: 8,
                totalTokens: 13,
                promptTokens: 5
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

    private static func reasoningTexts(
        in events: [MLXTranslatedStreamEvent]
    ) -> [String] {
        events.compactMap { event in
            if case .reasoningText(let text, _) = event {
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

    private static func expectResponseText(
        _ event: MLXTranslatedStreamEvent,
        text: String,
        tokenCount: Int
    ) {
        guard case .responseText(let actualText, let actualTokenCount) = event else {
            Issue.record("Expected responseText event, got \(event)")
            return
        }
        #expect(actualText == text)
        #expect(actualTokenCount == tokenCount)
    }
}

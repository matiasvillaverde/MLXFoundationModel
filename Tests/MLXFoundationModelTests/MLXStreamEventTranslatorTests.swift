import Foundation
@testable import MLXFoundationModel
import MLXLocalModels
import Testing

@Suite("MLX stream event translator")
struct MLXStreamEventTranslatorTests {
    @Test("streams visible response text before final tool call")
    func streamsVisibleResponseTextBeforeFinalToolCall() async throws {
        let output = try await translateControlledToolStream()
        let firstEvent = output.firstEvent
        let events = output.events
        let call = try #require(Self.toolCalls(in: events).first)
        Self.expectResponseText(firstEvent, text: "visible before ", tokenCount: 3)

        #expect(Self.responseTexts(in: events).joined() == "visible before  visible after")
        #expect(!Self.responseTexts(in: events).joined().contains("tool_call"))
        #expect(call.name == "weather")
        #expect(call.argumentsJSON.contains(#""city":"Berlin""#))
        #expect(Self.eventKinds(in: events) == [
            "responseText",
            "responseText",
            "toolCall",
            "toolUsage"
        ])
    }

    private func translateControlledToolStream() async throws -> (
        firstEvent: MLXTranslatedStreamEvent,
        events: [MLXTranslatedStreamEvent]
    ) {
        let source = LLMChunkStreamSource()
        let sink = RecordingStreamEventSink()
        let translation = makeTranslationTask(source: source, sink: sink)

        source.yield(Self.textChunk("visible before ", tokenCount: 3))
        let firstEvent = await sink.event(at: 0)
        source.yield(Self.textChunk("<too", tokenCount: 1))
        source.yield(Self.textChunk(Self.toolEnvelopeSuffix, tokenCount: 6))
        source.yield(Self.metricsChunk())
        source.finish()
        try await translation.value

        return (firstEvent: firstEvent, events: await sink.snapshot())
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

    @Test("emits completed tool events before source stream finishes")
    func emitsCompletedToolEventsBeforeSourceStreamFinishes() async throws {
        let source = LLMChunkStreamSource()
        let sink = RecordingStreamEventSink()
        let translation = makeTranslationTask(source: source, sink: sink)

        source.yield(Self.textChunk(Self.weatherToolCall, tokenCount: 5))
        let firstEvent = await sink.event(at: 0)
        source.yield(Self.metricsChunk())
        source.finish()
        try await translation.value

        Self.expectToolCall(firstEvent, name: "weather", argumentsJSON: #"{"city":"Berlin"}"#)
        #expect(Self.eventKinds(in: await sink.snapshot()) == ["toolCall", "toolUsage"])
    }

    @Test("routes no-tool streams directly to response usage")
    func routesNoToolStreamsDirectlyToResponseUsage() async throws {
        let sink = RecordingStreamEventSink()

        try await MLXStreamEventTranslator().translate(
            Self.stream([
                Self.textChunk("one ", tokenCount: 2),
                Self.textChunk("two", tokenCount: 3),
                Self.metricsChunk()
            ]),
            into: sink,
            tools: []
        )

        let events = await sink.snapshot()

        #expect(Self.responseTexts(in: events) == ["one ", "two"])
        #expect(Self.responseTokenCounts(in: events) == [2, 3])
        #expect(Self.eventKinds(in: events) == ["responseText", "responseText", "responseUsage"])
    }

    @Test("scopes protocol normalization to prompt style")
    func scopesProtocolNormalizationToPromptStyle() async throws {
        let sink = RecordingStreamEventSink()
        let text = "literal <|channel>thought\nnot Gemma<channel|>"

        try await MLXStreamEventTranslator().translate(
            Self.stream([
                Self.textChunk(text, tokenCount: 7),
                Self.metricsChunk()
            ]),
            into: sink,
            tools: [],
            promptStyle: .qwenXML
        )

        let events = await sink.snapshot()

        #expect(Self.responseTexts(in: events) == [text])
        #expect(Self.reasoningTexts(in: events).isEmpty)
        #expect(Self.eventKinds(in: events) == ["responseText", "responseUsage"])
    }

    @Test("normalizes matching prompt style protocol markers")
    func normalizesMatchingPromptStyleProtocolMarkers() async throws {
        let sink = RecordingStreamEventSink()

        try await MLXStreamEventTranslator().translate(
            Self.stream([
                Self.textChunk("<|channel>thought\nreasoning<channel|>Answer<turn|>", tokenCount: 7),
                Self.metricsChunk()
            ]),
            into: sink,
            tools: [],
            promptStyle: .gemma
        )

        let events = await sink.snapshot()

        #expect(Self.reasoningTexts(in: events) == ["reasoning"])
        #expect(Self.responseTexts(in: events) == ["Answer"])
        #expect(Self.eventKinds(in: events) == ["reasoningText", "responseText", "responseUsage"])
    }

    @Test("flushes retained protocol suffix before no-tool response usage")
    func flushesRetainedProtocolSuffixBeforeNoToolResponseUsage() async throws {
        let sink = RecordingStreamEventSink()

        try await MLXStreamEventTranslator().translate(
            Self.stream([
                Self.textChunk("literal <|chan", tokenCount: 4),
                Self.metricsChunk()
            ]),
            into: sink,
            tools: []
        )

        let events = await sink.snapshot()

        #expect(Self.responseTexts(in: events) == ["literal ", "<|chan"])
        #expect(Self.eventKinds(in: events) == ["responseText", "responseText", "responseUsage"])
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

    private static let toolEnvelopeSuffix = """
    l_call><function=weather><parameter=city>Berlin</parameter>\
    <parameter=count>"2"</parameter></function></tool_call> visible after
    """

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
        in events: [MLXTranslatedStreamEvent]
    ) -> [String] {
        events.compactMap { event in
            if case .responseText(let text, _) = event {
                return text
            }
            return nil
        }
    }

    private static func responseTokenCounts(
        in events: [MLXTranslatedStreamEvent]
    ) -> [Int] {
        events.compactMap { event in
            if case .responseText(_, let tokenCount) = event {
                return tokenCount
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

    private static func toolCalls(
        in events: [MLXTranslatedStreamEvent]
    ) -> [MLXExtractedToolCall] {
        events.compactMap { event in
            if case .toolCall(let call, _) = event {
                return call
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

    private static func expectToolCall(
        _ event: MLXTranslatedStreamEvent,
        name: String,
        argumentsJSON: String
    ) {
        guard case .toolCall(let call, _) = event else {
            Issue.record("Expected toolCall event, got \(event)")
            return
        }
        #expect(call.name == name)
        #expect(call.argumentsJSON == argumentsJSON)
    }
}

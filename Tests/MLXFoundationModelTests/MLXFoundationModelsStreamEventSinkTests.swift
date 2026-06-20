#if FOUNDATION_MODELS_PROVIDER_API && canImport(FoundationModels)
import FoundationModels
@testable import MLXFoundationModel
import MLXLocalModels
import Testing

@Suite("Foundation Models stream event sink")
struct MLXFoundationModelsStreamEventSinkTests {
    @Test("sends provider response text before source stream finishes")
    func sendsProviderResponseTextBeforeSourceStreamFinishes() async throws {
        guard #available(macOS 27.0, iOS 27.0, visionOS 27.0, *) else {
            return
        }

        try await Self.runIncrementalProviderResponseTest()
    }

    @Test("flushes retained visible text before provider tool calls")
    func flushesRetainedVisibleTextBeforeProviderToolCalls() async throws {
        guard #available(macOS 27.0, iOS 27.0, visionOS 27.0, *) else {
            return
        }

        try await Self.runRetainedTextFlushTest()
    }

    @available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
    private static func runIncrementalProviderResponseTest() async throws {
        let channel = LanguageModelExecutorGenerationChannel()
        let source = LLMChunkStreamSource()
        let stream = source.stream()
        let translation = Task {
            try await MLXEventTranslator().translate(
                stream,
                into: channel,
                tools: [Self.weatherTool]
            )
        }
        var iterator = channel.makeAsyncIterator()

        source.yield(Self.textChunk("visible before ", tokenCount: 3))
        let firstEvent = try await #require(iterator.next())
        Self.expectResponseText(firstEvent, text: "visible before ")

        #expect(!translation.isCancelled)
        source.yield(Self.textChunk(Self.toolCallText, tokenCount: 4))
        source.yield(Self.metricsChunk())
        source.finish()
        try await translation.value
    }

    @available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
    private static func runRetainedTextFlushTest() async throws {
        let channel = LanguageModelExecutorGenerationChannel()
        let source = LLMChunkStreamSource()
        let stream = source.stream()
        let translation = Task {
            try await MLXEventTranslator().translate(
                stream,
                into: channel,
                tools: [Self.weatherTool]
            )
        }

        source.yield(Self.textChunk("visible <thi", tokenCount: 2))
        source.yield(Self.textChunk(Self.toolCallText, tokenCount: 4))
        source.yield(Self.metricsChunk())
        source.finish()
        try await translation.value

        let events = try await Self.events(from: channel, count: 4)
        let toolCall = try #require(Self.toolCall(in: events))

        #expect(Self.responseTexts(in: events) == ["visible ", "<thi"])
        #expect(Self.eventKinds(in: events) == [
            "responseText",
            "responseText",
            "toolCall",
            "toolUsage"
        ])
        #expect(toolCall.name == "weather")
        #expect(toolCall.argumentsJSON == #"{"count":2}"#)
    }

    @available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
    private static func events(
        from channel: LanguageModelExecutorGenerationChannel,
        count: Int
    ) async throws -> [any LanguageModelExecutorGenerationChannel.Event] {
        var iterator = channel.makeAsyncIterator()
        var events: [any LanguageModelExecutorGenerationChannel.Event] = []
        for _ in 0..<count {
            if let event = try await iterator.next() {
                events.append(event)
            }
        }
        return events
    }

    @available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
    private static func expectResponseText(
        _ event: any LanguageModelExecutorGenerationChannel.Event,
        text: String
    ) {
        guard
            let response = event as? LanguageModelExecutorGenerationChannel.Response,
            case .appendText(let fragment) = response.action
        else {
            Issue.record("Expected response text event")
            return
        }
        #expect(fragment.content == text)
    }

    @available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
    private static func responseTexts(
        in events: [any LanguageModelExecutorGenerationChannel.Event]
    ) -> [String] {
        events.compactMap { event in
            guard
                let response = event as? LanguageModelExecutorGenerationChannel.Response,
                case .appendText(let fragment) = response.action
            else {
                return nil
            }
            return fragment.content
        }
    }

    @available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
    private static func toolCall(
        in events: [any LanguageModelExecutorGenerationChannel.Event]
    ) -> MLXExtractedToolCall? {
        events.compactMap(Self.toolCall(from:)).first
    }

    @available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
    private static func toolCall(
        from event: any LanguageModelExecutorGenerationChannel.Event
    ) -> MLXExtractedToolCall? {
        guard
            let toolCalls = event as? LanguageModelExecutorGenerationChannel.ToolCalls,
            case .toolCall(let call) = toolCalls.action,
            case .appendArguments(let fragment) = call.action
        else {
            return nil
        }
        return MLXExtractedToolCall(name: call.name, argumentsJSON: fragment.content)
    }

    @available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
    private static func eventKinds(
        in events: [any LanguageModelExecutorGenerationChannel.Event]
    ) -> [String] {
        events.map { event in
            switch event {
            case is LanguageModelExecutorGenerationChannel.Response:
                return "responseText"

            case let toolCalls as LanguageModelExecutorGenerationChannel.ToolCalls:
                return toolCallKind(toolCalls)

            default:
                return "other"
            }
        }
    }

    @available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
    private static func toolCallKind(
        _ event: LanguageModelExecutorGenerationChannel.ToolCalls
    ) -> String {
        switch event.action {
        case .toolCall:
            return "toolCall"

        case .updateUsage:
            return "toolUsage"

        default:
            return "toolCalls"
        }
    }

    private static var weatherTool: MLXBridgeToolDefinition {
        MLXBridgeToolDefinition(
            name: "weather",
            description: "Read weather",
            parametersJSONSchema: #"{"type":"object","properties":{"count":{"type":"integer"}}}"#
        )
    }

    private static let toolCallText = """
    <tool_call><function=weather><parameter=count>"2"</parameter></function></tool_call>
    """

    private static func textChunk(_ text: String, tokenCount: Int) -> LLMStreamChunk {
        LLMStreamChunk(text: text, event: .text, tokenCount: tokenCount)
    }

    private static func metricsChunk() -> LLMStreamChunk {
        LLMStreamChunk(
            text: "",
            event: .finished,
            metrics: ChunkMetrics(usage: UsageMetrics(
                generatedTokens: 6,
                totalTokens: 10,
                promptTokens: 4
            ))
        )
    }
}
#endif

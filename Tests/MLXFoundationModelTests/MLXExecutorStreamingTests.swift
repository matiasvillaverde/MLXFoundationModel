#if FOUNDATION_MODELS_PROVIDER_API && canImport(FoundationModels)
import Foundation
import FoundationModels
@testable import MLXFoundationModel
import MLXLocalModels
import Testing

@Suite("MLX executor streaming")
struct MLXExecutorStreamingTests {
    @available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
    @Generable
    struct WeatherArguments {
        let city: String
    }

    @Test("streams provider text before source generation finishes")
    func streamsProviderTextBeforeSourceGenerationFinishes() async throws {
        guard #available(macOS 27.0, iOS 27.0, visionOS 27.0, *) else {
            return
        }

        try await Self.runIncrementalToolEnabledStreamTest()
    }

    @Test("streams prompt-opened thinking as provider reasoning")
    func streamsPromptOpenedThinkingAsProviderReasoning() async throws {
        guard #available(macOS 27.0, iOS 27.0, visionOS 27.0, *) else {
            return
        }

        try await Self.runPromptOpenedReasoningStreamTest()
    }

    @available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
    private static func runIncrementalToolEnabledStreamTest() async throws {
        let model = Self.model()
        let source = LLMChunkStreamSource()
        let session = StreamingSession(source: source)
        let executor = try MLXExecutor(configuration: Self.configuration(for: model), session: session)
        let channel = LanguageModelExecutorGenerationChannel()
        let response = Task {
            try await executor.respond(to: Self.request(), model: model, streamingInto: channel)
        }
        var iterator = channel.makeAsyncIterator()

        try await Self.waitUntil { await session.streamCallCount() == 1 }
        source.yield(Self.textChunk("visible before ", tokenCount: 3))
        let firstEvent = try await #require(iterator.next())
        Self.expectResponseText(firstEvent, text: "visible before ")

        source.yield(Self.textChunk(Self.toolCallText, tokenCount: 4))
        source.yield(Self.metricsChunk())
        source.finish()
        try await response.value
    }

    @available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
    private static func runPromptOpenedReasoningStreamTest() async throws {
        let model = Self.model(reasoning: true)
        let source = LLMChunkStreamSource()
        let session = StreamingSession(source: source)
        let executor = try MLXExecutor(configuration: Self.configuration(for: model), session: session)
        let channel = LanguageModelExecutorGenerationChannel()
        let response = Task {
            try await executor.respond(to: Self.reasoningRequest(), model: model, streamingInto: channel)
        }
        var iterator = channel.makeAsyncIterator()

        try await Self.waitUntil { await session.streamCallCount() == 1 }
        source.yield(Self.textChunk("reasoning</think>\nAnswer", tokenCount: 5))
        let firstEvent = try await #require(iterator.next())
        let secondEvent = try await #require(iterator.next())
        Self.expectReasoningText(firstEvent, text: "reasoning")
        Self.expectResponseText(secondEvent, text: "Answer")

        source.yield(Self.metricsChunk())
        source.finish()
        try await response.value
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
    private static func expectReasoningText(
        _ event: any LanguageModelExecutorGenerationChannel.Event,
        text: String
    ) {
        guard
            let reasoning = event as? LanguageModelExecutorGenerationChannel.Reasoning,
            case .appendText(let fragment) = reasoning.action
        else {
            Issue.record("Expected reasoning text event")
            return
        }
        #expect(fragment.content == text)
    }

    @available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
    private static func model(reasoning: Bool = false) -> MLXLanguageModel {
        MLXLanguageModel(
            model: MLXModel(
                id: "executor-streaming-fixture",
                location: URL(fileURLWithPath: "/tmp/MLXFoundationModelStreaming"),
                promptStyle: .qwenXML,
                capabilities: MLXModelCapabilities(
                    toolCalling: true,
                    structuredOutput: true,
                    reasoning: reasoning
                )
            )
        )
    }

    @available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
    private static func configuration(for model: MLXLanguageModel) -> MLXExecutor.Configuration {
        MLXExecutor.Configuration(
            model: model.model,
            compute: model.compute,
            runtime: model.runtime,
            sampling: model.sampling,
            maximumResponseTokens: model.maximumResponseTokens
        )
    }

    @available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
    private static func request() -> LanguageModelExecutorGenerationRequest {
        LanguageModelExecutorGenerationRequest(
            id: UUID(),
            transcript: Self.transcript(prompt: "Read the weather."),
            enabledTools: [Self.weatherTool],
            generationOptions: GenerationOptions(samplingMode: .greedy, maximumResponseTokens: 16),
            contextOptions: ContextOptions(),
            metadata: [:]
        )
    }

    @available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
    private static func reasoningRequest() -> LanguageModelExecutorGenerationRequest {
        LanguageModelExecutorGenerationRequest(
            id: UUID(),
            transcript: Self.transcript(prompt: "Think briefly, then answer."),
            enabledTools: [],
            generationOptions: GenerationOptions(samplingMode: .greedy, maximumResponseTokens: 16),
            contextOptions: ContextOptions(reasoningLevel: .light),
            metadata: [:]
        )
    }

    @available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
    private static func transcript(prompt: String) -> Transcript {
        Transcript(entries: [
            .prompt(.init(segments: [.text(.init(content: prompt))]))
        ])
    }

    @available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
    private static var weatherTool: Transcript.ToolDefinition {
        Transcript.ToolDefinition(
            name: "weather",
            description: "Read local weather for a city",
            parameters: WeatherArguments.generationSchema
        )
    }

    private static func waitUntil(condition: () async -> Bool) async throws {
        for _ in 0 ..< 100 {
            if await condition() {
                return
            }
            try await Task.sleep(for: .milliseconds(10))
        }
        Issue.record("Timed out waiting for async condition")
    }

    private static let toolCallText = """
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
                generatedTokens: 7,
                totalTokens: 11,
                promptTokens: 4
            ))
        )
    }

    private actor StreamingSession: MLXGeneratingSession {
        private let source: LLMChunkStreamSource
        private var streamCalls = 0

        init(source: LLMChunkStreamSource) {
            self.source = source
        }

        func stream(_ input: LLMInput) -> AsyncThrowingStream<LLMStreamChunk, any Error> {
            _ = input
            streamCalls += 1
            return source.stream()
        }

        nonisolated func stop() {
            // No active backend process in executor streaming tests.
        }

        func preload(configuration: ProviderConfiguration) -> AsyncThrowingStream<Progress, any Error> {
            _ = configuration
            return AsyncThrowingStream { continuation in
                let progress = Progress(totalUnitCount: 1)
                progress.completedUnitCount = 1
                continuation.yield(progress)
                continuation.finish()
            }
        }

        func unload() async {
            // Nothing to unload in the fake session.
        }

        func streamCallCount() -> Int {
            streamCalls
        }
    }
}
#endif

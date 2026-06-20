@testable import MLXFoundationModel
@testable import MLXLocalModels
import Testing

@Suite(
    "MLX real-model tool calling",
    .serialized,
    .disabled(
        if: !MLXRealModelEnvironment.isEnabled,
        "Set MLX_RUN_REAL_MODEL_TESTS=1 and download models with scripts/download-test-models.sh"
    )
)
struct MLXRealModelToolCallingTests {
    @Test("Qwen3 emits a parseable tool call when tools are rendered")
    func qwen3EmitsParseableToolCall() async throws {
        let models = try MLXRealModelCatalog.load()
        guard let model = try MLXRealModelHarness.selectedModel("qwen3-0.6b-4bit", in: models) else {
            return
        }
        let style = try MLXRealModelHarness.inferredPromptStyle(for: model)
        let rendered = MLXPromptRenderer.render(Self.toolRequest, style: style)
        let observed = try await MLXGenerationDiagnostics.withRecording {
            try await MLXRealModelHarness.run(
                model: model,
                prompt: rendered.prompt,
                sampling: .deterministic,
                limits: ResourceLimits(maxTokens: 160, maxTime: .seconds(120), reusePromptCache: false)
            )
        }
        let result = observed.result
        let tokenEvents = MLXRealModelHarness.generatedTokenSnapshots(from: observed.events)

        #expect(style == .qwenXML)
        #expect(rendered.prompt.contains("<tool_call><function=weather>"))
        #expect(!rendered.prompt.contains("Available tools:"))
        MLXRealModelHarness.verifyGenerated(result)
        MLXRealModelHarness.verifyGeneratedTokenDiagnostics(tokenEvents, result: result)
        let call = try #require(MLXToolCallExtractor.extract(from: result.text, tools: [Self.weatherTool]))
        #expect(call.name == "weather")
        #expect(call.argumentsJSON.contains("Berlin"))
    }

    @Test("Qwen3 tool stream emits tool events without protocol markup")
    func qwen3ToolStreamEmitsToolEventsWithoutProtocolMarkup() async throws {
        let models = try MLXRealModelCatalog.load()
        guard let model = try MLXRealModelHarness.selectedModel("qwen3-0.6b-4bit", in: models) else {
            return
        }
        let observed = try await MLXGenerationDiagnostics.withRecording {
            try await MLXRealModelHarness.translateRenderedRequest(
                model: model,
                request: Self.toolRequest,
                limits: ResourceLimits(maxTokens: 160, maxTime: .seconds(120), reusePromptCache: false)
            )
        }
        let tokenEvents = MLXRealModelHarness.generatedTokenSnapshots(from: observed.events)
        let toolCall = try #require(Self.toolCalls(in: observed.result).first)
        let usage = try #require(Self.toolUsages(in: observed.result).last)

        Self.verifyGeneratedTokens(from: tokenEvents)
        #expect(usage.generatedTokens == tokenEvents.count)
        #expect(toolCall.name == "weather")
        #expect(toolCall.argumentsJSON.contains("Berlin"))
        #expect(Self.responseTexts(in: observed.result).allSatisfy { !$0.contains("<tool_call") })
        #expect(Self.eventKinds(in: observed.result).contains("toolCall"))
        #expect(Self.eventKinds(in: observed.result).contains("toolUsage"))
    }

    private static var toolRequest: MLXBridgeRequest {
        MLXBridgeRequest(
            messages: [
                MLXBridgeMessage(role: .user, content: "/no_think\nCall the weather tool for Berlin.")
            ],
            instructions: """
            You are a tool router. Do not think aloud.
            Return exactly one JSON object and no prose:
            {"tool_name":"weather","arguments":{"city":"Berlin"}}
            """,
            tools: [weatherTool]
        )
    }

    private static var weatherTool: MLXBridgeToolDefinition {
        MLXBridgeToolDefinition(
            name: "weather",
            description: "Read local weather for a city",
            parametersJSONSchema: """
            {"type":"object","required":["city"],"properties":{"city":{"type":"string"}}}
            """
        )
    }

    private static func toolCalls(
        in events: [MLXTranslatedStreamEvent]
    ) -> [MLXExtractedToolCall] {
        events.compactMap { event in
            guard case .toolCall(let call, _) = event else {
                return nil
            }
            return call
        }
    }

    private static func toolUsages(
        in events: [MLXTranslatedStreamEvent]
    ) -> [UsageMetrics] {
        events.compactMap { event in
            guard case .toolUsage(let usage) = event else {
                return nil
            }
            return usage
        }
    }

    private static func responseTexts(
        in events: [MLXTranslatedStreamEvent]
    ) -> [String] {
        events.compactMap { event in
            guard case .responseText(let text, _) = event else {
                return nil
            }
            return text
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

    private static func verifyGeneratedTokens(
        from tokens: [MLXGeneratedTokenSnapshot]
    ) {
        let summary = Comment(rawValue: tokens.map(\.tokenText).joined())
        #expect(!tokens.isEmpty, summary)
        #expect(tokens.map(\.index) == Array(1 ... tokens.count), summary)
        #expect(tokens.contains { !$0.tokenText.isEmpty }, summary)
    }
}

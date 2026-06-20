import Foundation
@testable import MLXFoundationModel
@testable import MLXLocalModels
import Testing

extension MLXRealModelToolCallingTests {
    @Test("Gemma 4 native tool stream emits structured tool events")
    func gemma4NativeToolStreamEmitsStructuredToolEvents() async throws {
        let models = try MLXRealModelCatalog.load()
        guard let model = try MLXRealModelHarness.selectedModel("gemma-4-e2b-it-4bit", in: models) else {
            return
        }
        let style = try MLXRealModelHarness.inferredPromptStyle(for: model)
        let rendered = MLXPromptRenderer.render(Self.gemmaToolRequest, style: style)
        let observed = try await MLXGenerationDiagnostics.withRecording {
            try await MLXRealModelHarness.translateRenderedRequest(
                model: model,
                request: Self.gemmaToolRequest,
                limits: ResourceLimits(maxTokens: 160, maxTime: .seconds(120), reusePromptCache: false),
                style: style
            )
        }
        let tokenEvents = MLXRealModelHarness.generatedTokenSnapshots(from: observed.events)
        let toolCall = try #require(Self.gemmaToolCalls(in: observed.result).first)
        let usage = try #require(Self.gemmaToolUsages(in: observed.result).last)

        Self.verifyGemmaRenderedPrompt(rendered, style: style)
        Self.verifyGemmaGeneratedTokens(tokenEvents)
        #expect(usage.generatedTokens == tokenEvents.count)
        #expect(toolCall.name == "weather")
        #expect(toolCall.argumentsJSON.contains("Berlin"))
        #expect(Self.gemmaResponseTexts(in: observed.result).allSatisfy { text in
            !text.contains("<|tool_call>")
                && !text.contains("<tool_call|>")
                && !text.contains("<|channel>")
                && !text.contains("<channel|>")
        })
    }

    @Test("Gemma 4 constrained native tool stream emits typed arguments")
    func gemma4ConstrainedNativeToolStreamEmitsTypedArguments() async throws {
        let models = try MLXRealModelCatalog.load()
        guard let model = try MLXRealModelHarness.selectedModel("gemma-4-e2b-it-4bit", in: models) else {
            return
        }
        let observed = try await MLXGenerationDiagnostics.withRecording {
            try await MLXRealModelHarness.translateRenderedRequest(
                model: model,
                request: Self.gemmaSchemaConstrainedToolRequest,
                limits: ResourceLimits(maxTokens: 80, maxTime: .seconds(120), reusePromptCache: false),
                style: .gemma,
                sampling: Self.gemmaSchemaConstrainedToolSampling
            )
        }
        let tokenEvents = MLXRealModelHarness.generatedTokenSnapshots(from: observed.events)
        let grammarEvents = MLXRealModelHarness.grammarSnapshots(from: observed.events)
        let toolCall = try #require(Self.gemmaToolCalls(in: observed.result).first)
        let arguments = try Self.gemmaJSONObject(from: toolCall.argumentsJSON)

        Self.verifyGemmaGeneratedTokens(tokenEvents)
        Self.verifyGemmaSuccessfulGrammarDiagnostics(grammarEvents, kind: .ebnf)
        #expect(toolCall.name == "weather")
        #expect(arguments["count"] as? Int == 2)
        #expect(arguments["enabled"] as? Bool == true)
        #expect(Self.gemmaResponseTexts(in: observed.result).allSatisfy { text in
            !text.contains("<|tool_call>") && !text.contains("<tool_call|>")
        })
    }

    private static var gemmaToolRequest: MLXBridgeRequest {
        MLXBridgeRequest(
            messages: [
                MLXBridgeMessage(role: .user, content: "Call the weather tool for Berlin.")
            ],
            instructions: """
            You are a tool router. Do not think aloud.
            Emit only this Gemma function call and no prose:
            <|tool_call>call:weather{city:<|"|>Berlin<|"|>}<tool_call|>
            """,
            tools: [gemmaWeatherTool]
        )
    }

    private static func verifyGemmaRenderedPrompt(
        _ rendered: MLXRenderedRequest,
        style: MLXPromptStyle
    ) {
        #expect(style == .gemma)
        #expect(rendered.prompt.contains("<|tool>declaration:weather"))
        #expect(rendered.prompt.contains("<|turn>model\n"))
        #expect(!rendered.prompt.contains("Available tools:"))
    }

    private static var gemmaSchemaConstrainedToolRequest: MLXBridgeRequest {
        MLXBridgeRequest(
            messages: [
                MLXBridgeMessage(
                    role: .user,
                    content: "Call the weather tool with count 2 and enabled true."
                )
            ],
            instructions: """
            Emit exactly one Gemma function call and no prose.
            The count argument must be 2. The enabled argument must be true.
            """,
            tools: [gemmaControlTool]
        )
    }

    private static var gemmaWeatherTool: MLXBridgeToolDefinition {
        MLXBridgeToolDefinition(
            name: "weather",
            description: "Read local weather for a city",
            parametersJSONSchema: """
            {"type":"object","required":["city"],"properties":{"city":{"type":"string"}}}
            """
        )
    }

    private static var gemmaControlTool: MLXBridgeToolDefinition {
        MLXBridgeToolDefinition(
            name: "weather",
            description: "Read local weather controls",
            parametersJSONSchema: """
            {"type":"object","required":["count","enabled"],\
            "properties":{"count":{"type":"integer"},"enabled":{"type":"boolean"}}}
            """
        )
    }

    private static var gemmaSchemaConstrainedToolSampling: SamplingParameters {
        SamplingParameters(
            temperature: 0,
            topP: 1,
            topK: 1,
            seed: 42,
            advanced: AdvancedSamplingParameters(
                grammar: GrammarSamplingConfiguration(grammar: gemmaSchemaConstrainedToolGrammar)
            )
        )
    }

    private static let gemmaSchemaConstrainedToolGrammar = """
    root ::= "<|tool_call>call:weather{count:" fm_json_integer \
    ",enabled:" fm_json_boolean "}<tool_call|>"
    fm_json_integer ::= "-" [0-9] [0-9]* | [0-9] [0-9]*
    fm_json_boolean ::= "true" | "false"
    """

    private static func gemmaToolCalls(
        in events: [MLXTranslatedStreamEvent]
    ) -> [MLXExtractedToolCall] {
        events.compactMap { event in
            guard case .toolCall(let call, _) = event else {
                return nil
            }
            return call
        }
    }

    private static func gemmaToolUsages(
        in events: [MLXTranslatedStreamEvent]
    ) -> [UsageMetrics] {
        events.compactMap { event in
            guard case .toolUsage(let usage) = event else {
                return nil
            }
            return usage
        }
    }

    private static func gemmaResponseTexts(
        in events: [MLXTranslatedStreamEvent]
    ) -> [String] {
        events.compactMap { event in
            guard case .responseText(let text, _) = event else {
                return nil
            }
            return text
        }
    }

    private static func gemmaJSONObject(from text: String) throws -> [String: Any] {
        let data = try #require(text.data(using: .utf8))
        return try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
    }

    private static func verifyGemmaSuccessfulGrammarDiagnostics(
        _ events: [MLXGrammarConstraintSnapshot],
        kind: GrammarConstraintKind
    ) {
        let summary = Comment(rawValue: gemmaGrammarEventSummary(events))
        #expect(events.contains { $0.stage == .matcherPrepared && $0.kind == kind }, summary)
        #expect(events.contains { $0.stage == .maskApplied && $0.kind == kind }, summary)
        #expect(events.contains { $0.stage == .mlxMaskPrepared && $0.kind == kind }, summary)
        #expect(events.contains { $0.stage == .tokenAccepted && $0.kind == kind }, summary)
        #expect(!events.contains { $0.stage == .tokenRejected }, summary)
        #expect(!events.contains { $0.stage == .processorFailedClosed }, summary)
    }

    private static func gemmaGrammarEventSummary(_ events: [MLXGrammarConstraintSnapshot]) -> String {
        events
            .map { event in
                [
                    "stage=\(event.stage.rawValue)",
                    "kind=\(event.kind.map(String.init(describing:)) ?? "nil")",
                    "mode=\(event.mode.map(String.init(describing:)) ?? "nil")",
                    "tokenCount=\(event.tokenCount.map(String.init) ?? "nil")",
                    "tokenID=\(event.tokenID.map(String.init) ?? "nil")",
                    "message=\(event.message ?? "nil")"
                ].joined(separator: " ")
            }
            .joined(separator: "\n")
    }

    private static func verifyGemmaGeneratedTokens(
        _ tokens: [MLXGeneratedTokenSnapshot]
    ) {
        let summary = Comment(rawValue: tokens.map(\.tokenText).joined())
        #expect(!tokens.isEmpty, summary)
        #expect(tokens.map(\.index) == Array(1 ... tokens.count), summary)
        #expect(tokens.contains { !$0.tokenText.isEmpty }, summary)
    }
}

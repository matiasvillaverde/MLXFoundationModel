import Foundation
@testable import MLXFoundationModel
@testable import MLXLocalModels
import Testing

extension MLXRealModelToolCallingTests {
    @Test("Mistral constrained native tool stream emits typed arguments")
    func mistralConstrainedNativeToolStreamEmitsTypedArguments() async throws {
        let models = try MLXRealModelCatalog.load()
        guard let model = try MLXRealModelHarness.selectedModel("mistral-7b-v0.3-4bit", in: models) else {
            return
        }
        let style = try MLXRealModelHarness.inferredPromptStyle(for: model)
        let rendered = MLXPromptRenderer.render(Self.mistralSchemaConstrainedToolRequest, style: style)
        let observed = try await MLXGenerationDiagnostics.withRecording {
            try await MLXRealModelHarness.translateRenderedRequest(
                model: model,
                request: Self.mistralSchemaConstrainedToolRequest,
                limits: ResourceLimits(maxTokens: 80, maxTime: .seconds(120), reusePromptCache: false),
                style: style,
                sampling: Self.mistralSchemaConstrainedToolSampling
            )
        }
        let tokenEvents = MLXRealModelHarness.generatedTokenSnapshots(from: observed.events)
        let grammarEvents = MLXRealModelHarness.grammarSnapshots(from: observed.events)
        let toolCall = try #require(Self.mistralToolCalls(in: observed.result).first)
        let arguments = try Self.mistralJSONObject(from: toolCall.argumentsJSON)

        #expect(style == .mistralToolCall)
        #expect(rendered.prompt.contains("[AVAILABLE_TOOLS]"))
        #expect(!rendered.prompt.contains("Available tools:"))
        Self.verifyMistralGeneratedTokens(tokenEvents)
        Self.verifyMistralSuccessfulGrammarDiagnostics(grammarEvents, kind: .ebnf)
        #expect(toolCall.name == "weather")
        #expect(arguments["count"] as? Int == 2)
        #expect(arguments["enabled"] as? Bool == true)
        #expect(Self.mistralResponseTexts(in: observed.result).allSatisfy { text in
            !text.contains("[TOOL_CALLS]") && !text.contains("[ARGS]")
        })
    }

    private static var mistralSchemaConstrainedToolRequest: MLXBridgeRequest {
        MLXBridgeRequest(
            messages: [
                MLXBridgeMessage(
                    role: .user,
                    content: "Call the weather tool with count 2 and enabled true."
                )
            ],
            instructions: """
            Emit exactly one Mistral tool call and no prose.
            The count argument must be 2. The enabled argument must be true.
            """,
            tools: [mistralControlTool]
        )
    }

    private static var mistralControlTool: MLXBridgeToolDefinition {
        MLXBridgeToolDefinition(
            name: "weather",
            description: "Read local weather controls",
            parametersJSONSchema: """
            {"type":"object","required":["count","enabled"],\
            "properties":{"count":{"type":"integer"},"enabled":{"type":"boolean"}}}
            """
        )
    }

    private static var mistralSchemaConstrainedToolSampling: SamplingParameters {
        SamplingParameters(
            temperature: 0,
            topP: 1,
            topK: 1,
            seed: 42,
            advanced: AdvancedSamplingParameters(
                grammar: GrammarSamplingConfiguration(grammar: mistralSchemaConstrainedToolGrammar)
            )
        )
    }

    private static let mistralSchemaConstrainedToolGrammar = """
    root ::= "[TOOL_CALLS]weather[ARGS]{\\"count\\":" fm_json_integer \
    ",\\"enabled\\":" fm_json_boolean "}"
    fm_json_integer ::= "-" [0-9] [0-9]* | [0-9] [0-9]*
    fm_json_boolean ::= "true" | "false"
    """

    private static func mistralToolCalls(
        in events: [MLXTranslatedStreamEvent]
    ) -> [MLXExtractedToolCall] {
        events.compactMap { event in
            guard case .toolCall(let call, _) = event else {
                return nil
            }
            return call
        }
    }

    private static func mistralResponseTexts(
        in events: [MLXTranslatedStreamEvent]
    ) -> [String] {
        events.compactMap { event in
            guard case .responseText(let text, _) = event else {
                return nil
            }
            return text
        }
    }

    private static func mistralJSONObject(from text: String) throws -> [String: Any] {
        let data = try #require(text.data(using: .utf8))
        return try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
    }

    private static func verifyMistralSuccessfulGrammarDiagnostics(
        _ events: [MLXGrammarConstraintSnapshot],
        kind: GrammarConstraintKind
    ) {
        let summary = Comment(rawValue: mistralGrammarEventSummary(events))
        #expect(events.contains { $0.stage == .matcherPrepared && $0.kind == kind }, summary)
        #expect(events.contains { $0.stage == .maskApplied && $0.kind == kind }, summary)
        #expect(events.contains { $0.stage == .mlxMaskPrepared && $0.kind == kind }, summary)
        #expect(events.contains { $0.stage == .tokenAccepted && $0.kind == kind }, summary)
        #expect(!events.contains { $0.stage == .tokenRejected }, summary)
        #expect(!events.contains { $0.stage == .processorFailedClosed }, summary)
    }

    private static func mistralGrammarEventSummary(_ events: [MLXGrammarConstraintSnapshot]) -> String {
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

    private static func verifyMistralGeneratedTokens(
        _ tokens: [MLXGeneratedTokenSnapshot]
    ) {
        let summary = Comment(rawValue: tokens.map(\.tokenText).joined())
        #expect(!tokens.isEmpty, summary)
        #expect(tokens.map(\.index) == Array(1 ... tokens.count), summary)
        #expect(tokens.contains { !$0.tokenText.isEmpty }, summary)
    }
}

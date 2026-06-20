import Foundation
@testable import MLXFoundationModel
@testable import MLXLocalModels
import Testing

extension MLXRealModelToolCallingTests {
    @Test("GLM constrained native tool stream emits typed arguments")
    func glmConstrainedNativeToolStreamEmitsTypedArguments() async throws {
        let models = try MLXRealModelCatalog.load()
        guard let model = try MLXRealModelHarness.selectedModel("glm-4-9b-0414-4bit", in: models) else {
            return
        }
        let style = try MLXRealModelHarness.inferredPromptStyle(for: model)
        let rendered = MLXPromptRenderer.render(Self.glmSchemaConstrainedToolRequest, style: style)
        let observed = try await MLXGenerationDiagnostics.withRecording {
            try await MLXRealModelHarness.translateRenderedRequest(
                model: model,
                request: Self.glmSchemaConstrainedToolRequest,
                limits: ResourceLimits(maxTokens: 96, maxTime: .seconds(120), reusePromptCache: false),
                style: style,
                sampling: Self.glmSchemaConstrainedToolSampling
            )
        }
        let tokenEvents = MLXRealModelHarness.generatedTokenSnapshots(from: observed.events)
        let grammarEvents = MLXRealModelHarness.grammarSnapshots(from: observed.events)
        let toolCall = try #require(Self.glmToolCalls(in: observed.result).first)
        let arguments = try Self.glmJSONObject(from: toolCall.argumentsJSON)

        #expect(style == .glmXML)
        #expect(rendered.prompt.contains("<arg_key>count</arg_key>"))
        #expect(!rendered.prompt.contains("Available tools:"))
        Self.verifyGLMGeneratedTokens(tokenEvents)
        Self.verifyGLMSuccessfulGrammarDiagnostics(grammarEvents, kind: .ebnf)
        #expect(toolCall.name == "weather")
        #expect(arguments["count"] as? Int == 2)
        #expect(arguments["enabled"] as? Bool == true)
        #expect(Self.glmResponseTexts(in: observed.result).allSatisfy { text in
            !text.contains("<tool_call>") && !text.contains("<arg_key>")
        })
    }

    private static var glmSchemaConstrainedToolRequest: MLXBridgeRequest {
        MLXBridgeRequest(
            messages: [
                MLXBridgeMessage(
                    role: .user,
                    content: "Call the weather tool with count 2 and enabled true."
                )
            ],
            instructions: """
            Emit exactly one GLM XML tool call and no prose.
            The count argument must be 2. The enabled argument must be true.
            """,
            tools: [glmControlTool]
        )
    }

    private static var glmControlTool: MLXBridgeToolDefinition {
        MLXBridgeToolDefinition(
            name: "weather",
            description: "Read local weather controls",
            parametersJSONSchema: """
            {"type":"object","required":["count","enabled"],\
            "properties":{"count":{"type":"integer"},"enabled":{"type":"boolean"}}}
            """
        )
    }

    private static var glmSchemaConstrainedToolSampling: SamplingParameters {
        SamplingParameters(
            temperature: 0,
            topP: 1,
            topK: 1,
            seed: 42,
            advanced: AdvancedSamplingParameters(
                grammar: GrammarSamplingConfiguration(grammar: glmSchemaConstrainedToolGrammar)
            )
        )
    }

    private static let glmSchemaConstrainedToolGrammar = """
    root ::= "<tool_call>weather<arg_key>count</arg_key><arg_value>" fm_json_integer \
    "</arg_value><arg_key>enabled</arg_key><arg_value>" fm_json_boolean \
    "</arg_value></tool_call>"
    fm_json_integer ::= "-" [0-9] [0-9]* | [0-9] [0-9]*
    fm_json_boolean ::= "true" | "false"
    """

    private static func glmToolCalls(
        in events: [MLXTranslatedStreamEvent]
    ) -> [MLXExtractedToolCall] {
        events.compactMap { event in
            guard case .toolCall(let call, _) = event else {
                return nil
            }
            return call
        }
    }

    private static func glmResponseTexts(
        in events: [MLXTranslatedStreamEvent]
    ) -> [String] {
        events.compactMap { event in
            guard case .responseText(let text, _) = event else {
                return nil
            }
            return text
        }
    }

    private static func glmJSONObject(from text: String) throws -> [String: Any] {
        let data = try #require(text.data(using: .utf8))
        return try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
    }

    private static func verifyGLMSuccessfulGrammarDiagnostics(
        _ events: [MLXGrammarConstraintSnapshot],
        kind: GrammarConstraintKind
    ) {
        let summary = Comment(rawValue: glmGrammarEventSummary(events))
        #expect(events.contains { $0.stage == .matcherPrepared && $0.kind == kind }, summary)
        #expect(events.contains { $0.stage == .maskApplied && $0.kind == kind }, summary)
        #expect(events.contains { $0.stage == .mlxMaskPrepared && $0.kind == kind }, summary)
        #expect(events.contains { $0.stage == .tokenAccepted && $0.kind == kind }, summary)
        #expect(!events.contains { $0.stage == .tokenRejected }, summary)
        #expect(!events.contains { $0.stage == .processorFailedClosed }, summary)
    }

    private static func glmGrammarEventSummary(_ events: [MLXGrammarConstraintSnapshot]) -> String {
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

    private static func verifyGLMGeneratedTokens(
        _ tokens: [MLXGeneratedTokenSnapshot]
    ) {
        let summary = Comment(rawValue: tokens.map(\.tokenText).joined())
        #expect(!tokens.isEmpty, summary)
        #expect(tokens.map(\.index) == Array(1 ... tokens.count), summary)
        #expect(tokens.contains { !$0.tokenText.isEmpty }, summary)
    }
}

import Foundation
@testable import MLXFoundationModel
@testable import MLXLocalModels
import Testing

extension MLXRealModelToolCallingTests {
    @Test(
        "Harmony constrained native tool stream emits typed arguments",
        .disabled(
            if: !MLXRealModelEnvironment.canRunModel(id: "gpt-oss"),
            "gpt-oss requires a larger memory budget on this host"
        )
    )
    func harmonyConstrainedNativeToolStreamEmitsTypedArguments() async throws {
        let models = try MLXRealModelCatalog.load()
        guard let model = try MLXRealModelHarness.selectedModel("gpt-oss", in: models) else {
            return
        }
        let observed = try await MLXGenerationDiagnostics.withRecording {
            try await MLXRealModelHarness.translateRenderedRequest(
                model: model,
                request: Self.harmonySchemaConstrainedToolRequest,
                limits: ResourceLimits(maxTokens: 96, maxTime: .seconds(120), reusePromptCache: false),
                style: .harmony,
                sampling: Self.harmonySchemaConstrainedToolSampling
            )
        }
        let tokenEvents = MLXRealModelHarness.generatedTokenSnapshots(from: observed.events)
        let grammarEvents = MLXRealModelHarness.grammarSnapshots(from: observed.events)
        let toolCall = try #require(Self.harmonyToolCalls(in: observed.result).first)
        let arguments = try Self.harmonyJSONObject(from: toolCall.argumentsJSON)

        Self.verifyHarmonyGeneratedTokens(tokenEvents)
        Self.verifyHarmonySuccessfulGrammarDiagnostics(grammarEvents, kind: .ebnf)
        #expect(toolCall.name == "weather")
        #expect(arguments["count"] as? Int == 2)
        #expect(arguments["enabled"] as? Bool == true)
        #expect(Self.harmonyResponseTexts(in: observed.result).allSatisfy { text in
            !text.contains("<|channel|>") && !text.contains("<|message|>")
        })
    }

    private static var harmonySchemaConstrainedToolRequest: MLXBridgeRequest {
        MLXBridgeRequest(
            messages: [
                MLXBridgeMessage(
                    role: .user,
                    content: "Call the weather tool with count 2 and enabled true."
                )
            ],
            instructions: """
            Emit exactly one Harmony commentary tool call and no final prose.
            The count argument must be 2. The enabled argument must be true.
            """,
            tools: [harmonyControlTool]
        )
    }

    private static var harmonyControlTool: MLXBridgeToolDefinition {
        MLXBridgeToolDefinition(
            name: "weather",
            description: "Read local weather controls",
            parametersJSONSchema: """
            {"type":"object","required":["count","enabled"],\
            "properties":{"count":{"type":"integer"},"enabled":{"type":"boolean"}}}
            """
        )
    }

    private static var harmonySchemaConstrainedToolSampling: SamplingParameters {
        SamplingParameters(
            temperature: 0,
            topP: 1,
            topK: 1,
            seed: 42,
            advanced: AdvancedSamplingParameters(
                grammar: GrammarSamplingConfiguration(grammar: harmonySchemaConstrainedToolGrammar)
            )
        )
    }

    private static let harmonySchemaConstrainedToolGrammar = """
    root ::= " to=functions.weather<|channel|>commentary<|message|>{\\"count\\":" \
    fm_json_integer ",\\"enabled\\":" fm_json_boolean "}<|call|>"
    fm_json_integer ::= "-" [0-9] [0-9]* | [0-9] [0-9]*
    fm_json_boolean ::= "true" | "false"
    """

    private static func harmonyToolCalls(
        in events: [MLXTranslatedStreamEvent]
    ) -> [MLXExtractedToolCall] {
        events.compactMap { event in
            guard case .toolCall(let call, _) = event else {
                return nil
            }
            return call
        }
    }

    private static func harmonyResponseTexts(
        in events: [MLXTranslatedStreamEvent]
    ) -> [String] {
        events.compactMap { event in
            guard case .responseText(let text, _) = event else {
                return nil
            }
            return text
        }
    }

    private static func harmonyJSONObject(from text: String) throws -> [String: Any] {
        let data = try #require(text.data(using: .utf8))
        return try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
    }

    private static func verifyHarmonySuccessfulGrammarDiagnostics(
        _ events: [MLXGrammarConstraintSnapshot],
        kind: GrammarConstraintKind
    ) {
        let summary = Comment(rawValue: harmonyGrammarEventSummary(events))
        #expect(events.contains { $0.stage == .matcherPrepared && $0.kind == kind }, summary)
        #expect(events.contains { $0.stage == .maskApplied && $0.kind == kind }, summary)
        #expect(events.contains { $0.stage == .mlxMaskPrepared && $0.kind == kind }, summary)
        #expect(events.contains { $0.stage == .tokenAccepted && $0.kind == kind }, summary)
        #expect(!events.contains { $0.stage == .tokenRejected }, summary)
        #expect(!events.contains { $0.stage == .processorFailedClosed }, summary)
    }

    private static func harmonyGrammarEventSummary(_ events: [MLXGrammarConstraintSnapshot]) -> String {
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

    private static func verifyHarmonyGeneratedTokens(
        _ tokens: [MLXGeneratedTokenSnapshot]
    ) {
        let summary = Comment(rawValue: tokens.map(\.tokenText).joined())
        #expect(!tokens.isEmpty, summary)
        #expect(tokens.map(\.index) == Array(1 ... tokens.count), summary)
        #expect(tokens.contains { !$0.tokenText.isEmpty }, summary)
    }
}

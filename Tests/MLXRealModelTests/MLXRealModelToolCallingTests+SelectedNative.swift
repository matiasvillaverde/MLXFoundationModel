import Foundation
@testable import MLXFoundationModel
@testable import MLXLocalModels
import Testing

extension MLXRealModelToolCallingTests {
    private struct SelectedNativeObservation {
        let result: [MLXTranslatedStreamEvent]
        let events: [MLXGenerationDiagnosticEvent]
        let grammar: GrammarSamplingConfiguration
    }

    @Test("Selected native tool models emit constrained typed tool calls")
    func selectedNativeToolModelsEmitConstrainedTypedToolCalls() async throws {
        let selected = MLXRealModelEnvironment.selectedModels(from: try MLXRealModelCatalog.load())
        let missing = selected.filter { !MLXRealModelEnvironment.hasModelFiles(for: $0) }

        #expect(!selected.isEmpty)
        #expect(
            missing.isEmpty,
            Comment(rawValue: MLXRealModelEnvironment.missingModelsMessage(missing))
        )
        guard missing.isEmpty else {
            return
        }

        var failures: [String] = []
        var checkedModelCount = 0
        for model in selected {
            do {
                checkedModelCount += try await Self.verifySelectedNativeToolConstraints(model: model) ? 1 : 0
            } catch {
                failures.append("\(model.id): \(error)")
            }
        }
        #expect(failures.isEmpty, Comment(rawValue: failures.joined(separator: "\n")))
        #expect(checkedModelCount > 0)
    }

    private static func verifySelectedNativeToolConstraints(
        model: MLXRealModelCatalog.Model
    ) async throws -> Bool {
        let style = try MLXRealModelHarness.inferredPromptStyle(for: model)
        guard FMNativeToolGrammarFormat(promptStyle: style) != nil else {
            return false
        }
        let observed = try await Self.selectedNativeObservation(model: model, style: style)
        let tokenEvents = MLXRealModelHarness.generatedTokenSnapshots(from: observed.events)
        let grammarEvents = MLXRealModelHarness.grammarSnapshots(from: observed.events)
        let toolCall = try #require(Self.selectedNativeToolCalls(in: observed.result).first)
        let arguments = try Self.selectedNativeToolJSONObject(from: toolCall.argumentsJSON)

        Self.verifySelectedNativeTokens(tokenEvents)
        Self.verifySelectedNativeGrammarDiagnostics(grammarEvents, kind: observed.grammar.kind)
        #expect(toolCall.name == "weather")
        #expect(arguments["count"] as? Int == 2)
        #expect(arguments["enabled"] as? Bool == true)
        #expect(Self.selectedNativeResponseTexts(in: observed.result).allSatisfy { text in
            MLXToolCallEnvelopeDetector.firstEnvelope(in: text) == nil
        })
        return true
    }

    private static func selectedNativeObservation(
        model: MLXRealModelCatalog.Model,
        style: MLXPromptStyle
    ) async throws -> SelectedNativeObservation {
        let grammar = MLXRequiredToolGrammarBuilder.grammar(
            from: [Self.selectedNativeToolControlTool],
            promptStyle: style
        )
        let observed = try await MLXGenerationDiagnostics.withRecording {
            try await MLXRealModelHarness.translateRenderedRequest(
                model: model,
                request: Self.selectedNativeToolRequest,
                limits: Self.selectedNativeToolLimits,
                style: style,
                sampling: Self.selectedNativeToolSampling(grammar: grammar)
            )
        }
        return SelectedNativeObservation(
            result: observed.result,
            events: observed.events,
            grammar: grammar
        )
    }

    private static func selectedNativeToolSampling(
        grammar: GrammarSamplingConfiguration
    ) -> SamplingParameters {
        SamplingParameters(
            temperature: 0,
            topP: 1,
            topK: 1,
            seed: 42,
            advanced: AdvancedSamplingParameters(grammar: grammar)
        )
    }

    private static var selectedNativeToolLimits: ResourceLimits {
        ResourceLimits(
            maxTokens: 96,
            maxTime: .seconds(MLXRealModelEnvironment.architectureGenerationTimeoutSeconds),
            reusePromptCache: false
        )
    }

    private static var selectedNativeToolRequest: MLXBridgeRequest {
        MLXBridgeRequest(
            messages: [
                MLXBridgeMessage(
                    role: .user,
                    content: "Call the weather tool with count 2 and enabled true."
                )
            ],
            instructions: """
            Emit exactly one native tool call and no prose.
            The count argument must be 2. The enabled argument must be true.
            """,
            tools: [selectedNativeToolControlTool]
        )
    }

    private static var selectedNativeToolControlTool: MLXBridgeToolDefinition {
        MLXBridgeToolDefinition(
            name: "weather",
            description: "Read local weather controls",
            parametersJSONSchema: """
            {"type":"object","required":["count","enabled"],\
            "properties":{"count":{"const":2,"type":"integer"},\
            "enabled":{"const":true,"type":"boolean"}}}
            """
        )
    }

    private static func selectedNativeToolCalls(
        in events: [MLXTranslatedStreamEvent]
    ) -> [MLXExtractedToolCall] {
        events.compactMap { event in
            guard case .toolCall(let call, _) = event else {
                return nil
            }
            return call
        }
    }

    private static func selectedNativeResponseTexts(
        in events: [MLXTranslatedStreamEvent]
    ) -> [String] {
        events.compactMap { event in
            guard case .responseText(let text, _) = event else {
                return nil
            }
            return text
        }
    }

    private static func selectedNativeToolJSONObject(from text: String) throws -> [String: Any] {
        let data = try #require(text.data(using: .utf8))
        return try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
    }

    private static func verifySelectedNativeGrammarDiagnostics(
        _ events: [MLXGrammarConstraintSnapshot],
        kind: GrammarConstraintKind
    ) {
        let summary = Comment(rawValue: selectedNativeGrammarEventSummary(events))
        #expect(events.contains { $0.stage == .matcherPrepared && $0.kind == kind }, summary)
        #expect(events.contains { $0.stage == .maskApplied && $0.kind == kind }, summary)
        #expect(events.contains { $0.stage == .mlxMaskPrepared && $0.kind == kind }, summary)
        #expect(events.contains { $0.stage == .tokenAccepted && $0.kind == kind }, summary)
        #expect(!events.contains { $0.stage == .tokenRejected }, summary)
        #expect(!events.contains { $0.stage == .processorFailedClosed }, summary)
    }

    private static func selectedNativeGrammarEventSummary(
        _ events: [MLXGrammarConstraintSnapshot]
    ) -> String {
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

    private static func verifySelectedNativeTokens(
        _ tokens: [MLXGeneratedTokenSnapshot]
    ) {
        let summary = Comment(rawValue: tokens.map(\.tokenText).joined())
        #expect(!tokens.isEmpty, summary)
        #expect(tokens.map(\.index) == Array(1 ... tokens.count), summary)
        #expect(tokens.contains { !$0.tokenText.isEmpty }, summary)
    }
}

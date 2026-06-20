import Foundation
import MLXFoundationModel
import MLXFoundationModelExamples
@testable import MLXLocalModels
import Testing

@Suite(
    "MLX real-model Foundation examples",
    .serialized,
    .disabled(
        if: !MLXRealModelEnvironment.isEnabled,
        "Set MLX_RUN_REAL_MODEL_TESTS=1 and download models with scripts/download-test-models.sh"
    )
)
struct MLXRealModelFoundationExamplesTests {
    @Test("Streaming chat example emits real tokens")
    func streamingChatExampleEmitsRealTokens() async throws {
        guard let observed = try await Self.run(FoundationModelPlaygroundExamples.streamingChat) else {
            return
        }

        Self.verifyRealTokenOutput(observed)
        #expect(!observed.result.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
    }

    @Test("Apple Trip Planner guided generation emits constrained JSON")
    func appleTripPlannerGuidedGenerationEmitsConstrainedJSON() async throws {
        guard let observed = try await Self.run(
            FoundationModelPlaygroundExamples.tripPlannerGuidedGeneration
        ) else {
            return
        }
        let json = try Self.extractJSONObject(from: observed.result.text)
        let day = try #require(json["day"] as? [String: Any])

        Self.verifyRealTokenOutput(observed)
        Self.verifySuccessfulGrammarDiagnostics(
            MLXRealModelHarness.grammarSnapshots(from: observed.events),
            kind: .jsonSchema
        )
        #expect(json["destinationName"] as? String == "Yosemite")
        #expect(day["title"] is String)
        #expect(day["activity"] is String)
        #expect(["sightseeing", "foodAndDining", "hotelAndLodging"].contains(day["activityKind"] as? String))
    }

    @Test("Apple tool-calling example emits parseable tool call")
    func appleToolCallingExampleEmitsParseableToolCall() async throws {
        guard let observed = try await Self.run(
            FoundationModelPlaygroundExamples.pointsOfInterestToolCalling
        ) else {
            return
        }
        let call = try #require(MLXToolCallExtractor.extract(from: observed.result.text))

        Self.verifyRealTokenOutput(observed)
        Self.verifySuccessfulGrammarDiagnostics(
            MLXRealModelHarness.grammarSnapshots(from: observed.events),
            kind: .jsonSchema
        )
        #expect(call.name == "findPointsOfInterest")
        #expect(call.argumentsJSON.contains("hotel"))
        #expect(call.argumentsJSON.contains("hotel near Yosemite"))
    }

    @Test("Apple finite-choice example only emits an allowed choice")
    func appleFiniteChoiceExampleOnlyEmitsAllowedChoice() async throws {
        guard let observed = try await Self.run(
            FoundationModelPlaygroundExamples.finiteChoiceGuidedGeneration
        ) else {
            return
        }
        let choice = observed.result.text.trimmingCharacters(in: .whitespacesAndNewlines)

        Self.verifyRealTokenOutput(observed)
        Self.verifySuccessfulGrammarDiagnostics(
            MLXRealModelHarness.grammarSnapshots(from: observed.events),
            kind: .choices
        )
        #expect(FoundationModelPlaygroundExamples.fruitChoices.contains(choice))
    }

    @Test("Apple content-tagging example emits constrained tags")
    func appleContentTaggingExampleEmitsConstrainedTags() async throws {
        guard let observed = try await Self.run(FoundationModelPlaygroundExamples.contentTagging) else {
            return
        }
        let json = try Self.extractJSONObject(from: observed.result.text)
        let tags = try #require(json["tags"] as? [String])

        Self.verifyRealTokenOutput(observed)
        Self.verifySuccessfulGrammarDiagnostics(
            MLXRealModelHarness.grammarSnapshots(from: observed.events),
            kind: .jsonSchema
        )
        #expect(tags.count == 3)
        #expect(tags.allSatisfy { !$0.isEmpty })
    }

    private static func run(
        _ example: FoundationModelPlaygroundExample
    ) async throws -> (
        result: MLXRealModelHarness.GenerationResult,
        events: [MLXGenerationDiagnosticEvent]
    )? {
        guard let model = try await selectedModel() else {
            return nil
        }
        let modelStyle = try MLXRealModelHarness.inferredPromptStyle(for: model)
        let rendered = MLXPromptRenderer.render(
            example.request,
            style: example.resolvedStyle(modelDefault: modelStyle)
        )
        let input = LLMInput(
            context: rendered.prompt,
            promptMetadata: PromptRenderMetadata(rendererID: rendered.rendererID),
            promptCacheIdentity: PromptCacheIdentity(stableFingerprint: rendered.cacheFingerprint),
            sampling: example.sampling,
            limits: example.limits
        )
        return try await MLXGenerationDiagnostics.withRecording {
            try await MLXRealModelHarness.run(model: model, input: input)
        }
    }

    private static func selectedModel() async throws -> MLXRealModelCatalog.Model? {
        let models = try MLXRealModelCatalog.load()
        return try MLXRealModelHarness.selectedModel("qwen3-0.6b-4bit", in: models)
    }

    private static func verifyRealTokenOutput(
        _ observed: (
            result: MLXRealModelHarness.GenerationResult,
            events: [MLXGenerationDiagnosticEvent]
        )
    ) {
        let tokens = MLXRealModelHarness.generatedTokenSnapshots(from: observed.events)
        let tokenSummary = tokens
            .map { "index=\($0.index) id=\($0.tokenID) text=\($0.tokenText.debugDescription)" }
            .joined(separator: "\n")
        let comment = Comment(rawValue: tokenSummary)

        MLXRealModelHarness.verifyGenerated(observed.result)
        MLXRealModelHarness.verifyGeneratedTokenDiagnostics(tokens, result: observed.result)
        #expect(tokens.contains { !$0.tokenText.isEmpty }, comment)
    }

    private static func extractJSONObject(from text: String) throws -> [String: Any] {
        let jsonText = try #require(MLXJSONTextExtractor.firstJSONObject(in: text))
        let data = try #require(jsonText.data(using: .utf8))
        return try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
    }

    private static func verifySuccessfulGrammarDiagnostics(
        _ events: [MLXGrammarConstraintSnapshot],
        kind: GrammarConstraintKind
    ) {
        let summary = Comment(rawValue: grammarEventSummary(events))
        #expect(events.contains { $0.stage == .matcherPrepared && $0.kind == kind }, summary)
        #expect(events.contains { $0.stage == .maskApplied && $0.kind == kind }, summary)
        #expect(events.contains { $0.stage == .mlxMaskPrepared && $0.kind == kind }, summary)
        #expect(events.contains { $0.stage == .tokenAccepted && $0.kind == kind }, summary)
        #expect(!events.contains { $0.stage == .tokenRejected }, summary)
        #expect(!events.contains { $0.stage == .processorFailedClosed }, summary)
    }

    private static func grammarEventSummary(_ events: [MLXGrammarConstraintSnapshot]) -> String {
        events
            .map { event in
                [
                    "stage=\(event.stage.rawValue)",
                    "kind=\(event.kind.map(String.init(describing:)) ?? "nil")",
                    "mode=\(event.mode.map(String.init(describing:)) ?? "nil")",
                    "tokenCount=\(event.tokenCount.map(String.init) ?? "nil")",
                    "tokenID=\(event.tokenID.map(String.init) ?? "nil")",
                    "vocabularySize=\(event.vocabularySize.map(String.init) ?? "nil")",
                    "message=\(event.message ?? "nil")"
                ].joined(separator: " ")
            }
            .joined(separator: "\n")
    }
}

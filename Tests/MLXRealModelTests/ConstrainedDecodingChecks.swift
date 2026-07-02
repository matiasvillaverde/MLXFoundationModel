import Foundation
import MLXFoundationModel
@testable import MLXLocalModels
import Testing

enum ConstrainedDecodingChecks {
    static func verifyJSONSchemaConstraints(
        model: MLXRealModelCatalog.Model
    ) async throws {
        let observed = try await MLXRealModelHarness.runWithDiagnostics(
            model: model,
            sampling: Self.schemaSampling,
            limits: ResourceLimits(
                maxTokens: 160,
                maxTime: .seconds(MLXRealModelEnvironment.architectureGenerationTimeoutSeconds),
                reusePromptCache: false
            ),
            prompt: """
            /no_think
            Return only this compact JSON object, with no spaces or Markdown:
            {"city":"Berlin","celsius":21}
            """
        )
        let parameters = try MLXRealModelHarness.parameterSnapshot(from: observed.events)
        let grammarEvents = MLXRealModelHarness.grammarSnapshots(from: observed.events)
        let tokenEvents = MLXRealModelHarness.generatedTokenSnapshots(from: observed.events)

        MLXRealModelHarness.verifyGenerated(observed.result)
        MLXRealModelHarness.verifyGeneratedTokenDiagnostics(tokenEvents, result: observed.result)
        let json = try Self.extractJSONObject(from: observed.result.text)
        #expect(parameters.grammarKind == .jsonSchema)
        Self.verifySuccessfulGrammarDiagnostics(grammarEvents, kind: .jsonSchema)
        #expect(json["city"] as? String == "Berlin")
        #expect(json["celsius"] as? Int == 21)
    }

    static func verifyFiniteChoiceConstraints(
        model: MLXRealModelCatalog.Model
    ) async throws {
        let observed = try await MLXRealModelHarness.runWithDiagnostics(
            model: model,
            sampling: Self.fruitChoiceSampling,
            limits: ResourceLimits(
                maxTokens: 8,
                maxTime: .seconds(MLXRealModelEnvironment.architectureGenerationTimeoutSeconds),
                reusePromptCache: false
            ),
            prompt: "/no_think\nDo not choose a fruit. Write the word orange."
        )
        let text = observed.result.text.trimmingCharacters(in: .whitespacesAndNewlines)
        let grammarEvents = MLXRealModelHarness.grammarSnapshots(from: observed.events)
        let tokenEvents = MLXRealModelHarness.generatedTokenSnapshots(from: observed.events)

        MLXRealModelHarness.verifyGenerated(observed.result)
        MLXRealModelHarness.verifyGeneratedTokenDiagnostics(tokenEvents, result: observed.result)
        Self.verifySuccessfulGrammarDiagnostics(grammarEvents, kind: .choices)
        #expect(Self.fruitChoices.contains(text), "Expected a constrained fruit choice, got \(text)")
    }

    static func verifyEBNFConstraint(model: MLXRealModelCatalog.Model) async throws {
        let observed = try await MLXRealModelHarness.runWithDiagnostics(
            model: model,
            sampling: Self.openBraceSampling,
            limits: ResourceLimits(maxTokens: 1, maxTime: .seconds(120), reusePromptCache: false),
            prompt: "Do not output JSON. Say hello in plain English."
        )
        let grammarEvents = MLXRealModelHarness.grammarSnapshots(from: observed.events)
        let tokenEvents = MLXRealModelHarness.generatedTokenSnapshots(from: observed.events)

        MLXRealModelHarness.verifyGenerated(observed.result)
        MLXRealModelHarness.verifyGeneratedTokenDiagnostics(tokenEvents, result: observed.result)
        Self.verifySuccessfulGrammarDiagnostics(grammarEvents, kind: .ebnf)
        #expect(tokenEvents.first?.tokenText.contains("{") == true)
        #expect(
            observed.result.text == "{",
            """
            Expected \(model.id) to emit the exact grammar-forced token, \
            got \(observed.result.text.debugDescription)
            """
        )
    }

    static func verifyRegexConstraint(model: MLXRealModelCatalog.Model) async throws {
        let observed = try await MLXRealModelHarness.runWithDiagnostics(
            model: model,
            sampling: Self.regexSampling,
            limits: ResourceLimits(maxTokens: 4, maxTime: .seconds(120), reusePromptCache: false),
            prompt: "Do not write a color. Reply with a city name."
        )
        let grammarEvents = MLXRealModelHarness.grammarSnapshots(from: observed.events)
        let tokenEvents = MLXRealModelHarness.generatedTokenSnapshots(from: observed.events)
        let text = observed.result.text.trimmingCharacters(in: .whitespacesAndNewlines)

        MLXRealModelHarness.verifyGenerated(observed.result)
        MLXRealModelHarness.verifyGeneratedTokenDiagnostics(tokenEvents, result: observed.result)
        Self.verifySuccessfulGrammarDiagnostics(grammarEvents, kind: .regex)
        #expect(["red", "blue"].contains(text), "Expected regex-constrained color, got \(text)")
    }

    static func verifyBuiltinJSONConstraint(model: MLXRealModelCatalog.Model) async throws {
        let observed = try await MLXRealModelHarness.runWithDiagnostics(
            model: model,
            sampling: Self.jsonSampling,
            limits: ResourceLimits(maxTokens: 4, maxTime: .seconds(120), reusePromptCache: false),
            prompt: "Return one compact JSON value now."
        )
        let grammarEvents = MLXRealModelHarness.grammarSnapshots(from: observed.events)
        let tokenEvents = MLXRealModelHarness.generatedTokenSnapshots(from: observed.events)
        let text = observed.result.text.trimmingCharacters(in: .whitespacesAndNewlines)

        MLXRealModelHarness.verifyGenerated(observed.result)
        MLXRealModelHarness.verifyGeneratedTokenDiagnostics(tokenEvents, result: observed.result)
        Self.verifySuccessfulGrammarDiagnostics(grammarEvents, kind: .builtinJSON)
        #expect(Self.startsLikeJSONValue(text), "Expected JSON-constrained text, got \(text)")
    }

    private static func extractJSONObject(from text: String) throws -> [String: Any] {
        let jsonText = try #require(MLXJSONTextExtractor.firstJSONObject(in: text))
        let data = try #require(jsonText.data(using: .utf8))
        return try #require(
            JSONSerialization.jsonObject(with: data) as? [String: Any]
        )
    }

    private static var schemaSampling: SamplingParameters {
        SamplingParameters(
            temperature: 0,
            topP: 1,
            topK: 1,
            seed: 42,
            advanced: AdvancedSamplingParameters(grammar: .jsonSchema(weatherSchema))
        )
    }

    private static var openBraceSampling: SamplingParameters {
        SamplingParameters(
            temperature: 0,
            topP: 1,
            topK: 1,
            seed: 42,
            advanced: AdvancedSamplingParameters(
                grammar: GrammarSamplingConfiguration(grammar: #"root ::= "{""#)
            )
        )
    }

    private static var fruitChoiceSampling: SamplingParameters {
        SamplingParameters(
            temperature: 0,
            topP: 1,
            topK: 1,
            seed: 42,
            advanced: AdvancedSamplingParameters(grammar: .choices(fruitChoices))
        )
    }

    private static var regexSampling: SamplingParameters {
        SamplingParameters(
            temperature: 0,
            topP: 1,
            topK: 1,
            seed: 42,
            advanced: AdvancedSamplingParameters(grammar: .regex("red|blue"))
        )
    }

    private static var jsonSampling: SamplingParameters {
        SamplingParameters(
            temperature: 0,
            topP: 1,
            topK: 1,
            seed: 42,
            advanced: AdvancedSamplingParameters(grammar: .json())
        )
    }

    private static func startsLikeJSONValue(_ text: String) -> Bool {
        guard let first = text.first else {
            return false
        }
        return first == "{"
            || first == "["
            || first == "\""
            || first == "t"
            || first == "f"
            || first == "n"
            || first == "-"
            || first.isNumber
    }

    private static let fruitChoices = [
        "apple",
        "pear",
        "banana"
    ]

    private static let weatherSchema = """
    {"type":"object","properties":{"city":{"enum":["Berlin"]},"celsius":{"enum":[21]}},\
    "required":["city","celsius"],"additionalProperties":false}
    """

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
                    "bitmaskSize=\(event.bitmaskSize.map(String.init) ?? "nil")",
                    "completed=\(event.isCompleted.map(String.init) ?? "nil")",
                    "terminated=\(event.isTerminated.map(String.init) ?? "nil")",
                    "message=\(event.message ?? "nil")"
                ].joined(separator: " ")
            }
            .joined(separator: "\n")
    }
}

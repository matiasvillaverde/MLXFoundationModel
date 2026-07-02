import Foundation
import MLXFoundationModel
@testable import MLXLocalModels
import Testing

@Suite(
    "MLX real-model constrained decoding",
    .serialized,
    .disabled(
        if: !MLXRealModelEnvironment.isEnabled,
        "Set MLX_RUN_REAL_MODEL_TESTS=1 and download models with scripts/download-test-models.sh"
    )
)
struct MLXRealModelConstrainedDecodingTests {
    @Test("selected models generate valid JSON through token-level schema constraints")
    func selectedModelsGenerateValidJSONThroughTokenLevelSchemaConstraints() async throws {
        let models = try MLXRealModelCatalog.load()
        let selected = MLXRealModelEnvironment.selectedModels(from: models)
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
        for model in selected {
            do {
                try await ConstrainedDecodingChecks.verifyJSONSchemaConstraints(model: model)
            } catch {
                failures.append("\(model.id): \(error)")
            }
        }
        #expect(failures.isEmpty, Comment(rawValue: failures.joined(separator: "\n")))
    }

    @Test("selected models generate only one finite-choice token sequence")
    func selectedModelsGenerateOnlyOneFiniteChoiceTokenSequence() async throws {
        let models = try MLXRealModelCatalog.load()
        let selected = MLXRealModelEnvironment.selectedModels(from: models)
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
        for model in selected {
            do {
                try await ConstrainedDecodingChecks.verifyFiniteChoiceConstraints(model: model)
            } catch {
                failures.append("\(model.id): \(error)")
            }
        }
        #expect(failures.isEmpty, Comment(rawValue: failures.joined(separator: "\n")))
    }

    @Test("Qwen3 generates valid JSON through token-level schema constraints")
    func qwen3GeneratesValidJSONThroughTokenLevelSchemaConstraints() async throws {
        let models = try MLXRealModelCatalog.load()
        guard let model = try MLXRealModelHarness.selectedModel("qwen3-0.6b-4bit", in: models) else {
            return
        }
        try await ConstrainedDecodingChecks.verifyJSONSchemaConstraints(model: model)
    }

    @Test("Qwen3 generates only one finite-choice token sequence")
    func qwen3GeneratesOnlyOneFiniteChoiceTokenSequence() async throws {
        let models = try MLXRealModelCatalog.load()
        guard let model = try MLXRealModelHarness.selectedModel("qwen3-0.6b-4bit", in: models) else {
            return
        }
        try await ConstrainedDecodingChecks.verifyFiniteChoiceConstraints(model: model)
    }

    @Test("Selected architectures force grammar-valid token sequences")
    func selectedArchitecturesForceGrammarValidTokenSequences() async throws {
        let models = MLXRealModelEnvironment.selectedModels(from: try MLXRealModelCatalog.load())
        let missingModels = models.filter { !MLXRealModelEnvironment.hasModelFiles(for: $0) }
        #expect(!models.isEmpty)
        #expect(
            missingModels.isEmpty,
            Comment(rawValue: MLXRealModelEnvironment.missingModelsMessage(missingModels))
        )

        for model in models where MLXRealModelEnvironment.hasModelFiles(for: model) {
            try await ConstrainedDecodingChecks.verifyEBNFConstraint(model: model)
            try await ConstrainedDecodingChecks.verifyBuiltinJSONConstraint(model: model)
            try await ConstrainedDecodingChecks.verifyRegexConstraint(model: model)
        }
    }
}

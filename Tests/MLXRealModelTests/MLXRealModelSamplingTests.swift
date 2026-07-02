@testable import MLXLocalModels
import Testing

@Suite(
    "MLX real-model sampling controls",
    .serialized,
    .disabled(
        if: !MLXRealModelEnvironment.isEnabled,
        "Set MLX_RUN_REAL_MODEL_TESTS=1 and download models with scripts/download-test-models.sh"
    )
)
struct MLXRealModelSamplingTests {
    @Test("selected catalog models apply sampling and logits controls")
    func selectedCatalogModelsApplySamplingAndLogitsControls() async throws {
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
                try await Self.verifySamplingAndLogitsControls(
                    model: model,
                    maxTokens: min(
                        model.maxTokens,
                        MLXRealModelEnvironment.architectureGenerationTokenLimit
                    )
                )
            } catch {
                failures.append("\(model.id): \(error)")
            }
        }
        #expect(failures.isEmpty, Comment(rawValue: failures.joined(separator: "\n")))
    }

    @Test("Qwen3 generation applies sampling and logits controls")
    func qwen3GenerationAppliesSamplingAndLogitsControls() async throws {
        let models = try MLXRealModelCatalog.load()
        guard let model = try MLXRealModelHarness.selectedModel("qwen3-0.6b-4bit", in: models) else {
            return
        }
        try await Self.verifySamplingAndLogitsControls(model: model, maxTokens: 10)
    }

    @Test("selected catalog models apply advanced sampling controls")
    func selectedCatalogModelsApplyAdvancedSamplingControls() async throws {
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
                try await Self.verifyAdvancedSamplingControls(model: model)
            } catch {
                failures.append("\(model.id): \(error)")
            }
        }
        #expect(failures.isEmpty, Comment(rawValue: failures.joined(separator: "\n")))
    }

    private static func verifySamplingAndLogitsControls(
        model: MLXRealModelCatalog.Model,
        maxTokens: Int
    ) async throws {
        let observed = try await MLXRealModelHarness.runWithDiagnostics(
            model: model,
            sampling: Self.samplingControls,
            limits: ResourceLimits(
                maxTokens: maxTokens,
                maxTime: .seconds(MLXRealModelEnvironment.architectureGenerationTimeoutSeconds),
                reusePromptCache: false
            )
        )
        let parameters = try MLXRealModelHarness.parameterSnapshot(from: observed.events)
        let tokenEvents = MLXRealModelHarness.generatedTokenSnapshots(from: observed.events)

        MLXRealModelHarness.verifyGenerated(observed.result)
        MLXRealModelHarness.verifyGeneratedTokenDiagnostics(tokenEvents, result: observed.result)
        #expect(parameters.temperature == 0.7)
        #expect(parameters.topP == 0.85)
        #expect(parameters.topK == 8)
        #expect(parameters.minP == 0.01)
        #expect(parameters.repetitionPenalty == 1.05)
        #expect(parameters.repetitionContextSize == 16)
        #expect(parameters.presencePenalty == 0.10)
        #expect(parameters.frequencyPenalty == 0.10)
        #expect(parameters.seed == 1_234)
        #expect(parameters.logitBiasCount == 1)
    }

    private static func verifyAdvancedSamplingControls(
        model: MLXRealModelCatalog.Model
    ) async throws {
        try await Self.verifyMirostatControls(model: model)
        try await Self.verifyDRYAndAdaptiveP(model: model)
    }

    private static func verifyMirostatControls(
        model: MLXRealModelCatalog.Model
    ) async throws {
        let observed = try await Self.runAdvancedSampling(
            model: model,
            sampling: Self.mirostatControls
        )
        let parameters = try MLXRealModelHarness.parameterSnapshot(from: observed.events)
        let tokenEvents = MLXRealModelHarness.generatedTokenSnapshots(from: observed.events)

        MLXRealModelHarness.verifyGenerated(observed.result)
        MLXRealModelHarness.verifyGeneratedTokenDiagnostics(tokenEvents, result: observed.result)
        #expect(parameters.temperature == 0.7)
        #expect(parameters.mirostatVersion == .v2)
        #expect(parameters.mirostatTau == 4.5)
        #expect(parameters.mirostatEta == 0.2)
        #expect(parameters.mirostatLearningTokens == 64)
    }

    private static func verifyDRYAndAdaptiveP(
        model: MLXRealModelCatalog.Model
    ) async throws {
        let observed = try await Self.runAdvancedSampling(
            model: model,
            sampling: Self.dryAdaptivePControls
        )
        let parameters = try MLXRealModelHarness.parameterSnapshot(from: observed.events)
        let tokenEvents = MLXRealModelHarness.generatedTokenSnapshots(from: observed.events)

        MLXRealModelHarness.verifyGenerated(observed.result)
        MLXRealModelHarness.verifyGeneratedTokenDiagnostics(tokenEvents, result: observed.result)
        #expect(parameters.temperature == 0.8)
        #expect(parameters.typicalP == 0.8)
        #expect(parameters.topNSigma == 1.25)
        #expect(parameters.xtcProbability == 0.42)
        #expect(parameters.xtcThreshold == 0.2)
        #expect(parameters.xtcMinKeep == 2)
        #expect(parameters.xtcProtectedTokenCount > 0)
        #expect(parameters.dryMultiplier == 1.5)
        #expect(parameters.dryBase == 1.6)
        #expect(parameters.dryAllowedLength == 3)
        #expect(parameters.dryPenaltyLastTokens == 64)
        #expect(parameters.drySequenceBreakerCount > 0)
        #expect(parameters.adaptivePTarget == 0.2)
        #expect(parameters.adaptivePDecay == 0.8)
    }

    private static func runAdvancedSampling(
        model: MLXRealModelCatalog.Model,
        sampling: SamplingParameters
    ) async throws -> (result: MLXRealModelHarness.GenerationResult, events: [MLXGenerationDiagnosticEvent]) {
        try await MLXRealModelHarness.runWithDiagnostics(
            model: model,
            sampling: sampling,
            limits: ResourceLimits(
                maxTokens: min(model.maxTokens, 4),
                maxTime: .seconds(MLXRealModelEnvironment.architectureGenerationTimeoutSeconds),
                reusePromptCache: false
            ),
            prompt: "Write a short color name."
        )
    }

    private static var samplingControls: SamplingParameters {
        SamplingParameters(
            temperature: 0.7,
            topP: 0.85,
            topK: 8,
            repetitionPenalty: 1.05,
            frequencyPenalty: 0.10,
            presencePenalty: 0.10,
            repetitionPenaltyRange: 16,
            seed: 1_234,
            advanced: AdvancedSamplingParameters(
                minP: 0.01,
                logitBias: [0: -1.0]
            )
        )
    }

    private static var mirostatControls: SamplingParameters {
        SamplingParameters(
            temperature: 0.7,
            topP: 1.0,
            seed: 42,
            advanced: AdvancedSamplingParameters(
                mirostat: MirostatSamplingConfiguration(
                    version: .v2,
                    tau: 4.5,
                    eta: 0.2,
                    learningTokens: 64
                )
            )
        )
    }

    private static var dryAdaptivePControls: SamplingParameters {
        SamplingParameters(
            temperature: 0.8,
            topP: 0.95,
            seed: 84,
            advanced: AdvancedSamplingParameters(
                typicalP: 0.8,
                topNSigma: 1.25,
                xtc: XtcSamplingConfiguration(
                    probability: 0.42,
                    threshold: 0.2,
                    minKeep: 2
                ),
                dry: DrySamplingConfiguration(
                    multiplier: 1.5,
                    base: 1.6,
                    allowedLength: 3,
                    penaltyLastTokens: 64,
                    sequenceBreakers: ["\n"]
                ),
                adaptiveP: AdaptivePSamplingConfiguration(target: 0.2, decay: 0.8)
            )
        )
    }
}

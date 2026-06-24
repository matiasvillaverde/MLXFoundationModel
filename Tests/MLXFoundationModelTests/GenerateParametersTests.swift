@testable import MLXLocalModels
import Testing

@Suite("Generate parameters")
struct GenerateParametersTests {
    @Test("normalizes invalid scalar and token options")
    func normalizesInvalidScalarAndTokenOptions() {
        let parameters = GenerateParameters(
            maxTokens: -3,
            maxKVSize: 0,
            kvGroupSize: 0,
            topK: -8,
            xtcProtectedTokenIds: [-1, 2],
            logitBias: [-1: 3, 2: 4],
            reasoningBudgetTokens: 0,
            reasoningEndTokenIds: [-1, 7],
            suppressTokenIds: [-2, 5],
            prefillStepSize: 0
        )

        #expect(parameters.maxTokens == 0)
        #expect(parameters.maxKVSize == 1)
        #expect(parameters.kvGroupSize == 1)
        #expect(parameters.topK == 0)
        #expect(parameters.xtcProtectedTokenIds == [2])
        #expect(parameters.logitBias == [2: Float(4)])
        #expect(parameters.reasoningBudgetTokens == 1)
        #expect(parameters.reasoningEndTokenIds == [7])
        #expect(parameters.suppressTokenIds == [5])
        #expect(parameters.prefillStepSize == 1)
    }

    @Test("selects sampler from normalized generation options")
    func selectsSamplerFromNormalizedGenerationOptions() {
        #expect(GenerateParameters(temperature: 0).sampler() is ArgMaxSampler)
        #expect(GenerateParameters(temperature: 0.7).sampler() is CategoricalSampler)
        #expect(GenerateParameters(temperature: 0.7, topK: 3).sampler() is TopPSampler)
        #expect(GenerateParameters(
            temperature: 0.7,
            mirostat: MirostatSamplingConfiguration(version: .v2)
        ).sampler() is MirostatSampler)
        #expect(GenerateParameters(
            temperature: 0.7,
            adaptiveP: AdaptivePSamplingConfiguration(target: 0.2)
        ).sampler() is AdaptivePSampler)
    }

    @Test("zero temperature keeps greedy sampling above adaptive samplers")
    func zeroTemperatureKeepsGreedySamplingAboveAdaptiveSamplers() {
        let parameters = GenerateParameters(
            temperature: 0,
            mirostat: MirostatSamplingConfiguration(version: .v2),
            adaptiveP: AdaptivePSamplingConfiguration(target: 0.2)
        )

        #expect(!parameters.usesMirostatSampler)
        #expect(!parameters.usesAdaptivePSampler)
        #expect(parameters.sampler() is ArgMaxSampler)
    }

    @Test("builds processor only when at least one processor is active")
    func buildsProcessorOnlyWhenAtLeastOneProcessorIsActive() throws {
        #expect(try GenerateParameters().processor() == nil)

        let processor = try #require(try GenerateParameters(
            logitBias: [-1: 1, 2: 0.5],
            reasoningBudgetTokens: 3,
            reasoningEndTokenIds: [-1, 6],
            suppressTokenIds: [-2, 4]
        ).processor())
        let penalty = try #require(processor as? PenaltyProcessor)

        #expect(penalty.logitBiasContext != nil)
        #expect(penalty.suppressTokenContext != nil)
        #expect(penalty.reasoningBudgetContext != nil)
        #expect(penalty.repetitionContext == nil)
        #expect(penalty.presenceContext == nil)
        #expect(penalty.frequencyContext == nil)
        #expect(penalty.dryContext == nil)
        #expect(penalty.grammarContext == nil)
    }
}

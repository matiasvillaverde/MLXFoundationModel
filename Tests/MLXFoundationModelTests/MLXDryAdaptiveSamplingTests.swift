import Foundation
import MLX
@testable import MLXLocalModels
import Testing

@Suite("MLX DRY and adaptive-p sampling")
struct MLXDryAdaptiveSamplingTests {
    @Test("generation parameters carry DRY and adaptive-p from public sampling options")
    func generationParametersCarryDryAndAdaptivePFromPublicSamplingOptions() async {
        let session = MLXSession()
        let sampling = SamplingParameters(
            temperature: 0.8,
            topP: 0.95,
            advanced: AdvancedSamplingParameters(
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

        let parameters = await session.createGenerateParameters(
            from: sampling,
            limits: ResourceLimits(maxTokens: 8),
            drySequenceBreakerTokenIds: [[13]]
        )

        #expect(parameters.dry?.multiplier == 1.5)
        #expect(parameters.dry?.base == 1.6)
        #expect(parameters.dry?.allowedLength == 3)
        #expect(parameters.dry?.penaltyLastTokens == 64)
        #expect(parameters.drySequenceBreakerTokenIds == [[13]])
        #expect(parameters.adaptiveP?.target == 0.2)
        #expect(parameters.adaptiveP?.decay == 0.8)
        #expect(parameters.usesAdaptivePSampler)
    }

    @Test("generation parameters normalize invalid DRY and adaptive-p values")
    func generationParametersNormalizeInvalidDryAndAdaptivePValues() {
        let disabledDry = GenerateParameters(
            dry: DrySamplingConfiguration(
                multiplier: 0,
                base: 1.75,
                allowedLength: 2,
                penaltyLastTokens: 64
            )
        )
        let clampedAdaptiveP = GenerateParameters(
            adaptiveP: AdaptivePSamplingConfiguration(target: 2.0, decay: 4.0)
        )
        let disabledAdaptiveP = GenerateParameters(
            adaptiveP: AdaptivePSamplingConfiguration(target: -1.0, decay: 0.5)
        )

        #expect(disabledDry.dry == nil)
        #expect(clampedAdaptiveP.adaptiveP?.target == 1.0)
        #expect(clampedAdaptiveP.adaptiveP?.decay == 0.99)
        #expect(disabledAdaptiveP.adaptiveP == nil)
    }

    @Test("diagnostics include DRY and adaptive-p sampler state")
    func diagnosticsIncludeDryAndAdaptivePSamplerState() async throws {
        let parameters = GenerateParameters(
            dry: DrySamplingConfiguration(
                multiplier: 1.25,
                base: 1.5,
                allowedLength: 4,
                penaltyLastTokens: 48
            ),
            drySequenceBreakerTokenIds: [[10], [20, 21]],
            adaptiveP: AdaptivePSamplingConfiguration(target: 0.15, decay: 0.7)
        )

        let recorded = try await MLXGenerationDiagnostics.withRecording {
            MLXGenerationDiagnostics.recordParameters(parameters)
        }
        let snapshot = try #require(recorded.events.compactMap { event in
            if case .parameters(let snapshot) = event {
                return snapshot
            }
            return nil
        }.first)

        #expect(snapshot.dryMultiplier == 1.25)
        #expect(snapshot.dryBase == 1.5)
        #expect(snapshot.dryAllowedLength == 4)
        #expect(snapshot.dryPenaltyLastTokens == 48)
        #expect(snapshot.drySequenceBreakerCount == 2)
        #expect(snapshot.adaptivePTarget == 0.15)
        #expect(snapshot.adaptivePDecay == 0.7)
    }

    @Test(
        "DRY penalizes repeated sequence continuations",
        .disabled(
            if: ProcessInfo.processInfo.environment["MLX_RUN_ARRAY_UNIT_TESTS"] != "1",
            "Set MLX_RUN_ARRAY_UNIT_TESTS=1 on a launcher with MLX Metal resources available"
        )
    )
    func dryPenalizesRepeatedSequenceContinuations() {
        let values = Device.withDefaultDevice(.cpu) {
            var processor = DryPenaltyContext(
                configuration: DrySamplingConfiguration(
                    multiplier: 100,
                    base: 2,
                    allowedLength: 2,
                    penaltyLastTokens: -1
                ),
                totalContextSize: 16,
                sequenceBreakers: []
            )
            processor.prompt(MLXArray([Int32(0), 1, 2, 0, 1, 2]))

            let logits = MLXArray([Float(10), 0, 0]).reshaped(1, 3)
            let processed = processor.process(logits: logits)
            eval(processed)
            return processed.asArray(Float.self)
        }

        #expect(values == [-190, 0, 0])
    }

    @Test(
        "DRY keeps single-token sequence breakers unpenalized",
        .disabled(
            if: ProcessInfo.processInfo.environment["MLX_RUN_ARRAY_UNIT_TESTS"] != "1",
            "Set MLX_RUN_ARRAY_UNIT_TESTS=1 on a launcher with MLX Metal resources available"
        )
    )
    func dryKeepsSingleTokenSequenceBreakersUnpenalized() {
        let values = Device.withDefaultDevice(.cpu) {
            var processor = DryPenaltyContext(
                configuration: DrySamplingConfiguration(
                    multiplier: 100,
                    base: 2,
                    allowedLength: 2,
                    penaltyLastTokens: -1
                ),
                totalContextSize: 16,
                sequenceBreakers: [[0]]
            )
            processor.prompt(MLXArray([Int32(0), 1, 2, 0, 1, 2]))

            let logits = MLXArray([Float(10), 0, 0]).reshaped(1, 3)
            let processed = processor.process(logits: logits)
            eval(processed)
            return processed.asArray(Float.self)
        }

        #expect(values == [10, 0, 0])
    }

    @Test(
        "adaptive-p samples near the target probability",
        .disabled(
            if: ProcessInfo.processInfo.environment["MLX_RUN_ARRAY_UNIT_TESTS"] != "1",
            "Set MLX_RUN_ARRAY_UNIT_TESTS=1 on a launcher with MLX Metal resources available"
        )
    )
    func adaptivePSamplesNearTheTargetProbability() {
        let token = Device.withDefaultDevice(.cpu) {
            let sampler = AdaptivePSampler(
                configuration: AdaptivePSamplingConfiguration(target: 0.2, decay: 0.8),
                temperature: 1.0,
                seed: 7
            )
            let logits = MLXArray([
                Float(log(0.79999)), Float(log(0.2)), Float(log(0.00001))
            ]).reshaped(1, 3)

            let sampled = sampler.sample(logits: logits)
            eval(sampled)
            return sampled.item(Int.self)
        }

        #expect(token == 1)
    }
}

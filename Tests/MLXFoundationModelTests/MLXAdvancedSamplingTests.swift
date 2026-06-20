import Foundation
import MLX
@testable import MLXLocalModels
import Testing

@Suite("MLX advanced sampling")
struct MLXAdvancedSamplingTests {
    @Test("generation parameters carry XTC from public sampling options")
    func generationParametersCarryXTCFromPublicSamplingOptions() async {
        let session = MLXSession()
        let sampling = SamplingParameters(
            temperature: 0.7,
            topP: 1.0,
            advanced: AdvancedSamplingParameters(
                typicalP: 0.8,
                topNSigma: 1.25,
                xtc: XtcSamplingConfiguration(
                    probability: 0.42,
                    threshold: 0.2,
                    minKeep: 2
                )
            )
        )

        let parameters = await session.createGenerateParameters(
            from: sampling,
            limits: ResourceLimits(maxTokens: 8),
            suppressTokenIds: [9],
            xtcProtectedTokenIds: [0, 1]
        )

        #expect(parameters.xtcProbability == 0.42)
        #expect(parameters.typicalP == 0.8)
        #expect(parameters.topNSigma == 1.25)
        #expect(parameters.xtcThreshold == 0.2)
        #expect(parameters.xtcMinKeep == 2)
        #expect(parameters.xtcProtectedTokenIds == [0, 1])
        #expect(parameters.suppressTokenIds == [9])
    }

    @Test("generation parameters normalize invalid XTC values")
    func generationParametersNormalizeInvalidXTCValues() {
        let parameters = GenerateParameters(
            typicalP: 3.0,
            topNSigma: -1.0,
            xtcProbability: 3.0,
            xtcThreshold: 0.9,
            xtcMinKeep: 0
        )

        #expect(parameters.typicalP == 1.0)
        #expect(parameters.topNSigma == nil)
        #expect(parameters.xtcProbability == 1.0)
        #expect(parameters.xtcThreshold == 0.5)
        #expect(parameters.xtcMinKeep == 1)
    }

    @Test("diagnostics include XTC sampler state")
    func diagnosticsIncludeXTCSamplerState() async throws {
        let parameters = GenerateParameters(
            typicalP: 0.6,
            topNSigma: 1.5,
            xtcProbability: 0.75,
            xtcThreshold: 0.15,
            xtcMinKeep: 3,
            xtcProtectedTokenIds: [0, 2, 4]
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

        #expect(snapshot.typicalP == 0.6)
        #expect(snapshot.topNSigma == 1.5)
        #expect(snapshot.xtcProbability == 0.75)
        #expect(snapshot.xtcThreshold == 0.15)
        #expect(snapshot.xtcMinKeep == 3)
        #expect(snapshot.xtcProtectedTokenCount == 3)
    }

    @Test(
        "typical-p filtering is applied before top-k",
        .disabled(
            if: ProcessInfo.processInfo.environment["MLX_RUN_ARRAY_UNIT_TESTS"] != "1",
            "Set MLX_RUN_ARRAY_UNIT_TESTS=1 on a launcher with MLX Metal resources available"
        )
    )
    func typicalPFilteringIsAppliedBeforeTopK() {
        let token = Device.withDefaultDevice(.cpu) {
            let sampler = TopPSampler(
                temperature: 1.0,
                topK: 1,
                typicalP: 0.01,
                seed: 7
            )
            let logits = MLXArray([
                Float(log(0.45)), Float(log(0.35)), Float(log(0.20))
            ]).reshaped([1, 3])

            let sampled = sampler.sample(logits: logits)
            eval(sampled)
            return sampled.item(Int.self)
        }

        #expect(token == 1)
    }

    @Test(
        "top-n-sigma filtering is applied before sampling",
        .disabled(
            if: ProcessInfo.processInfo.environment["MLX_RUN_ARRAY_UNIT_TESTS"] != "1",
            "Set MLX_RUN_ARRAY_UNIT_TESTS=1 on a launcher with MLX Metal resources available"
        )
    )
    func topNSigmaFilteringIsAppliedBeforeSampling() {
        let token = Device.withDefaultDevice(.cpu) {
            let sampler = TopPSampler(
                temperature: 1.0,
                topNSigma: 0.2,
                seed: 7
            )
            let logits = MLXArray([
                Float(5.0), Float(4.0), Float(0.0)
            ]).reshaped([1, 3])

            let sampled = sampler.sample(logits: logits)
            eval(sampled)
            return sampled.item(Int.self)
        }

        #expect(token == 0)
    }

    @Test(
        "XTC removes high probability tokens before top-k",
        .disabled(
            if: ProcessInfo.processInfo.environment["MLX_RUN_ARRAY_UNIT_TESTS"] != "1",
            "Set MLX_RUN_ARRAY_UNIT_TESTS=1 on a launcher with MLX Metal resources available"
        )
    )
    func xtcRemovesHighProbabilityTokensBeforeTopK() {
        let token = Device.withDefaultDevice(.cpu) {
            let sampler = TopPSampler(
                temperature: 1.0,
                topK: 1,
                xtcProbability: 1.0,
                xtcThreshold: 0.1,
                seed: 7
            )
            let logits = MLXArray([
                Float(8.0), Float(7.5), Float(7.0), Float(0.0)
            ]).reshaped([1, 4])

            let sampled = sampler.sample(logits: logits)
            eval(sampled)
            return sampled.item(Int.self)
        }

        #expect(token == 2)
    }

    @Test(
        "XTC honors min keep before top-k",
        .disabled(
            if: ProcessInfo.processInfo.environment["MLX_RUN_ARRAY_UNIT_TESTS"] != "1",
            "Set MLX_RUN_ARRAY_UNIT_TESTS=1 on a launcher with MLX Metal resources available"
        )
    )
    func xtcHonorsMinKeepBeforeTopK() {
        let token = Device.withDefaultDevice(.cpu) {
            let sampler = TopPSampler(
                temperature: 1.0,
                topK: 1,
                xtcProbability: 1.0,
                xtcThreshold: 0.1,
                xtcMinKeep: 2,
                seed: 7
            )
            let logits = MLXArray([
                Float(8.0), Float(7.5), Float(7.0), Float(0.0)
            ]).reshaped([1, 4])

            let sampled = sampler.sample(logits: logits)
            eval(sampled)
            return sampled.item(Int.self)
        }

        #expect(token == 1)
    }

    @Test(
        "XTC protects special tokens",
        .disabled(
            if: ProcessInfo.processInfo.environment["MLX_RUN_ARRAY_UNIT_TESTS"] != "1",
            "Set MLX_RUN_ARRAY_UNIT_TESTS=1 on a launcher with MLX Metal resources available"
        )
    )
    func xtcProtectsSpecialTokens() {
        let token = Device.withDefaultDevice(.cpu) {
            let sampler = TopPSampler(
                temperature: 1.0,
                topK: 1,
                xtcProbability: 1.0,
                xtcThreshold: 0.1,
                xtcProtectedTokenIds: [0],
                seed: 7
            )
            let logits = MLXArray([
                Float(8.0), Float(7.5), Float(7.0), Float(0.0)
            ]).reshaped([1, 4])

            let sampled = sampler.sample(logits: logits)
            eval(sampled)
            return sampled.item(Int.self)
        }

        #expect(token == 0)
    }
}

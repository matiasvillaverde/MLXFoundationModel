import Foundation
import MLX
@testable import MLXLocalModels
import Testing

@Suite("MLX Mirostat sampling")
struct MLXMirostatSamplingTests {
    @Test("generation parameters carry Mirostat from public sampling options")
    func generationParametersCarryMirostatFromPublicSamplingOptions() async {
        let session = MLXSession()
        let sampling = SamplingParameters(
            temperature: 0.7,
            topP: 1.0,
            advanced: AdvancedSamplingParameters(mirostat: Self.mirostat())
        )

        let parameters = await session.createGenerateParameters(
            from: sampling,
            limits: ResourceLimits(maxTokens: 8)
        )

        #expect(parameters.mirostat?.version == .v2)
        #expect(parameters.mirostat?.tau == 4.5)
        #expect(parameters.mirostat?.eta == 0.2)
        #expect(parameters.mirostat?.learningTokens == 64)
        #expect(parameters.usesMirostatSampler)
    }

    @Test("generation parameters reject invalid Mirostat values")
    func generationParametersRejectInvalidMirostatValues() {
        let parameters = GenerateParameters(
            mirostat: MirostatSamplingConfiguration(
                version: .v1,
                tau: 0,
                eta: -1,
                learningTokens: 1
            )
        )

        #expect(parameters.mirostat == nil)
        #expect(!parameters.usesMirostatSampler)
    }

    @Test("diagnostics include Mirostat sampler state")
    func diagnosticsIncludeMirostatSamplerState() async throws {
        let snapshot = try await Self.parameterSnapshot(for: GenerateParameters(
            mirostat: MirostatSamplingConfiguration(
                version: .v1,
                tau: 3.5,
                eta: 0.25,
                learningTokens: 48
            )
        ))

        #expect(snapshot.mirostatVersion == MirostatSamplingVersion.v1)
        #expect(snapshot.mirostatTau == 3.5)
        #expect(snapshot.mirostatEta == 0.25)
        #expect(snapshot.mirostatLearningTokens == 48)
    }

    @Test("zero temperature keeps greedy sampling despite Mirostat")
    func zeroTemperatureKeepsGreedySamplingDespiteMirostat() {
        let parameters = GenerateParameters(
            temperature: 0,
            mirostat: MirostatSamplingConfiguration(version: .v2)
        )

        #expect(!parameters.usesMirostatSampler)
    }

    @Test(
        "Mirostat v2 truncates candidates by target surprise",
        .disabled(
            if: ProcessInfo.processInfo.environment["MLX_RUN_ARRAY_UNIT_TESTS"] != "1",
            "Set MLX_RUN_ARRAY_UNIT_TESTS=1 on a launcher with MLX Metal resources available"
        )
    )
    func mirostatV2TruncatesCandidatesByTargetSurprise() {
        let token = Self.sample(
            configuration: .init(version: .v2, tau: 0.5, eta: 0.1),
            probabilities: [0.90, 0.09, 0.01]
        )

        #expect(token == 0)
    }

    @Test(
        "Mirostat v1 samples from adaptive top-k",
        .disabled(
            if: ProcessInfo.processInfo.environment["MLX_RUN_ARRAY_UNIT_TESTS"] != "1",
            "Set MLX_RUN_ARRAY_UNIT_TESTS=1 on a launcher with MLX Metal resources available"
        )
    )
    func mirostatV1SamplesFromAdaptiveTopK() {
        let token = Self.sample(
            configuration: .init(version: .v1, tau: 0.5, eta: 0.1, learningTokens: 3),
            probabilities: [0.98, 0.015, 0.005]
        )

        #expect(token == 0)
    }

    private static func mirostat() -> MirostatSamplingConfiguration {
        MirostatSamplingConfiguration(
            version: .v2,
            tau: 4.5,
            eta: 0.2,
            learningTokens: 64
        )
    }

    private static func parameterSnapshot(
        for parameters: GenerateParameters
    ) async throws -> MLXGenerationParameterSnapshot {
        let recorded = try await MLXGenerationDiagnostics.withRecording {
            MLXGenerationDiagnostics.recordParameters(parameters)
        }
        return try #require(recorded.events.compactMap { event in
            if case .parameters(let snapshot) = event {
                return snapshot
            }
            return nil
        }.first)
    }

    private static func sample(
        configuration: MirostatSamplingConfiguration,
        probabilities: [Float]
    ) -> Int {
        Device.withDefaultDevice(.cpu) {
            let sampler = MirostatSampler(
                configuration: configuration,
                temperature: 1.0,
                seed: 7
            )
            let logits = MLXArray(probabilities.map(log)).reshaped([1, probabilities.count])
            let sampled = sampler.sample(logits: logits)
            eval(sampled)
            return sampled.item(Int.self)
        }
    }
}

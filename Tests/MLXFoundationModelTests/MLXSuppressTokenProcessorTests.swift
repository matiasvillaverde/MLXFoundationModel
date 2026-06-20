import Foundation
import MLX
@testable import MLXLocalModels
import Testing

@Suite("MLX suppress token processing")
struct MLXSuppressTokenProcessorTests {
    @Test("decodes suppress tokens from generation config")
    func decodesSuppressTokensFromGenerationConfig() throws {
        let data = Data("""
        {
            "eos_token_id": [1, 2],
            "suppress_tokens": [3, 5, 8]
        }
        """.utf8)

        let config = try JSONDecoder.json5().decode(GenerationConfigFile.self, from: data)

        #expect(config.eosTokenIds?.values == [1, 2])
        #expect(config.suppressTokenIds?.values == [3, 5, 8])
    }

    @Test(
        "masks suppressed token logits before sampling",
        .disabled(
            if: ProcessInfo.processInfo.environment["MLX_RUN_ARRAY_UNIT_TESTS"] != "1",
            "Set MLX_RUN_ARRAY_UNIT_TESTS=1 on a launcher with MLX Metal resources available"
        )
    )
    func masksSuppressedTokenLogitsBeforeSampling() throws {
        let values = Device.withDefaultDevice(.cpu) {
            var processor = SuppressTokensProcessor(tokenIds: [1, 3])
            let logits = MLXArray([
                Float(0), Float(10), Float(20), Float(30),
                Float(40), Float(50), Float(60), Float(70)
            ]).reshaped([2, 4])

            let masked = processor.process(logits: logits)
            eval(masked)
            return masked.asArray(Float.self)
        }

        #expect(values == [
            0, -Float.infinity, 20, -Float.infinity,
            40, -Float.infinity, 60, -Float.infinity
        ])
    }

    @Test("parameters include model suppress token processor")
    func parametersIncludeModelSuppressTokenProcessor() async throws {
        let parameters = GenerateParameters(suppressTokenIds: [2, 5])

        let recorded = try await MLXGenerationDiagnostics.withRecording {
            MLXGenerationDiagnostics.recordParameters(parameters)
        }
        let snapshot = try #require(recorded.events.compactMap { event in
            if case .parameters(let snapshot) = event {
                return snapshot
            }
            return nil
        }.first)

        #expect(parameters.suppressTokenIds == [2, 5])
        #expect(snapshot.suppressTokenCount == 2)
    }
}

#if FOUNDATION_MODELS_PROVIDER_API && canImport(FoundationModels)
import Foundation
import FoundationModels
import MLXFoundationModel
import Testing

@Suite("Foundation Models session compatibility")
struct MLXSessionCompatibilityTests {
    @available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
    @Generable
    struct CompatibilityAnswer {
        let summary: String
    }

    @Test("MLXLanguageModel type-checks with Apple session overloads")
    func mlxLanguageModelTypeChecksWithAppleSessionOverloads() async throws {
        guard #available(macOS 27.0, iOS 27.0, visionOS 27.0, *) else {
            return
        }

        let session = LanguageModelSession(
            model: Self.compatibilityModel,
            instructions: "Answer briefly."
        )

        _ = session.streamResponse(
            to: "Write one sentence about local inference.",
            options: GenerationOptions(samplingMode: .greedy, maximumResponseTokens: 4)
        )
        _ = session.streamResponse(
            to: "Generate a summary.",
            generating: CompatibilityAnswer.self,
            options: GenerationOptions(samplingMode: .greedy, maximumResponseTokens: 16)
        )
    }

    @available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
    private static var compatibilityModel: MLXLanguageModel {
        MLXLanguageModel(
            model: MLXModel(
                id: "compatibility-fixture",
                location: URL(fileURLWithPath: "/tmp/MLXFoundationModelCompatibility"),
                promptStyle: .chatML,
                capabilities: MLXModelCapabilities(toolCalling: true, structuredOutput: true)
            )
        )
    }
}
#endif

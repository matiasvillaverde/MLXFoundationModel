#if FOUNDATION_MODELS_PROVIDER_API && canImport(FoundationModels)
import Foundation
import FoundationModels
@testable import MLXFoundationModel
import Testing

@Suite("Foundation Models provider reasoning contract")
struct MLXSessionProviderReasoningContractTests {
    @Test("request builder maps Gemma reasoning hints to thought channel markers")
    func requestBuilderMapsGemmaReasoningHintsToThoughtChannelMarkers() throws {
        guard #available(macOS 27.0, iOS 27.0, visionOS 27.0, *) else {
            return
        }

        let input = try FoundationModelsRequestBuilder.build(
            from: Self.reasoningRequest,
            model: Self.gemmaModel
        )

        #expect(input.context.hasSuffix("<|turn>model\n<|channel>thought\n"))
        #expect(input.sampling.advanced.reasoningBudget?.endMarker == "<channel|>")
    }

    @available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
    private static var reasoningRequest: LanguageModelExecutorGenerationRequest {
        LanguageModelExecutorGenerationRequest(
            id: UUID(),
            transcript: Transcript(entries: [
                .prompt(.init(segments: [.text(.init(content: "Answer after thinking."))]))
            ]),
            enabledTools: [],
            schema: nil,
            generationOptions: GenerationOptions(samplingMode: .greedy, maximumResponseTokens: 8),
            contextOptions: ContextOptions(reasoningLevel: .light),
            metadata: [:]
        )
    }

    @available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
    private static var gemmaModel: MLXLanguageModel {
        MLXLanguageModel(
            model: MLXModel(
                id: "gemma-reasoning-fixture",
                location: URL(fileURLWithPath: "/tmp/MLXFoundationModelGemmaReasoning"),
                promptStyle: .gemma,
                capabilities: .init(toolCalling: true, structuredOutput: true, reasoning: true)
            )
        )
    }
}
#endif

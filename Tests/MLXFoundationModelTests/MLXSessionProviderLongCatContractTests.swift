#if FOUNDATION_MODELS_PROVIDER_API && canImport(FoundationModels)
import Foundation
import FoundationModels
@testable import MLXFoundationModel
import Testing

@Suite("Foundation Models LongCat provider contract")
struct MLXSessionProviderLongCatContractTests {
    @Test("request builder maps LongCat reasoning hints to native think marker")
    func requestBuilderMapsLongCatReasoningHintsToNativeThinkMarker() throws {
        guard #available(macOS 27.0, iOS 27.0, visionOS 27.0, *) else {
            return
        }

        let request = LanguageModelExecutorGenerationRequest(
            id: UUID(),
            transcript: Self.transcript(prompt: "Answer after thinking."),
            enabledTools: [],
            generationOptions: GenerationOptions(samplingMode: .greedy, maximumResponseTokens: 16),
            contextOptions: ContextOptions(reasoningLevel: .light),
            metadata: [:]
        )
        let model = Self.longCatModel

        let input = try FoundationModelsRequestBuilder.build(from: request, model: model)

        #expect(input.sampling.advanced.reasoningBudget?.maximumTokens == 128)
        #expect(input.sampling.advanced.reasoningBudget?.endMarker == "</longcat_think>")
        #expect(input.context.hasSuffix("<longcat_think>\n"))
    }

    @available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
    private static var longCatModel: MLXLanguageModel {
        MLXLanguageModel(
            model: MLXModel(
                id: "longcat-contract-fixture",
                location: URL(fileURLWithPath: "/tmp/MLXFoundationModelLongCatContract"),
                promptStyle: .longCat,
                capabilities: .init(toolCalling: true, structuredOutput: true, reasoning: true)
            )
        )
    }

    @available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
    private static func transcript(prompt: String) -> Transcript {
        Transcript(entries: [
            .prompt(.init(segments: [.text(.init(content: prompt))]))
        ])
    }
}
#endif

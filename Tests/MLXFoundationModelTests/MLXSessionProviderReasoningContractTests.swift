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

    @Test("request builder replays Gemma reasoning transcript entries as native thought channel")
    func requestBuilderReplaysGemmaReasoningTranscriptEntriesAsNativeThoughtChannel() throws {
        guard #available(macOS 27.0, iOS 27.0, visionOS 27.0, *) else {
            return
        }

        let transcript = Transcript(entries: [
            .reasoning(.init(
                segments: [.text(.init(content: "The answer needs a city lookup."))]
            )),
            .prompt(.init(segments: [.text(.init(content: "Continue."))]))
        ])
        let request = Self.request(transcript: transcript)

        let input = try FoundationModelsRequestBuilder.build(from: request, model: Self.gemmaModel)
        let expected = """
        <|turn>model
        <|channel>thought
        The answer needs a city lookup.<channel|><turn|>
        """

        #expect(input.context.contains(expected))
        #expect(!input.context.contains("Reasoning:"))
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
    private static func request(transcript: Transcript) -> LanguageModelExecutorGenerationRequest {
        LanguageModelExecutorGenerationRequest(
            id: UUID(),
            transcript: transcript,
            enabledTools: [],
            schema: nil,
            generationOptions: GenerationOptions(samplingMode: .greedy, maximumResponseTokens: 8),
            contextOptions: ContextOptions(),
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

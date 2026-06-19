#if FOUNDATION_MODELS_PROVIDER_API && canImport(FoundationModels)
import CoreGraphics
import Foundation
import FoundationModels
@testable import MLXFoundationModel
import Testing

@Suite("Foundation Models provider contract")
struct MLXSessionProviderContractTests {
    @available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
    @Generable
    struct ContractAnswer {
        let summary: String
    }

    @Test("request builder replays custom transcript segments as text")
    func requestBuilderReplaysCustomTranscriptSegmentsAsText() throws {
        guard #available(macOS 27.0, iOS 27.0, visionOS 27.0, *) else {
            return
        }

        let transcript = Transcript(entries: [
            .prompt(.init(segments: [.custom(ContractCustomSegment(content: "custom context"))]))
        ])
        let request = Self.request(transcript: transcript)

        let input = try FoundationModelsRequestBuilder.build(from: request, model: Self.contractModel)

        #expect(input.context.contains("custom context"))
        #expect(input.images.isEmpty)
    }

    @Test("request builder replays reasoning transcript entries")
    func requestBuilderReplaysReasoningTranscriptEntries() throws {
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

        let input = try FoundationModelsRequestBuilder.build(from: request, model: Self.contractModel)

        #expect(input.context.contains("Reasoning:"))
        #expect(input.context.contains("The answer needs a city lookup."))
    }

    @Test("request builder rejects image attachments when vision is unavailable")
    func requestBuilderRejectsImageAttachmentsWhenVisionIsUnavailable() throws {
        guard #available(macOS 27.0, iOS 27.0, visionOS 27.0, *) else {
            return
        }

        let request = Self.request(transcript: try Self.imageTranscript())

        try Self.expectUnsupportedCapability(.vision) {
            _ = try FoundationModelsRequestBuilder.build(from: request, model: Self.contractModel)
        }
    }

    @Test("request builder carries image attachments for vision-capable models")
    func requestBuilderCarriesImageAttachmentsForVisionCapableModels() throws {
        guard #available(macOS 27.0, iOS 27.0, visionOS 27.0, *) else {
            return
        }

        let request = Self.request(transcript: try Self.imageTranscript())
        let model = Self.contractModel(capabilities: .init(
            toolCalling: true,
            structuredOutput: true,
            vision: true
        ))

        let input = try FoundationModelsRequestBuilder.build(from: request, model: model)

        #expect(input.context.contains("[Image: reference]"))
        #expect(input.images.count == 1)
    }

    @Test("request builder rejects schemas when guided generation is unavailable")
    func requestBuilderRejectsSchemasWhenGuidedGenerationIsUnavailable() throws {
        guard #available(macOS 27.0, iOS 27.0, visionOS 27.0, *) else {
            return
        }

        let request = Self.request(
            transcript: Self.transcript(prompt: "Generate a summary."),
            schema: ContractAnswer.generationSchema
        )
        let model = Self.contractModel(capabilities: .init(
            toolCalling: true,
            structuredOutput: false
        ))

        try Self.expectUnsupportedCapability(.guidedGeneration) {
            _ = try FoundationModelsRequestBuilder.build(from: request, model: model)
        }
    }

    @Test("request builder rejects tools when tool calling is unavailable")
    func requestBuilderRejectsToolsWhenToolCallingIsUnavailable() throws {
        guard #available(macOS 27.0, iOS 27.0, visionOS 27.0, *) else {
            return
        }

        let request = Self.request(
            transcript: Self.transcript(prompt: "Call a tool."),
            enabledTools: [Self.weatherTool]
        )
        let model = Self.contractModel(capabilities: .init(
            toolCalling: false,
            structuredOutput: true
        ))

        try Self.expectUnsupportedCapability(.toolCalling) {
            _ = try FoundationModelsRequestBuilder.build(from: request, model: model)
        }
    }

    @Test("request builder maps reasoning level hints for reasoning-capable models")
    func requestBuilderMapsReasoningLevelHintsForReasoningCapableModels() throws {
        guard #available(macOS 27.0, iOS 27.0, visionOS 27.0, *) else {
            return
        }

        let request = Self.request(
            transcript: Self.transcript(prompt: "Answer after thinking."),
            contextOptions: ContextOptions(reasoningLevel: .deep)
        )
        let model = Self.contractModel(capabilities: .init(
            toolCalling: true,
            structuredOutput: true,
            reasoning: true
        ))

        let input = try FoundationModelsRequestBuilder.build(from: request, model: model)

        #expect(input.context.contains("Use deep internal reasoning before answering."))
    }

    @Test("request builder rejects reasoning hints when reasoning is unavailable")
    func requestBuilderRejectsReasoningHintsWhenReasoningIsUnavailable() throws {
        guard #available(macOS 27.0, iOS 27.0, visionOS 27.0, *) else {
            return
        }

        let request = Self.request(
            transcript: Self.transcript(prompt: "Answer after thinking."),
            contextOptions: ContextOptions(reasoningLevel: .light)
        )

        try Self.expectUnsupportedCapability(.reasoning) {
            _ = try FoundationModelsRequestBuilder.build(from: request, model: Self.contractModel)
        }
    }

    @available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
    private static var contractModel: MLXLanguageModel {
        contractModel(capabilities: MLXModelCapabilities(toolCalling: true, structuredOutput: true))
    }

    @available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
    private static func contractModel(
        capabilities: MLXModelCapabilities,
        style: MLXPromptStyle = .chatML
    ) -> MLXLanguageModel {
        MLXLanguageModel(
            model: MLXModel(
                id: "contract-fixture",
                location: URL(fileURLWithPath: "/tmp/MLXFoundationModelContract"),
                promptStyle: style,
                capabilities: capabilities
            )
        )
    }

    @available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
    private static var weatherTool: Transcript.ToolDefinition {
        Transcript.ToolDefinition(
            name: "weather",
            description: "Read local weather for a city",
            parameters: ContractAnswer.generationSchema
        )
    }

    @available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
    private static func request(
        transcript: Transcript,
        enabledTools: [Transcript.ToolDefinition] = [],
        schema: GenerationSchema? = nil,
        contextOptions: ContextOptions = ContextOptions()
    ) -> LanguageModelExecutorGenerationRequest {
        LanguageModelExecutorGenerationRequest(
            id: UUID(),
            transcript: transcript,
            enabledTools: enabledTools,
            schema: schema,
            generationOptions: GenerationOptions(samplingMode: .greedy, maximumResponseTokens: 8),
            contextOptions: contextOptions,
            metadata: [:]
        )
    }

    @available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
    private static func transcript(prompt: String) -> Transcript {
        Transcript(entries: [
            .prompt(.init(segments: [.text(.init(content: prompt))]))
        ])
    }

    @available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
    private static func imageTranscript() throws -> Transcript {
        let attachment = Transcript.AttachmentSegment(
            content: .image(.init(try Self.makeTestImage())),
            label: "reference"
        )
        return Transcript(entries: [
            .prompt(.init(segments: [
                .text(.init(content: "Describe this image.")),
                .attachment(attachment)
            ]))
        ])
    }

    private static func makeTestImage() throws -> CGImage {
        guard
            let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
            let context = CGContext(
                data: nil,
                width: 1,
                height: 1,
                bitsPerComponent: 8,
                bytesPerRow: 4,
                space: colorSpace,
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
            ),
            let image = context.makeImage()
        else {
            throw CocoaError(.coderInvalidValue)
        }
        return image
    }

    @available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
    private static func expectUnsupportedCapability(
        _ capability: LanguageModelCapabilities.Capability,
        operation: () throws -> Void
    ) throws {
        do {
            try operation()
            Issue.record("Expected unsupported capability \(capability)")
        } catch let error as LanguageModelError {
            guard case .unsupportedCapability(let context) = error else {
                Issue.record("Expected unsupported capability, got \(error)")
                return
            }
            #expect(context.capability == capability)
        } catch {
            Issue.record("Expected LanguageModelError, got \(error)")
        }
    }

    @available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
    private struct ContractCustomSegment: Transcript.CustomSegment {
        let id = "contract-custom"
        let content: String
        var description: String {
            content
        }
    }
}
#endif

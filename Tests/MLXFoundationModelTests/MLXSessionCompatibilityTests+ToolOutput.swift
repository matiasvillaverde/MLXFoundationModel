#if FOUNDATION_MODELS_PROVIDER_API && canImport(FoundationModels)
import Foundation
import FoundationModels
@testable import MLXFoundationModel
import Testing

extension MLXSessionCompatibilityTests {
    @Test("request builder compacts oversized tool outputs")
    func requestBuilderCompactsOversizedToolOutputs() throws {
        guard #available(macOS 27.0, iOS 27.0, visionOS 27.0, *) else {
            return
        }

        let request = Self.largeToolOutputRequest()
        let input = try FoundationModelsRequestBuilder.build(
            from: request,
            model: Self.compactionModel
        )

        #expect(input.context.contains("Tool weather:"))
        #expect(input.context.contains("MLXFoundationModel truncated"))
        #expect(input.context.contains("prefix-"))
        #expect(input.context.contains("-suffix"))
        #expect(!input.context.contains(String(repeating: "x", count: 12_000)))
    }

    @available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
    private static func largeToolOutputRequest() -> LanguageModelExecutorGenerationRequest {
        LanguageModelExecutorGenerationRequest(
            id: UUID(),
            transcript: largeToolOutputTranscript(),
            enabledTools: [compactionWeatherTool],
            generationOptions: GenerationOptions(samplingMode: .greedy, maximumResponseTokens: 8),
            contextOptions: ContextOptions(),
            metadata: [:]
        )
    }

    @available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
    private static func largeToolOutputTranscript() -> Transcript {
        Transcript(entries: [
            .prompt(.init(segments: [.text(.init(content: "Read weather."))])),
            .toolOutput(.init(
                id: "tool-call-id",
                toolName: "weather",
                segments: [.text(.init(content: largeToolOutput()))]
            ))
        ])
    }

    private static func largeToolOutput() -> String {
        "prefix-"
            + String(repeating: "x", count: MLXToolOutputCompactor.defaultCharacterLimit)
            + "-middle-"
            + String(repeating: "y", count: MLXToolOutputCompactor.defaultCharacterLimit)
            + "-suffix"
    }

    @available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
    private static var compactionWeatherTool: Transcript.ToolDefinition {
        Transcript.ToolDefinition(
            name: "weather",
            description: "Read local weather for a city",
            parameters: WeatherArguments.generationSchema
        )
    }

    @available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
    private static var compactionModel: MLXLanguageModel {
        MLXLanguageModel(
            model: MLXModel(
                id: "tool-output-compaction-fixture",
                location: URL(fileURLWithPath: "/tmp/MLXFoundationModelToolOutput"),
                promptStyle: .plain,
                capabilities: MLXModelCapabilities(toolCalling: true, structuredOutput: true)
            )
        )
    }
}
#endif

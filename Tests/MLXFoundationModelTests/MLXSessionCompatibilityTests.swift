#if FOUNDATION_MODELS_PROVIDER_API && canImport(FoundationModels)
import Foundation
import FoundationModels
@testable import MLXFoundationModel
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

    @Test("request builder keeps schema grammar when schema is omitted from the prompt")
    func requestBuilderKeepsSchemaGrammarWhenSchemaIsOmittedFromPrompt() throws {
        guard #available(macOS 27.0, iOS 27.0, visionOS 27.0, *) else {
            return
        }

        let request = LanguageModelExecutorGenerationRequest(
            id: UUID(),
            transcript: Self.transcript(prompt: "Generate a compact weather summary."),
            enabledTools: [],
            schema: CompatibilityAnswer.generationSchema,
            generationOptions: GenerationOptions(samplingMode: .greedy, maximumResponseTokens: 12),
            contextOptions: ContextOptions(includeSchemaInPrompt: false),
            metadata: [:]
        )

        let input = try FoundationModelsRequestBuilder.build(from: request, model: Self.compatibilityModel)

        #expect(input.limits.maxTokens == 12)
        #expect(input.sampling.temperature == 0)
        #expect(input.sampling.topK == 1)
        #expect(input.sampling.advanced.grammar?.kind == .jsonSchema)
        #expect(!input.context.contains("Response constraints:"))
    }

    @Test("request builder constrains Foundation Models string choices")
    func requestBuilderConstrainsFoundationModelsStringChoices() throws {
        guard #available(macOS 27.0, iOS 27.0, visionOS 27.0, *) else {
            return
        }

        let schema = GenerationSchema(type: String.self, anyOf: ["apple", "pear", "banana"])
        let request = LanguageModelExecutorGenerationRequest(
            id: UUID(),
            transcript: Self.transcript(prompt: "Choose one fruit."),
            enabledTools: [],
            schema: schema,
            generationOptions: GenerationOptions(samplingMode: .greedy, maximumResponseTokens: 8),
            contextOptions: ContextOptions(includeSchemaInPrompt: false),
            metadata: [:]
        )

        let input = try FoundationModelsRequestBuilder.build(from: request, model: Self.compatibilityModel)
        let grammar = try #require(input.sampling.advanced.grammar)

        #expect(grammar.kind == .jsonSchema)
        #expect(grammar.grammar.contains("apple"))
        #expect(grammar.grammar.contains("pear"))
        #expect(grammar.grammar.contains("banana"))
        #expect(!input.context.contains("Response constraints:"))
    }

    @Test("request builder honors tool calling disallowed mode")
    func requestBuilderHonorsToolCallingDisallowedMode() throws {
        guard #available(macOS 27.0, iOS 27.0, visionOS 27.0, *) else {
            return
        }

        let request = LanguageModelExecutorGenerationRequest(
            id: UUID(),
            transcript: Self.transcript(prompt: "Answer without tools."),
            enabledTools: [Self.weatherTool],
            generationOptions: GenerationOptions(
                samplingMode: .greedy,
                maximumResponseTokens: 8,
                toolCallingMode: .disallowed
            ),
            contextOptions: ContextOptions(),
            metadata: [:]
        )

        let input = try FoundationModelsRequestBuilder.build(from: request, model: Self.compatibilityModel)

        #expect(!input.context.contains("Available tools:"))
        #expect(input.sampling.advanced.grammar == nil)
    }

    @Test("request builder constrains required tool calls")
    func requestBuilderConstrainsRequiredToolCalls() throws {
        guard #available(macOS 27.0, iOS 27.0, visionOS 27.0, *) else {
            return
        }

        let request = LanguageModelExecutorGenerationRequest(
            id: UUID(),
            transcript: Self.transcript(prompt: "Call the weather tool for Berlin."),
            enabledTools: [Self.weatherTool],
            generationOptions: GenerationOptions(
                samplingMode: .greedy,
                maximumResponseTokens: 24,
                toolCallingMode: .required
            ),
            contextOptions: ContextOptions(),
            metadata: [:]
        )

        let input = try FoundationModelsRequestBuilder.build(from: request, model: Self.compatibilityModel)
        let grammar = try #require(input.sampling.advanced.grammar)

        #expect(input.context.contains("Available tools:"))
        #expect(input.context.contains("Call one of the available tools"))
        #expect(grammar.kind == .jsonSchema)
        #expect(grammar.grammar.contains(#""tool_name""#))
        #expect(grammar.grammar.contains("weather"))
    }

    @Test("request builder carries sampling seeds")
    func requestBuilderCarriesSamplingSeeds() throws {
        guard #available(macOS 27.0, iOS 27.0, visionOS 27.0, *) else {
            return
        }

        let request = LanguageModelExecutorGenerationRequest(
            id: UUID(),
            transcript: Self.transcript(prompt: "Write a sentence."),
            enabledTools: [],
            generationOptions: GenerationOptions(
                samplingMode: .random(probabilityThreshold: 0.75, seed: 123),
                temperature: 0.4,
                maximumResponseTokens: 10
            ),
            contextOptions: ContextOptions(),
            metadata: [:]
        )

        let input = try FoundationModelsRequestBuilder.build(from: request, model: Self.compatibilityModel)

        #expect(input.sampling.topP == 0.75)
        #expect(input.sampling.temperature == 0.4)
        #expect(input.sampling.seed == 123)
    }

    @Test("request builder maps tool output names")
    func requestBuilderMapsToolOutputNames() throws {
        guard #available(macOS 27.0, iOS 27.0, visionOS 27.0, *) else {
            return
        }

        let transcript = Transcript(entries: [
            .prompt(.init(segments: [.text(.init(content: "Read weather."))])),
            .toolOutput(.init(
                id: "tool-call-id",
                toolName: "weather",
                segments: [.text(.init(content: "21 C"))]
            ))
        ])
        let request = LanguageModelExecutorGenerationRequest(
            id: UUID(),
            transcript: transcript,
            enabledTools: [Self.weatherTool],
            generationOptions: GenerationOptions(samplingMode: .greedy, maximumResponseTokens: 8),
            contextOptions: ContextOptions(),
            metadata: [:]
        )
        let model = Self.compatibilityModel(style: .plain)

        let input = try FoundationModelsRequestBuilder.build(from: request, model: model)

        #expect(input.context.contains("Tool weather:"))
        #expect(!input.context.contains("Tool tool-call-id:"))
    }

    @Test("request builder rejects unsupported transcript segments")
    func requestBuilderRejectsUnsupportedTranscriptSegments() throws {
        guard #available(macOS 27.0, iOS 27.0, visionOS 27.0, *) else {
            return
        }

        let transcript = Transcript(entries: [
            .prompt(.init(segments: [.custom(CompatibilityCustomSegment(content: "custom"))]))
        ])
        let request = LanguageModelExecutorGenerationRequest(
            id: UUID(),
            transcript: transcript,
            enabledTools: [],
            generationOptions: GenerationOptions(samplingMode: .greedy, maximumResponseTokens: 8),
            contextOptions: ContextOptions(),
            metadata: [:]
        )

        do {
            _ = try FoundationModelsRequestBuilder.build(from: request, model: Self.compatibilityModel)
            Issue.record("Expected unsupported transcript content")
        } catch let error as LanguageModelError {
            guard case .unsupportedTranscriptContent(let context) = error else {
                Issue.record("Expected unsupported transcript content, got \(error)")
                return
            }
            #expect(context.unsupportedContent.count == 1)
        } catch {
            Issue.record("Expected LanguageModelError, got \(error)")
        }
    }

    @available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
    private static var compatibilityModel: MLXLanguageModel {
        compatibilityModel(style: .chatML)
    }

    @available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
    private static func compatibilityModel(style: MLXPromptStyle) -> MLXLanguageModel {
        MLXLanguageModel(
            model: MLXModel(
                id: "compatibility-fixture",
                location: URL(fileURLWithPath: "/tmp/MLXFoundationModelCompatibility"),
                promptStyle: style,
                capabilities: MLXModelCapabilities(toolCalling: true, structuredOutput: true)
            )
        )
    }

    @available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
    private static var weatherTool: Transcript.ToolDefinition {
        Transcript.ToolDefinition(
            name: "weather",
            description: "Read local weather for a city",
            parameters: CompatibilityAnswer.generationSchema
        )
    }

    @available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
    private static func transcript(prompt: String) -> Transcript {
        Transcript(entries: [
            .instructions(.init(
                segments: [.text(.init(content: "Answer briefly."))],
                toolDefinitions: []
            )),
            .prompt(.init(segments: [.text(.init(content: prompt))]))
        ])
    }

    @available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
    private struct CompatibilityCustomSegment: Transcript.CustomSegment {
        let id = "compatibility-custom"
        let content: String
        var description: String {
            content
        }
    }
}
#endif

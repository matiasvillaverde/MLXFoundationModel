import Foundation
import MLXFoundationModel
import MLXFoundationModelExamples
import MLXLocalModels

#if FOUNDATION_MODELS_PROVIDER_API && canImport(FoundationModels)
import FoundationModels
#endif

@main
enum FoundationModelsPlayground {
    static func main() async throws {
        let configuration = try PlaygroundConfiguration.parse()

        #if FOUNDATION_MODELS_PROVIDER_API && canImport(FoundationModels)
        if #available(macOS 27.0, iOS 27.0, visionOS 27.0, *) {
            try await FoundationModelsSessionPlayground(configuration: configuration).run()
            return
        }
        #endif

        try await DirectMLXPlayground(configuration: configuration).run()
    }
}

struct PlaygroundConfiguration: Sendable {
    let modelURL: URL
    let modelID: String
    let exampleID: String?

    static func parse() throws -> Self {
        let arguments = Array(CommandLine.arguments.dropFirst())
        let environment = ProcessInfo.processInfo.environment
        guard
            let modelPath = value(for: "--model-path", in: arguments)
                ?? environment["MLX_FOUNDATION_MODEL_PATH"],
            !modelPath.isEmpty
        else {
            throw PlaygroundConfigurationError.missingModelPath
        }

        return Self(
            modelURL: URL(fileURLWithPath: NSString(string: modelPath).expandingTildeInPath),
            modelID: value(for: "--model-id", in: arguments)
                ?? environment["MLX_FOUNDATION_MODEL_ID"]
                ?? "local-mlx-model",
            exampleID: value(for: "--example", in: arguments)
        )
    }

    var selectedExamples: [FoundationModelPlaygroundExample] {
        guard let exampleID else {
            return FoundationModelPlaygroundExamples.all
        }
        return FoundationModelPlaygroundExamples.all.filter { $0.id == exampleID }
    }

    private static func value(
        for flag: String,
        in arguments: [String]
    ) -> String? {
        guard let index = arguments.firstIndex(of: flag) else {
            return nil
        }
        let valueIndex = arguments.index(after: index)
        guard arguments.indices.contains(valueIndex) else {
            return nil
        }
        return arguments[valueIndex]
    }
}

enum PlaygroundConfigurationError: Error, LocalizedError {
    case missingModelPath

    var errorDescription: String? {
        switch self {
        case .missingModelPath:
            """
            Missing model path. Pass --model-path /path/to/model or set MLX_FOUNDATION_MODEL_PATH.
            Download test models with: MLX_ASSUME_YES=1 MLX_MODEL_FILTER=smoke make download-test-models
            """
        }
    }
}

struct DirectMLXPlayground {
    let configuration: PlaygroundConfiguration

    func run() async throws {
        let session = MLXSessionFactory.create()
        do {
            try await preload(session)
            for example in configuration.selectedExamples {
                try await run(example, session: session)
            }
            await session.unload()
        } catch {
            await session.unload()
            throw error
        }
    }

    private func preload(_ session: any MLXGeneratingSession) async throws {
        let progress = await session.preload(configuration: ProviderConfiguration(
            location: configuration.modelURL,
            authentication: .noAuth,
            modelName: configuration.modelID,
            compute: .small,
            runtime: ModelRuntimePreferences(promptCachePolicy: .memory)
        ))
        for try await _ in progress {
            // Drain preload progress before generation.
        }
    }

    private func run(
        _ example: FoundationModelPlaygroundExample,
        session: any MLXGeneratingSession
    ) async throws {
        print("\n=== \(example.title) ===")
        let rendered = MLXPromptRenderer.render(example.request, style: example.style)
        let input = LLMInput(
            context: rendered.prompt,
            promptMetadata: PromptRenderMetadata(rendererID: rendered.rendererID),
            promptCacheIdentity: PromptCacheIdentity(stableFingerprint: rendered.cacheFingerprint),
            sampling: example.sampling,
            limits: example.limits
        )

        var metrics: ChunkMetrics?
        for try await chunk in await session.stream(input) {
            if case .text = chunk.event, !chunk.text.isEmpty {
                print(chunk.text, terminator: "")
                fflush(stdout)
            }
            metrics = chunk.metrics ?? metrics
        }
        print()
        print(Self.usageLine(metrics))
    }

    private static func usageLine(_ metrics: ChunkMetrics?) -> String {
        guard let usage = metrics?.usage else {
            return "usage unavailable"
        }
        let promptTokens = usage.promptTokens.map(String.init) ?? "unknown"
        return "tokens prompt=\(promptTokens) generated=\(usage.generatedTokens) total=\(usage.totalTokens)"
    }
}

#if FOUNDATION_MODELS_PROVIDER_API && canImport(FoundationModels)
@available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
struct FoundationModelsSessionPlayground {
    let configuration: PlaygroundConfiguration

    func run() async throws {
        try await runStreamingChat()
        try await runTripPlannerGuidedGeneration()
        try await runPointsOfInterestToolCalling()
        try await runFiniteChoiceGuidedGeneration()
    }

    private var model: MLXLanguageModel {
        MLXLanguageModel(
            model: MLXModel(
                id: configuration.modelID,
                location: configuration.modelURL,
                promptStyle: .chatML,
                capabilities: MLXModelCapabilities(toolCalling: true, structuredOutput: true)
            ),
            compute: .small,
            runtime: ModelRuntimePreferences(promptCachePolicy: .memory),
            maximumResponseTokens: 160
        )
    }

    private func runStreamingChat() async throws {
        let session = LanguageModelSession(
            model: model,
            instructions: "You are a concise assistant. Do not think aloud."
        )
        session.prewarm()
        print("\n=== Streaming chat ===")
        var printed = ""
        for try await snapshot in session.streamResponse(
            to: "/no_think\nWrite one short sentence about local MLX inference.",
            options: GenerationOptions(samplingMode: .greedy, maximumResponseTokens: 32)
        ) {
            print(snapshot.content.dropFirst(printed.count), terminator: "")
            fflush(stdout)
            printed = snapshot.content
        }
        print()
        print(Self.usageLine(session.usage))
    }

    private func runTripPlannerGuidedGeneration() async throws {
        let session = LanguageModelSession(
            model: model,
            tools: [FindPointsOfInterestTool()],
            instructions: """
            Your job is to create an itinerary for the person.
            Each day needs an activity, hotel, or restaurant.
            """
        )
        session.prewarm()
        print("\n=== Apple Trip Planner guided generation ===")
        for try await snapshot in session.streamResponse(
            to: """
            Generate a one-day itinerary to Yosemite. Give it a fun title, \
            choose the Yosemite destination, and include one activity.
            """,
            generating: PlaygroundTrip.self,
            includeSchemaInPrompt: false,
            options: GenerationOptions(samplingMode: .greedy, maximumResponseTokens: 128)
        ) {
            print(snapshot.content)
        }
        print(Self.usageLine(session.usage))
    }

    private func runPointsOfInterestToolCalling() async throws {
        let session = LanguageModelSession(
            model: model,
            tools: [FindPointsOfInterestTool()],
            instructions: "Use the findPointsOfInterest tool when a location lookup helps."
        )
        session.prewarm()
        print("\n=== Apple points-of-interest tool calling ===")
        let response = try await session.respond(
            to: "Find one hotel near Yosemite, then summarize it in one sentence.",
            options: GenerationOptions(
                samplingMode: .greedy,
                maximumResponseTokens: 96,
                toolCallingMode: .allowed
            )
        )
        print(response.content)
        print(Self.usageLine(response.usage))
    }

    private func runFiniteChoiceGuidedGeneration() async throws {
        let session = LanguageModelSession(model: model)
        session.prewarm()
        print("\n=== Apple finite-choice guided generation ===")
        let response = try await session.respond(
            to: "Choose one fruit.",
            generating: FruitChoice.self,
            options: GenerationOptions(samplingMode: .greedy, maximumResponseTokens: 16)
        )
        print(response.content.fruit)
        print(Self.usageLine(response.usage))
    }

    private static func usageLine(_ usage: LanguageModelSession.Usage) -> String {
        """
        tokens input=\(usage.input.totalTokenCount) cached=\(usage.input.cachedTokenCount) \
        output=\(usage.output.totalTokenCount)
        """
    }
}

@available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
@Generable
struct PlaygroundTrip {
    @Guide(description: "An exciting name for the trip.")
    let title: String

    @Guide(.anyOf(["Yosemite"]))
    let destinationName: String

    let day: PlaygroundDayPlan
}

@available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
@Generable
struct PlaygroundDayPlan {
    let title: String
    let activityKind: PlaygroundActivityKind
    let activity: String
}

@available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
@Generable
enum PlaygroundActivityKind {
    case sightseeing
    case foodAndDining
    case hotelAndLodging
}

@available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
@Generable
struct FruitChoice {
    @Guide(.anyOf(["apple", "pear", "banana"]))
    let fruit: String
}

@available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
struct FindPointsOfInterestTool: Tool {
    let name = "findPointsOfInterest"
    let description = "Finds points of interest for a landmark."

    @Generable
    enum Category {
        case hotel
    }

    @Generable
    struct Arguments {
        @Guide(description: "The type of destination to look up.")
        let pointOfInterest: Category

        @Guide(description: "The natural language query of what to search for.")
        let naturalLanguageQuery: String
    }

    func call(arguments: Arguments) async throws -> String {
        "There are these \(arguments.pointOfInterest) in Yosemite: Yosemite Valley Lodge."
    }
}
#endif

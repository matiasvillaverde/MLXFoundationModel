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
        let arguments = Array(CommandLine.arguments.dropFirst())
        if arguments.contains("--help") || arguments.contains("-h") {
            printHelp()
            return
        }
        if arguments.contains("--list-examples") {
            printExamples()
            return
        }

        let configuration = try PlaygroundConfiguration.parse()
        guard !configuration.selectedExamples.isEmpty else {
            throw PlaygroundConfigurationError.unknownExample(configuration.exampleID ?? "")
        }

        #if FOUNDATION_MODELS_PROVIDER_API && canImport(FoundationModels)
        if #available(macOS 27.0, iOS 27.0, visionOS 27.0, *) {
            try await FoundationModelsSessionPlayground(configuration: configuration).run()
            return
        }
        #endif

        try await DirectMLXPlayground(configuration: configuration).run()
    }

    private static func printHelp() {
        print([
            "FoundationModelsPlayground",
            "",
            "Run Apple-style requests against a local MLX model.",
            "",
            "Usage:",
            "  swift run FoundationModelsPlayground \\",
            "    --model-path .models/Qwen3-0.6B-4bit \\",
            "    --model-id qwen3-0.6b-4bit",
            "  swift run FoundationModelsPlayground --list-examples",
            "",
            "Options:",
            "  --model-path PATH      Downloaded MLX model directory.",
            "  --model-id ID          Catalog or display identifier.",
            "  --example ID           Run one example.",
            "  --prompt TEXT          Run a benchmark prompt.",
            "  --max-tokens N         Benchmark token cap."
        ].joined(separator: "\n"))
    }

    private static func printExamples() {
        for example in FoundationModelPlaygroundExamples.all {
            print("\(example.id)\t\(example.title)")
        }
    }
}

struct PlaygroundConfiguration: Sendable {
    let modelURL: URL
    let modelID: String
    let exampleID: String?
    let benchmarkPrompt: String?
    let benchmarkInstructions: String?
    let benchmarkMaxTokens: Int?

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

        let expandedModelPath = expandingTilde(in: modelPath)
        return Self(
            modelURL: URL(fileURLWithPath: expandedModelPath),
            modelID: value(for: "--model-id", in: arguments)
                ?? environment["MLX_FOUNDATION_MODEL_ID"]
                ?? "local-mlx-model",
            exampleID: value(for: "--example", in: arguments),
            benchmarkPrompt: value(for: "--prompt", in: arguments)
                ?? environment["MLX_PROFILE_PROMPT"],
            benchmarkInstructions: value(for: "--instructions", in: arguments)
                ?? environment["MLX_PROFILE_INSTRUCTIONS"],
            benchmarkMaxTokens: intValue(for: "--max-tokens", in: arguments)
                ?? intValue(environment["MLX_PROFILE_MAX_TOKENS"])
        )
    }

    var selectedExamples: [FoundationModelPlaygroundExample] {
        if let benchmarkPrompt, !benchmarkPrompt.isEmpty {
            return [
                FoundationModelPlaygroundExample(
                    id: "benchmark",
                    title: "Benchmark",
                    request: MLXBridgeRequest(
                        messages: [
                            MLXBridgeMessage(role: .user, content: benchmarkPrompt)
                        ],
                        instructions: benchmarkInstructions
                            ?? "You are a concise assistant. Do not think aloud."
                    ),
                    sampling: .deterministic,
                    limits: ResourceLimits(
                        maxTokens: benchmarkMaxTokens ?? 128,
                        maxTime: .seconds(300),
                        reusePromptCache: false
                    )
                )
            ]
        }

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

    private static func intValue(
        for flag: String,
        in arguments: [String]
    ) -> Int? {
        value(for: flag, in: arguments).flatMap(intValue)
    }

    private static func intValue(_ value: String?) -> Int? {
        guard let value, let parsed = Int(value), parsed > 0 else {
            return nil
        }
        return parsed
    }

    private static func expandingTilde(in path: String) -> String {
        if path == "~" {
            return NSHomeDirectory()
        }
        guard path.hasPrefix("~/") else {
            return path
        }
        return NSHomeDirectory() + String(path.dropFirst())
    }
}

enum PlaygroundConfigurationError: Error, LocalizedError {
    case missingModelPath

    case unknownExample(String)

    var errorDescription: String? {
        switch self {
        case .missingModelPath:
            "Missing model path. Pass --model-path or set MLX_FOUNDATION_MODEL_PATH. "
                + "Try the default model with: make demo"

        case .unknownExample(let exampleID):
            "Unknown playground example: \(exampleID). List examples with --list-examples."
        }
    }
}

struct DirectMLXPlayground {
    let configuration: PlaygroundConfiguration

    func run() async throws {
        let session = MLXSessionFactory.create()
        do {
            let modelPromptStyle = Self.modelPromptStyle(for: configuration)
            try await preload(session)
            for example in configuration.selectedExamples {
                try await run(example, session: session, modelPromptStyle: modelPromptStyle)
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
        session: any MLXGeneratingSession,
        modelPromptStyle: MLXPromptStyle
    ) async throws {
        print("\n=== \(example.title) ===")
        let rendered = MLXPromptRenderer.render(
            example.request,
            style: example.resolvedStyle(modelDefault: modelPromptStyle)
        )
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
        let tokenSummary = "tokens prompt=\(promptTokens) "
            + "generated=\(usage.generatedTokens) "
            + "total=\(usage.totalTokens)"
        var fields = [
            tokenSummary
        ]

        if let timing = metrics?.timing {
            let totalSeconds = seconds(timing.totalTime)
            let promptSeconds = timing.promptProcessingTime.map(seconds)
            let generationSeconds = max(totalSeconds - (promptSeconds ?? 0), 0)
            if generationSeconds > 0 {
                fields.append(String(
                    format: "generation=%.2f tok/s",
                    Double(usage.generatedTokens) / generationSeconds
                ))
            }
            if let promptSeconds {
                fields.append(String(format: "prompt=%.3fs", promptSeconds))
            }
            fields.append(String(format: "total=%.3fs", totalSeconds))
        }

        return fields.joined(separator: " ")
    }

    private static func seconds(_ duration: Duration) -> Double {
        Double(duration.components.seconds) + Double(duration.components.attoseconds) / 1e18
    }

    private static func modelPromptStyle(
        for configuration: PlaygroundConfiguration
    ) -> MLXPromptStyle {
        (try? MLXModelProfile.load(
            from: configuration.modelURL,
            id: configuration.modelID
        ).promptStyle) ?? .chatML
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
                promptStyle: Self.modelPromptStyle(for: configuration),
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

    private static func modelPromptStyle(
        for configuration: PlaygroundConfiguration
    ) -> MLXPromptStyle {
        (try? MLXModelProfile.load(
            from: configuration.modelURL,
            id: configuration.modelID
        ).promptStyle) ?? .chatML
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

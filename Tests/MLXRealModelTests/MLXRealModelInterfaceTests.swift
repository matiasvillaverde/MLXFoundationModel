import MLXFoundationModel
@testable import MLXLocalModels
import Testing

@Suite(
    "MLX real-model Foundation Models-style interface",
    .serialized,
    .disabled(
        if: !MLXRealModelEnvironment.isEnabled,
        "Set MLX_RUN_REAL_MODEL_TESTS=1 and download models with scripts/download-test-models.sh"
    )
)
struct MLXRealModelInterfaceTests {
    private struct LifecycleEventIndexes {
        let requestStart: Int
        let promptStart: Int
        let promptProgress: Int
        let promptEnd: Int
        let decodeStart: Int
    }

    @Test("selected models run rendered session-style requests")
    func selectedModelsRunRenderedSessionStyleRequests() async throws {
        let models = try MLXRealModelCatalog.load()
        let selected = MLXRealModelEnvironment
            .selectedModels(from: models)
            .filter { !$0.tags.contains("native-template-only") }
        let missing = selected.filter { !MLXRealModelEnvironment.hasModelFiles(for: $0) }

        #expect(!selected.isEmpty)
        #expect(
            missing.isEmpty,
            Comment(rawValue: MLXRealModelEnvironment.missingModelsMessage(missing))
        )
        guard missing.isEmpty else {
            return
        }

        var failures: [String] = []
        for model in selected {
            do {
                try await Self.verifySessionStyleRequest(on: model)
            } catch {
                failures.append("\(model.id): \(error)")
            }
        }
        #expect(failures.isEmpty, Comment(rawValue: failures.joined(separator: "\n")))
    }

    @Test("Qwen3 streams multiple chunks from a rendered text request")
    func qwen3StreamsMultipleChunksFromRenderedTextRequest() async throws {
        let models = try MLXRealModelCatalog.load()
        guard let model = try MLXRealModelHarness.selectedModel("qwen3-0.6b-4bit", in: models) else {
            return
        }
        let result = try await MLXRealModelHarness.runRenderedRequest(
            model: model,
            request: Self.textStreamingRequest,
            limits: ResourceLimits(maxTokens: 24, maxTime: .seconds(120), reusePromptCache: false),
            style: .chatML
        )

        MLXRealModelHarness.verifyGenerated(result)
        #expect(result.textChunkCount > 1)
    }

    @Test("Qwen3 raw stream reports lifecycle phase boundaries")
    func qwen3RawStreamReportsLifecyclePhaseBoundaries() async throws {
        let models = try MLXRealModelCatalog.load()
        guard let model = try MLXRealModelHarness.selectedModel("qwen3-0.6b-4bit", in: models) else {
            return
        }

        let result = try await MLXRealModelHarness.run(
            model: model,
            prompt: "/no_think\nReply with one short word.",
            limits: ResourceLimits(maxTokens: 4, maxTime: .seconds(120), reusePromptCache: false)
        )

        MLXRealModelHarness.verifyGenerated(result)
        try Self.verifyLifecycleEvents(result.lifecycleEvents)
    }

    @Test("Qwen3 on-demand stream reports model-load progress")
    func qwen3OnDemandStreamReportsModelLoadProgress() async throws {
        let models = try MLXRealModelCatalog.load()
        guard let model = try MLXRealModelHarness.selectedModel("qwen3-0.6b-4bit", in: models) else {
            return
        }

        let input = LLMInput(
            context: "/no_think\nReply with one short word.",
            sampling: .deterministic,
            limits: ResourceLimits(maxTokens: 4, maxTime: .seconds(120), reusePromptCache: false)
        )
        let result = try await Self.runOnDemandStream(model: model, input: input)

        MLXRealModelHarness.verifyGenerated(result)
        try Self.verifyModelLoadProgressEvents(result.lifecycleEvents)
    }

    private static var sessionStyleRequest: MLXBridgeRequest {
        MLXBridgeRequest(
            messages: [
                MLXBridgeMessage(
                    role: .user,
                    content: "/no_think\nProject: MLXFoundationModel. Reply briefly."
                )
            ],
            instructions: "Answer in short plain text."
        )
    }

    private static func verifySessionStyleRequest(
        on model: MLXRealModelCatalog.Model
    ) async throws {
        let result = try await MLXRealModelHarness.runRenderedRequest(
            model: model,
            request: Self.sessionStyleRequest,
            limits: ResourceLimits(
                maxTokens: min(
                    8,
                    max(4, MLXRealModelEnvironment.architectureGenerationTokenLimit)
                ),
                maxTime: .seconds(MLXRealModelEnvironment.architectureGenerationTimeoutSeconds),
                reusePromptCache: false
            )
        )
        MLXRealModelHarness.verifyGenerated(result)
    }

    private static var textStreamingRequest: MLXBridgeRequest {
        MLXBridgeRequest(
            messages: [
                MLXBridgeMessage(
                    role: .user,
                    content: "/no_think\nWrite two short clauses about local model adapters."
                )
            ],
            instructions: "Answer in plain text. Do not include Markdown."
        )
    }

    private static func providerConfiguration(
        for model: MLXRealModelCatalog.Model
    ) -> ProviderConfiguration {
        ProviderConfiguration(
            location: MLXRealModelEnvironment.modelURL(for: model),
            authentication: .noAuth,
            modelName: model.displayName,
            compute: .small,
            runtime: MLXRealModelEnvironment.runtimePreferences(for: model)
        )
    }

    private static func runOnDemandStream(
        model: MLXRealModelCatalog.Model,
        input: LLMInput
    ) async throws -> MLXRealModelHarness.GenerationResult {
        let session = MLXSession(configuration: Self.providerConfiguration(for: model))
        do {
            let result = try await MLXRealModelHarness.collectGeneration(
                from: await session.stream(input)
            )
            await session.unload()
            return result
        } catch {
            await session.unload()
            throw error
        }
    }

    private static func verifyLifecycleEvents(
        _ events: [StreamLifecycleEvent]
    ) throws {
        let indexes = try Self.lifecycleEventIndexes(in: events)
        Self.verifyLifecycleOrdering(indexes)
        Self.verifyPromptProgressEvent(events[indexes.promptProgress])
        Self.verifyPromptEndEvent(events[indexes.promptEnd])
    }

    private static func lifecycleEventIndexes(
        in events: [StreamLifecycleEvent]
    ) throws -> LifecycleEventIndexes {
        LifecycleEventIndexes(
            requestStart: try Self.requiredIndex(of: .request, state: .started, in: events),
            promptStart: try Self.requiredIndex(
                of: .promptProcessing,
                state: .started,
                in: events
            ),
            promptProgress: try Self.requiredIndex(
                of: .promptProcessing,
                state: .progress,
                in: events
            ),
            promptEnd: try Self.requiredIndex(of: .promptProcessing, state: .ended, in: events),
            decodeStart: try Self.requiredIndex(of: .decode, state: .started, in: events)
        )
    }

    private static func verifyLifecycleOrdering(_ indexes: LifecycleEventIndexes) {
        #expect(indexes.requestStart < indexes.promptStart)
        #expect(indexes.promptStart < indexes.promptProgress)
        #expect(indexes.promptProgress < indexes.promptEnd)
        #expect(indexes.promptEnd < indexes.decodeStart)
    }

    private static func verifyPromptProgressEvent(_ event: StreamLifecycleEvent) {
        #expect((event.totalUnitCount ?? 0) > 0)
        #expect(event.completedUnitCount == 0)
        #expect(event.cachedUnitCount == 0)
    }

    private static func verifyPromptEndEvent(_ event: StreamLifecycleEvent) {
        #expect((event.totalUnitCount ?? 0) > 0)
        #expect(event.completedUnitCount == event.totalUnitCount)
        #expect(event.cachedUnitCount == 0)
    }

    private static func verifyModelLoadProgressEvents(
        _ events: [StreamLifecycleEvent]
    ) throws {
        let loadStart = try #require(Self.index(of: .modelLoad, state: .started, in: events))
        let loadEnd = try #require(Self.index(of: .modelLoad, state: .ended, in: events))
        let promptStart = try #require(Self.index(
            of: .promptProcessing,
            state: .started,
            in: events
        ))
        let progressEvents = events.enumerated().filter { _, event in
            event.phase == .modelLoad && event.state == .progress
        }

        #expect(loadStart < loadEnd)
        #expect(loadEnd < promptStart)
        #expect(!progressEvents.isEmpty)
        #expect(progressEvents.allSatisfy { index, _ in loadStart < index && index < loadEnd })
        #expect(progressEvents.last?.element.totalUnitCount == 100)
        #expect(progressEvents.last?.element.completedUnitCount == 100)
    }

    private static func index(
        of phase: StreamLifecyclePhase,
        state: StreamLifecycleState,
        in events: [StreamLifecycleEvent]
    ) -> Int? {
        events.firstIndex { event in
            event.phase == phase && event.state == state
        }
    }

    private static func requiredIndex(
        of phase: StreamLifecyclePhase,
        state: StreamLifecycleState,
        in events: [StreamLifecycleEvent]
    ) throws -> Int {
        try #require(Self.index(of: phase, state: state, in: events))
    }
}

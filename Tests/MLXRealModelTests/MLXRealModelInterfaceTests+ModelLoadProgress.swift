import MLXFoundationModel
@testable import MLXLocalModels
import Testing

extension MLXRealModelInterfaceTests {
    @Test("selected models report on-demand stream model-load progress")
    func selectedModelsReportOnDemandStreamModelLoadProgress() async throws {
        let models = try MLXRealModelCatalog.load()
        let selected = MLXRealModelEnvironment.selectedModels(from: models)
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
                try await Self.verifyOnDemandModelLoadProgress(model: model)
            } catch {
                failures.append("\(model.id): \(error)")
            }
        }
        #expect(failures.isEmpty, Comment(rawValue: failures.joined(separator: "\n")))
    }

    private static func verifyOnDemandModelLoadProgress(
        model: MLXRealModelCatalog.Model
    ) async throws {
        let input = LLMInput(
            context: "/no_think\nReply with one short word.",
            sampling: .deterministic,
            limits: ResourceLimits(
                maxTokens: min(
                    4,
                    max(1, MLXRealModelEnvironment.architectureGenerationTokenLimit)
                ),
                maxTime: .seconds(MLXRealModelEnvironment.architectureGenerationTimeoutSeconds),
                reusePromptCache: false
            )
        )
        let result = try await Self.runOnDemandProgressStream(model: model, input: input)

        MLXRealModelHarness.verifyGenerated(result)
        try Self.verifyModelLoadProgress(result.lifecycleEvents)
    }

    private static func runOnDemandProgressStream(
        model: MLXRealModelCatalog.Model,
        input: LLMInput
    ) async throws -> MLXRealModelHarness.GenerationResult {
        let session = MLXSession(configuration: Self.progressProviderConfiguration(for: model))
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

    private static func progressProviderConfiguration(
        for model: MLXRealModelCatalog.Model
    ) -> ProviderConfiguration {
        ProviderConfiguration(
            location: MLXRealModelEnvironment.modelURL(for: model),
            authentication: .noAuth,
            modelName: "\(model.displayName) progress-e2e",
            compute: .small,
            runtime: MLXRealModelEnvironment.runtimePreferences(for: model)
        )
    }

    private static func verifyModelLoadProgress(
        _ events: [StreamLifecycleEvent]
    ) throws {
        let loadStart = try #require(Self.lifecycleIndex(
            of: .modelLoad,
            state: .started,
            in: events
        ))
        let loadEnd = try #require(Self.lifecycleIndex(of: .modelLoad, state: .ended, in: events))
        let promptStart = try #require(Self.lifecycleIndex(
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

    private static func lifecycleIndex(
        of phase: StreamLifecyclePhase,
        state: StreamLifecycleState,
        in events: [StreamLifecycleEvent]
    ) -> Int? {
        events.firstIndex { event in
            event.phase == phase && event.state == state
        }
    }
}

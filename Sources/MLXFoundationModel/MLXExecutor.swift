#if FOUNDATION_MODELS_PROVIDER_API && canImport(FoundationModels)
import FoundationModels
import MLXLocalModels

/// Executes Foundation Models requests against a local MLX session.
@available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
public struct MLXExecutor: LanguageModelExecutor {
    public typealias Model = MLXLanguageModel

    public struct Configuration: Equatable, Hashable, Sendable {
        public let model: MLXModel
        public let compute: ComputeConfiguration
        public let runtime: ModelRuntimePreferences
        public let sampling: SamplingParameters
        public let maximumResponseTokens: Int

        public init(
            model: MLXModel,
            compute: ComputeConfiguration,
            runtime: ModelRuntimePreferences,
            sampling: SamplingParameters,
            maximumResponseTokens: Int
        ) {
            self.model = model
            self.compute = compute
            self.runtime = runtime
            self.sampling = sampling
            self.maximumResponseTokens = maximumResponseTokens
        }
    }

    private let configuration: Configuration
    private let session: any MLXGeneratingSession

    public init(configuration: Configuration) throws {
        self.configuration = configuration
        session = MLXSessionFactory.create()
    }

    public func prewarm(model: MLXLanguageModel, transcript: Transcript) {
        // MLX model loading is asynchronous; respond(to:) drains preload progress before generation.
        _ = model
        _ = transcript
    }

    public func respond(
        to request: LanguageModelExecutorGenerationRequest,
        model: MLXLanguageModel,
        streamingInto channel: LanguageModelExecutorGenerationChannel
    ) async throws {
        do {
            let input = try FoundationModelsRequestBuilder.build(from: request, model: model)
            try await preloadIfNeeded(model.providerConfiguration)
            try await MLXEventTranslator().translate(
                await session.stream(input),
                into: channel,
                toolDefinitionsEnabled: !request.enabledToolDefinitions.isEmpty
            )
        } catch {
            throw MLXErrorMapper.map(error)
        }
    }

    private func preloadIfNeeded(_ providerConfiguration: ProviderConfiguration) async throws {
        let progress = await session.preload(configuration: providerConfiguration)
        for try await _ in progress {
            // Drain progress before the first generation request.
        }
    }
}
#endif

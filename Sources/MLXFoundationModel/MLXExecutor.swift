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
    private let prewarmCoordinator = MLXPrewarmCoordinator()

    public init(configuration: Configuration) throws {
        try self.init(configuration: configuration, session: MLXSessionFactory.create())
    }

    /// Creates an executor that leases sessions from a shared model pool.
    public init(
        configuration: Configuration,
        pool: MLXModelPool
    ) throws {
        try self.init(
            configuration: configuration,
            session: MLXPooledSession(
                model: Self.languageModel(from: configuration),
                pool: pool
            )
        )
    }

    internal init(
        configuration: Configuration,
        session: any MLXGeneratingSession
    ) throws {
        self.configuration = configuration
        self.session = session
    }

    public func prewarm(model: MLXLanguageModel, transcript: Transcript) {
        prewarmCoordinator.prewarm(
            configuration: model.providerConfiguration,
            session: session
        )
        _ = transcript
    }

    public func respond(
        to request: LanguageModelExecutorGenerationRequest,
        model: MLXLanguageModel,
        streamingInto channel: LanguageModelExecutorGenerationChannel
    ) async throws {
        do {
            let input = try FoundationModelsRequestBuilder.build(from: request, model: model)
            let tools = FoundationModelsRequestBuilder.bridgeToolDefinitions(from: request)
            try await preloadIfNeeded(model.providerConfiguration)
            try await MLXEventTranslator().translate(
                await session.stream(input),
                into: channel,
                tools: tools,
                promptStyle: model.model.promptStyle,
                reasoningStartsOpen: MLXPromptTemplateRenderer.generationStartsInReasoning(
                    reasoningOptions: FoundationModelsRequestBuilder.reasoningOptions(
                        for: request.contextOptions.reasoningLevel,
                        model: model
                    ) ?? .disabled,
                    style: model.model.promptStyle
                )
            )
        } catch {
            throw MLXErrorMapper.map(error)
        }
    }

    private func preloadIfNeeded(_ providerConfiguration: ProviderConfiguration) async throws {
        try await prewarmCoordinator.ensureLoaded(
            configuration: providerConfiguration,
            session: session
        )
    }

    private static func languageModel(
        from configuration: Configuration
    ) -> MLXLanguageModel {
        MLXLanguageModel(
            model: configuration.model,
            compute: configuration.compute,
            runtime: configuration.runtime,
            sampling: configuration.sampling,
            maximumResponseTokens: configuration.maximumResponseTokens
        )
    }
}
#endif

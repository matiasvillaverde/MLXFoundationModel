import Foundation
import MLX
import OSLog

// swiftlint:disable type_body_length
/// MLX implementation of the LLMSession protocol
internal actor MLXSession: LLMSession {
    // MARK: - Properties

    let logger = MLXObservability.logger(for: .generation)

    #if DEBUG
    private let debugLogger = MLXObservability.logger(for: .generation)
    #endif

    var configuration: ProviderConfiguration?
    var modelContainer: ModelContainer?
    var speculativeDecoding: MLXSpeculativeDecodingConfiguration?
    var runtimeCapabilities: MLXGenerationRuntimeCapabilities
    var runtimePreferences: ModelRuntimePreferences = .default
    var generationExecutionPlan: MLXGenerationExecutionPlan?
    var continuousBatchEngine: MLXContinuousBatchSessionEngine?
    var persistentPromptCacheURL: URL?
    var memoryProfile: MLXModelMemoryProfile?
    var activeGenerationCount = 0
    var isGenerating: Bool { activeGenerationCount > 0 }
    var pendingUnloadAfterGeneration = false
    let stopFlag = StopFlag()
    let generationAdmission = MLXGenerationAdmissionController()
    let clock = ContinuousClock() // Metrics tracking

    internal init(
        configuration: ProviderConfiguration? = nil,
        modelContainer: ModelContainer? = nil,
        speculativeDecoding: MLXSpeculativeDecodingConfiguration? = nil,
        runtimeCapabilities: MLXGenerationRuntimeCapabilities = .scalar
    ) {
        self.configuration = configuration
        self.modelContainer = modelContainer
        self.speculativeDecoding = speculativeDecoding
        self.runtimeCapabilities = runtimeCapabilities
    }

    /// Stream text generation based on the provided configuration
    internal func stream(_ input: LLMInput) -> AsyncThrowingStream<LLMStreamChunk, Error> {
        AsyncThrowingStream { continuation in
            let stopFlag = self.stopFlag
            let cancellationState = MLXGenerationCancellationState()
            let generationTask = Task {
                await self.runStreamTask(
                    input: input,
                    cancellationState: cancellationState,
                    continuation: continuation
                )
            }
            continuation.onTermination = { @Sendable _ in
                if cancellationState.shouldSignalStop() {
                    stopFlag.set(true)
                }
                generationTask.cancel()
            }
        }
    }

    /// Stop the current generation
    nonisolated internal func stop() {
        logger.info("Stop requested - setting stop flag")
        stopFlag.set(true)
    }

    func streamModelLoad(
        continuation: AsyncThrowingStream<Progress, Error>.Continuation
    ) async throws {
        guard let configuration else {
            logger.error("Cannot preload model: Configuration not set")
            throw LLMError.invalidConfiguration("Configuration not set. Call preload first.")
        }

        let span = MLXObservability.startSpan(
            .modelLoad,
            attributes: ["model": configuration.modelName]
        )
        defer { span.end() }

        let loadStart = clock.now
        logger.info("Preloading model from \(configuration.location.path)")

        let progress = Progress(totalUnitCount: 100)
        progress.localizedDescription = "Loading MLX model"
        continuation.yield(progress)

        try await preflightRuntimeForModelLoad()

        let modelConfig = ModelConfiguration(directory: configuration.location)
        modelContainer = try await LLMModelFactory.shared.loadContainer(
            configuration: modelConfig
        ) { continuation.yield($0) }
        try await applyRuntimeConfiguration()

        let duration = loadStart.duration(to: clock.now)
        logger.info("Model preloaded in \(duration)")
        if duration > .seconds(30) {
            logger.warning("Slow preload: \(duration)")
        }
    }

    func loadModel() async throws {
        guard let configuration else {
            logger.error("Model not loaded: Configuration must be set via preload() before generation")
            throw LLMError.invalidConfiguration("Configuration not set. Call preload first.")
        }

        let span = MLXObservability.startSpan(
            .modelLoad,
            attributes: ["model": configuration.modelName]
        )
        defer { span.end() }

        let loadStart = clock.now
        logger.info("Loading model from \(configuration.location.path)")

        try await preflightRuntimeForModelLoad()

        let modelConfig = ModelConfiguration(directory: configuration.location)
        modelContainer = try await LLMModelFactory.shared.loadContainer(
            configuration: modelConfig
        ) { _ in /* Progress callback not used */ }
        try await applyRuntimeConfiguration()

        let duration = loadStart.duration(to: clock.now)
        logger.info("Model loaded in \(duration)")
        if duration > .seconds(10) {
            logger.warning("Slow load: \(duration)")
        }
    }

    func generateStream(
        input: LLMInput,
        continuation: AsyncThrowingStream<LLMStreamChunk, Error>.Continuation
    ) async throws {
        if !input.images.isEmpty {
            logger.error(
                "Invalid input: MLX models don't support image inputs. Received \(input.images.count)"
            )
            throw LLMError.invalidConfiguration("MLX models don't support image inputs")
        }
        if !input.videoURLs.isEmpty {
            logger.error(
                "Invalid input: MLX models don't support video inputs. Received \(input.videoURLs.count)"
            )
            throw LLMError.invalidConfiguration("MLX models don't support video inputs")
        }
        guard let container = modelContainer else {
            logger.error("Model not loaded: Configuration must be set via preload() before generation")
            throw LLMError.modelNotFound("Model not loaded")
        }
        try await performGeneration(container: container, input: input, continuation: continuation)
    }

    private func performGeneration(
        container: ModelContainer,
        input: LLMInput,
        continuation: AsyncThrowingStream<LLMStreamChunk, Error>.Continuation
    ) async throws {
        let span = MLXObservability.startSpan(
            .generation,
            attributes: [
                "model": configuration?.modelName ?? "unknown",
                "strategy": String(describing: generationExecutionPlan?.selectedStrategy ?? .scalar)
            ]
        )
        defer { span.end() }

        let request = await makeScalarGenerationRunRequest(
            input: input,
            container: container,
            continuation: continuation
        )
        let metricsData = try await generateTokens(container: container, request: request)

        let totalDuration = metricsData.generationStartTime.duration(to: clock.now)
        let metrics = metricsData.chunkMetrics(
            totalDuration: totalDuration,
            contextWindowSize: configuration?.compute.contextSize
        )
        recordGenerationSummary(metricsData: metricsData, totalDuration: totalDuration)
        continuation.yield(LLMStreamChunk(text: "", event: .finished, metrics: metrics))

        logGenerationMetrics(metricsData: metricsData, startTime: request.generationStartTime)
    }

    private func makeScalarGenerationRunRequest(
        input: LLMInput,
        container: ModelContainer,
        continuation: AsyncThrowingStream<LLMStreamChunk, Error>.Continuation
    ) async -> MLXGenerationRunRequest {
        let currentRuntimePreferences = runtimePreferences
        let generateParams = await createGenerateParameters(
            from: input,
            container: container,
            runtimePreferences: currentRuntimePreferences,
        )
        MLXGenerationDiagnostics.recordParameters(generateParams)
        #if DEBUG
        debugLogger.info(
            "Starting generation: \(input.limits.maxTokens) tokens, temp \(generateParams.temperature)"
        )
        #endif
        return MLXGenerationRunRequest(
            input: input,
            parameters: generateParams,
            runtimePreferences: currentRuntimePreferences,
            generationStartTime: clock.now,
            continuation: continuation
        )
    }

    private func logGenerationMetrics(metricsData: MetricsData, startTime: ContinuousClock.Instant) {
        let totalDuration = startTime.duration(to: clock.now)
        let durationSeconds = Double(totalDuration.components.seconds) +
            Double(totalDuration.components.attoseconds) / 1e18
        let tokensPerSecond = Double(metricsData.generatedTokenCount) / durationSeconds
        let tokPerSec = String(format: "%.2f", tokensPerSecond)

        #if DEBUG
        let totalTokens = metricsData.promptTokenCount + metricsData.generatedTokenCount
        debugLogger.info("""
            Generation complete: \(metricsData.generatedTokenCount)/\(totalTokens) tokens, \
            \(totalDuration), \(tokPerSec) tok/s
            """)
        #endif

        logger.info(
            "Generation: \(metricsData.generatedTokenCount) tokens, \(totalDuration), \(tokPerSec) tok/s"
        )

        // Performance warnings
        if tokensPerSecond < 1.0 {
            logger.warning("Very slow generation: \(tokPerSec) tok/s")
        } else if tokensPerSecond < 5.0 {
            logger.warning("Slow generation: \(tokPerSec) tok/s")
        }
    }

    private func generateTokens(
        container: ModelContainer,
        request: MLXGenerationRunRequest
    ) async throws -> MetricsData {
        let speculativeDecoding = self.speculativeDecoding
        let promptCacheVariant = Self.promptCacheVariant(for: speculativeDecoding)
        let profile = self.memoryProfile
        return try await container.performWithPromptCache { context, promptCacheEntries in
            let genContext = GenerationContext(
                modelContext: context,
                input: request.input,
                parameters: request.parameters,
                generationStartTime: request.generationStartTime,
                continuation: request.continuation,
                clock: clock,
                runtimePreferences: request.runtimePreferences,
                memoryProfile: profile
            )
            return try self.executeGeneration(
                genContext,
                promptCacheEntries: &promptCacheEntries,
                speculativeDecoding: speculativeDecoding,
                promptCacheVariant: promptCacheVariant
            )
        }
    }

    // swiftlint:disable:next function_body_length
    nonisolated private func executeGeneration(
        _ genContext: GenerationContext,
        promptCacheEntries: inout [PromptCacheEntry],
        speculativeDecoding: MLXSpeculativeDecodingConfiguration?,
        promptCacheVariant: String?
    ) throws -> MetricsData {
        let prepared = try prepareGeneration(
            genContext: genContext,
            promptCacheEntries: &promptCacheEntries,
            speculativeDecoding: speculativeDecoding,
            promptCacheVariant: promptCacheVariant
        )
        let promptCacheLease = prepared.cachePlan.lease
        defer {
            if let lease = promptCacheLease {
                PromptCachePlanner.release(lease, in: &promptCacheEntries)
            }
        }

        let iterator = try MLXGenerationDiagnostics.withAdaptivePrefillController(
            prepared.adaptivePrefillController
        ) {
            try makeIterator(
                genContext: prepared.genContext,
                cachePlan: prepared.cachePlan,
                fullInput: prepared.fullInput,
                speculativeDecoding: speculativeDecoding
            )
        }

        let promptEndTime = genContext.clock.now

        let reusableState = iterator.cacheForPromptReuse
        updatePromptCacheEntries(
            &promptCacheEntries,
            tokenIds: prepared.promptTokenIDs,
            reusableState: reusableState,
            genContext: prepared.genContext,
            promptCacheVariant: promptCacheVariant
        )

        #if DEBUG
        if prepared.cachePlan.reusedTokenCount > 0 {
            let reusedTokenCount = prepared.cachePlan.reusedTokenCount
            let promptTokenCount = prepared.promptTokenIDs.count
            debugLogger.info("""
                Prompt cache reused \(reusedTokenCount)/\(promptTokenCount) tokens
                """)
        }
        #endif

        for token in iterator {
            if Task.isCancelled || stopFlag.get() {
                prepared.state.stopReason = .userRequested
                break
            }
            if isTimedOut(genContext) {
                prepared.state.stopReason = .timeout
                break
            }
            if isStopToken(token, tokenContext: prepared.tokenContext) {
                prepared.state.stopReason = .endOfSequence
                break
            }
            if processToken(token: token, tokenContext: prepared.tokenContext) == .stop {
                break
            }
        }

        flushPendingText(tokenContext: prepared.tokenContext)
        Stream().synchronize()

        let kvCacheBytes = Int64(PromptCachePlanner.cacheByteCount(reusableState.cache))
        let kvCacheEntries = reusableState.cache.map(\.offset).max()

        return MetricsData(
            generationStartTime: genContext.generationStartTime,
            promptStartTime: prepared.promptStartTime,
            promptEndTime: promptEndTime,
            firstTokenTime: prepared.state.firstTokenTime,
            promptTokenCount: prepared.promptTokenIDs.count,
            generatedTokenCount: prepared.state.generatedTokenCount,
            kvCacheBytes: kvCacheBytes,
            kvCacheEntries: kvCacheEntries,
            promptCacheReusedTokenCount: prepared.cachePlan.reusedTokenCount,
            stopReason: prepared.state.stopReason,
            parameters: prepared.genContext.parameters
        )
    }
}
// swiftlint:enable type_body_length

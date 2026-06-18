import Foundation
import MLX
import OSLog

// swiftlint:disable type_body_length
/// MLX implementation of the LLMSession protocol
internal actor MLXSession: LLMSession {
    // MARK: - Properties

    let logger = Logger(subsystem: "MLXSession", category: "MLXSession")

    #if DEBUG
    private let debugLogger = Logger(subsystem: "MLXSession", category: "MLXSession.Debug")
    #endif

    var configuration: ProviderConfiguration?
    var modelContainer: ModelContainer?
    var speculativeDecoding: MLXSpeculativeDecodingConfiguration?
    var runtimePreferences: ModelRuntimePreferences = .default
    var persistentPromptCacheURL: URL?
    private var isGenerating = false
    let stopFlag = StopFlag()
    private let clock = ContinuousClock() // Metrics tracking

    internal init(
        configuration: ProviderConfiguration? = nil,
        modelContainer: ModelContainer? = nil,
        speculativeDecoding: MLXSpeculativeDecodingConfiguration? = nil
    ) {
        self.configuration = configuration
        self.modelContainer = modelContainer
        self.speculativeDecoding = speculativeDecoding
    }

    /// Stream text generation based on the provided configuration
    internal func stream(_ input: LLMInput) -> AsyncThrowingStream<LLMStreamChunk, Error> {
        AsyncThrowingStream { continuation in
            let stopFlag = self.stopFlag
            let generationTask = Task {
                do {
                    stopFlag.reset()
                    try Task.checkCancellation()
                    isGenerating = true
                    defer { isGenerating = false }

                    logger.debug("Starting stream generation")

                    if modelContainer == nil {
                        logger.info("Model not preloaded - loading on demand")
                        try await loadModel()
                    }

                    try Task.checkCancellation()
                    try await generateStream(input: input, continuation: continuation)
                    continuation.finish()
                } catch is CancellationError {
                    continuation.finish()
                } catch {
                    logger.error("Stream generation failed: \(error.localizedDescription)")
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { @Sendable _ in
                stopFlag.set(true)
                generationTask.cancel()
            }
        }
    }

    /// Stop the current generation
    nonisolated internal func stop() {
        logger.info("Stop requested - setting stop flag")
        stopFlag.set(true)
    }

    /// Unload a model from memory
    internal func unload() async {
        if modelContainer != nil {
            logger.info("Unloading model from memory")
            try? await persistPromptCacheIfNeeded()
            modelContainer = nil
        } else {
            logger.debug("Unload called but no model was loaded")
        }
    }
    // MARK: - Private Methods

    func streamModelLoad(
        continuation: AsyncThrowingStream<Progress, Error>.Continuation
    ) async throws {
        guard let configuration else {
            logger.error("Cannot preload model: Configuration not set")
            throw LLMError.invalidConfiguration("Configuration not set. Call preload first.")
        }

        let loadStart = clock.now
        logger.info("Preloading model from \(configuration.location.path)")

        let progress = Progress(totalUnitCount: 100)
        progress.localizedDescription = "Loading MLX model"
        continuation.yield(progress)

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

    private func loadModel() async throws {
        guard let configuration else {
            logger.error("Model not loaded: Configuration must be set via preload() before generation")
            throw LLMError.invalidConfiguration("Configuration not set. Call preload first.")
        }

        let loadStart = clock.now
        logger.info("Loading model from \(configuration.location.path)")

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

    private func generateStream(
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
        let generateParams = createGenerateParameters(
            from: input.sampling,
            limits: input.limits
        )
        MLXGenerationDiagnostics.recordParameters(generateParams)
        let generationStartTime = clock.now

        #if DEBUG
        debugLogger.info(
            "Starting generation: \(input.limits.maxTokens) tokens, temp \(generateParams.temperature)"
        )
        #endif

        let metricsData = try await generateTokens(
            container: container,
            input: input,
            parameters: generateParams,
            generationStartTime: generationStartTime,
            continuation: continuation
        )

        let metrics = metricsData.chunkMetrics(
            totalDuration: metricsData.generationStartTime.duration(to: clock.now),
            contextWindowSize: configuration?.compute.contextSize
        )
        continuation.yield(LLMStreamChunk(text: "", event: .finished, metrics: metrics))

        logGenerationMetrics(metricsData: metricsData, startTime: generationStartTime)
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
        input: LLMInput,
        parameters: GenerateParameters,
        generationStartTime: ContinuousClock.Instant,
        continuation: AsyncThrowingStream<LLMStreamChunk, Error>.Continuation
    ) async throws -> MetricsData {
        let speculativeDecoding = self.speculativeDecoding
        let promptCacheVariant = Self.promptCacheVariant(for: speculativeDecoding)
        return try await container.performWithPromptCache { [clock] context, promptCacheEntries in
            let genContext = GenerationContext(
                modelContext: context,
                input: input,
                parameters: parameters,
                generationStartTime: generationStartTime,
                continuation: continuation,
                clock: clock
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
        let promptStartTime = genContext.clock.now

        let fullInput = try genContext.modelContext.tokenize(input: genContext.input)
        eval(fullInput.text.tokens)

        let tokenizationEndTime = genContext.clock.now
        let tokenizeDuration = promptStartTime.duration(to: tokenizationEndTime)
        let promptTokenIds = fullInput.text.tokens.asArray(Int.self)

        #if DEBUG
        debugLogger.info("Tokenization complete: \(promptTokenIds.count) tokens, \(tokenizeDuration)")
        #endif

        if tokenizeDuration > .seconds(10) {
            logger.warning("Slow tokenization: \(tokenizeDuration) for \(promptTokenIds.count) tokens")
        }

        let cachePlan = PromptCachePlanner.plan(
            fullInput: fullInput,
            tokenIds: promptTokenIds,
            parameters: genContext.parameters,
            cacheVariant: promptCacheVariant,
            promptCacheIdentity: genContext.input.promptCacheIdentity,
            existingEntries: promptCacheEntries,
            reuseEnabled: genContext.input.limits.reusePromptCache,
            requiresDraftCache: speculativeDecoding != nil
        )
        MLXGenerationDiagnostics.recordPromptCachePlan(
            promptTokenCount: promptTokenIds.count,
            reusedTokenCount: cachePlan.reusedTokenCount
        )

        let state = GenerationState()
        let tokenContext = TokenContext(
            state: state,
            context: genContext.modelContext,
            input: genContext.input,
            continuation: genContext.continuation,
            clock: genContext.clock
        )
        state.stopDetector = StopSequenceDetector(
            sequences: genContext.input.sampling.stopSequences
        )
        state.detokenizer = NaiveStreamingDetokenizer(tokenizer: genContext.modelContext.tokenizer)

        let iterator = try makeIterator(
            genContext: genContext,
            cachePlan: cachePlan,
            fullInput: fullInput,
            speculativeDecoding: speculativeDecoding
        )

        let promptEndTime = genContext.clock.now

        let reusableState = iterator.cacheForPromptReuse
        updatePromptCacheEntries(
            &promptCacheEntries,
            tokenIds: promptTokenIds,
            reusableState: reusableState,
            genContext: genContext,
            promptCacheVariant: promptCacheVariant
        )

        #if DEBUG
        if cachePlan.reusedTokenCount > 0 {
            debugLogger.info("""
                Prompt cache reused \(cachePlan.reusedTokenCount)/\(promptTokenIds.count) tokens
                """)
        }
        #endif

        for token in iterator {
            if Task.isCancelled || stopFlag.get() {
                state.stopReason = .userRequested
                break
            }
            if isTimedOut(genContext) {
                state.stopReason = .timeout
                break
            }
            if isStopToken(token, context: genContext.modelContext) {
                state.stopReason = .endOfSequence
                break
            }
            if processToken(token: token, tokenContext: tokenContext) == .stop {
                break
            }
        }

        flushPendingText(tokenContext: tokenContext)
        Stream().synchronize()

        let kvCacheBytes = Int64(PromptCachePlanner.cacheByteCount(reusableState.cache))
        let kvCacheEntries = reusableState.cache.map(\.offset).max()

        return MetricsData(
            generationStartTime: genContext.generationStartTime,
            promptStartTime: promptStartTime,
            promptEndTime: promptEndTime,
            firstTokenTime: state.firstTokenTime,
            promptTokenCount: promptTokenIds.count,
            generatedTokenCount: state.generatedTokenCount,
            kvCacheBytes: kvCacheBytes,
            kvCacheEntries: kvCacheEntries,
            promptCacheReusedTokenCount: cachePlan.reusedTokenCount,
            stopReason: state.stopReason,
            parameters: genContext.parameters
        )
    }

    nonisolated private static func promptCacheVariant(
        for speculativeDecoding: MLXSpeculativeDecodingConfiguration?
    ) -> String? {
        speculativeDecoding.map { "speculative:\($0.draftContext.configuration.name)" }
    }

    nonisolated private func isTimedOut(_ genContext: GenerationContext) -> Bool {
        guard let maxTime: Duration = genContext.input.limits.maxTime else {
            return false
        }
        return genContext.generationStartTime.duration(to: genContext.clock.now) >= maxTime
    }

    // MARK: - Metrics Helpers
}
// swiftlint:enable type_body_length

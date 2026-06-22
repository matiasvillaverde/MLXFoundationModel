import Foundation

extension MLXSession {
    func generateContinuousBatchStream(
        input: LLMInput,
        continuation: AsyncThrowingStream<LLMStreamChunk, any Error>.Continuation
    ) async throws {
        let runtime = try continuousBatchRuntime()
        try validateContinuousBatchInput(input)
        let generationStartTime = clock.now
        let request = await makeGenerationRunRequest(
            input: input,
            container: runtime.container,
            generationStartTime: generationStartTime,
            continuation: continuation
        )
        let prepared = try await makeContinuousBatchPreparedRequest(
            container: runtime.container,
            request: request
        )
        try await runContinuousBatchPreparedRequest(
            prepared,
            engine: runtime.engine,
            generationStartTime: generationStartTime,
            continuation: continuation
        )
    }

    func configureContinuousBatchEngine(
        for container: ModelContainer
    ) async {
        if generationExecutionPlan?.selectedStrategy == .continuousBatching {
            continuousBatchEngine = MLXContinuousBatchSessionEngine(
                container: container,
                scheduling: runtimePreferences.scheduling
            )
        } else {
            await closeContinuousBatchEngine()
        }
    }

    func closeContinuousBatchEngine() async {
        guard let engine = continuousBatchEngine else {
            return
        }
        await engine.close()
        continuousBatchEngine = nil
    }

    private func continuousBatchRuntime() throws -> (
        container: ModelContainer,
        engine: MLXContinuousBatchSessionEngine
    ) {
        guard let container = modelContainer else {
            logger.error("Model not loaded: Configuration must be set via preload() before generation")
            throw LLMError.modelNotFound("Model not loaded")
        }
        guard let engine = continuousBatchEngine else {
            throw LLMError.invalidConfiguration(
                "continuousBatching selected but the MLXSession batched stream executor is unavailable."
            )
        }
        return (container, engine)
    }

    private func makeGenerationRunRequest(
        input: LLMInput,
        container: ModelContainer,
        generationStartTime: ContinuousClock.Instant,
        continuation: AsyncThrowingStream<LLMStreamChunk, any Error>.Continuation
    ) async -> MLXGenerationRunRequest {
        let runtimePreferences = self.runtimePreferences
        let parameters = await createGenerateParameters(
            from: input,
            container: container,
            runtimePreferences: runtimePreferences
        )
        MLXGenerationDiagnostics.recordParameters(parameters)
        return MLXGenerationRunRequest(
            input: input,
            parameters: parameters,
            runtimePreferences: runtimePreferences,
            generationStartTime: generationStartTime,
            continuation: continuation
        )
    }

    private func makeContinuousBatchPreparedRequest(
        container: ModelContainer,
        request: MLXGenerationRunRequest
    ) async throws -> MLXContinuousBatchPreparedRequest {
        let speculativeDecoding = self.speculativeDecoding
        let promptCacheVariant = Self.promptCacheVariant(for: speculativeDecoding)
        let clock = self.clock
        let memoryProfile = self.memoryProfile
        return try await container.performWithPromptCache { context, promptCacheEntries in
            let genContext = GenerationContext(
                modelContext: context,
                input: request.input,
                parameters: request.parameters,
                generationStartTime: request.generationStartTime,
                continuation: request.continuation,
                clock: clock,
                runtimePreferences: request.runtimePreferences,
                memoryProfile: memoryProfile
            )
            return try makeContinuousBatchPreparedRequest(
                genContext: genContext,
                promptCacheEntries: &promptCacheEntries,
                speculativeDecoding: speculativeDecoding,
                promptCacheVariant: promptCacheVariant
            )
        }
    }

    nonisolated private func makeContinuousBatchPreparedRequest(
        genContext: GenerationContext,
        promptCacheEntries: inout [PromptCacheEntry],
        speculativeDecoding: MLXSpeculativeDecodingConfiguration?,
        promptCacheVariant: String?
    ) throws -> MLXContinuousBatchPreparedRequest {
        let prepared = try prepareGeneration(
            genContext: genContext,
            promptCacheEntries: &promptCacheEntries,
            speculativeDecoding: speculativeDecoding,
            promptCacheVariant: promptCacheVariant
        )
        defer {
            if let lease = prepared.cachePlan.lease {
                PromptCachePlanner.release(lease, in: &promptCacheEntries)
            }
        }
        let completion = MLXContinuousBatchStreamCompletion(state: prepared.state)
        let prefillRequest = try prepared.makeContinuousBatchPrefillRequest(
            sink: completion.streamSink(continuation: genContext.continuation),
            promptCacheVariant: promptCacheVariant
        )
        return MLXContinuousBatchPreparedRequest(
            completion: completion,
            parameters: genContext.parameters,
            prefillRequest: prefillRequest,
            promptCacheReusedTokenCount: prepared.cachePlan.reusedTokenCount,
            promptStartTime: prepared.promptStartTime,
            promptTokenCount: prepared.promptTokenIDs.count,
            state: prepared.state
        )
    }

    private func waitForContinuousBatchCompletion(
        _ prepared: MLXContinuousBatchPreparedRequest,
        queueID: MLXContinuousBatchRequestID,
        engine: MLXContinuousBatchSessionEngine
    ) async throws {
        try await withTaskCancellationHandler {
            try await prepared.completion.wait()
        } onCancel: {
            prepared.completion.cancel()
            Task {
                await engine.cancel(id: queueID)
            }
        }
    }

    private func runContinuousBatchPreparedRequest(
        _ prepared: MLXContinuousBatchPreparedRequest,
        engine: MLXContinuousBatchSessionEngine,
        generationStartTime: ContinuousClock.Instant,
        continuation: AsyncThrowingStream<LLMStreamChunk, any Error>.Continuation
    ) async throws {
        let span = MLXObservability.startSpan(
            .continuousBatch,
            attributes: ["model": configuration?.modelName ?? "unknown"]
        )
        defer { span.end() }

        let queueID = try await engine.enqueue(prepared.prefillRequest)
        try await waitForContinuousBatchCompletion(
            prepared,
            queueID: queueID,
            engine: engine
        )
        yieldContinuousBatchFinishedChunk(
            prepared,
            generationStartTime: generationStartTime,
            continuation: continuation
        )
    }

    private func yieldContinuousBatchFinishedChunk(
        _ prepared: MLXContinuousBatchPreparedRequest,
        generationStartTime: ContinuousClock.Instant,
        continuation: AsyncThrowingStream<LLMStreamChunk, any Error>.Continuation
    ) {
        let now = clock.now
        let metricsData = MetricsData(
            generationStartTime: generationStartTime,
            promptStartTime: prepared.promptStartTime,
            promptEndTime: prepared.state.firstTokenTime ?? now,
            firstTokenTime: prepared.state.firstTokenTime,
            promptTokenCount: prepared.promptTokenCount,
            generatedTokenCount: prepared.state.generatedTokenCount,
            kvCacheBytes: nil,
            kvCacheEntries: nil,
            promptCacheReusedTokenCount: prepared.promptCacheReusedTokenCount,
            stopReason: prepared.state.stopReason,
            parameters: prepared.parameters
        )
        let totalDuration = generationStartTime.duration(to: now)
        let metrics = metricsData.chunkMetrics(
            totalDuration: totalDuration,
            contextWindowSize: configuration?.compute.contextSize
        )
        recordGenerationSummary(metricsData: metricsData, totalDuration: totalDuration)
        continuation.yield(LLMStreamChunk(text: "", event: .finished, metrics: metrics))
    }

    private func validateContinuousBatchInput(_ input: LLMInput) throws {
        if !input.images.isEmpty {
            throw LLMError.invalidConfiguration("MLX models don't support image inputs")
        }
        if !input.videoURLs.isEmpty {
            throw LLMError.invalidConfiguration("MLX models don't support video inputs")
        }
    }
}

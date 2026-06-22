import Foundation

extension MLXSession {
    func runStreamTask(
        input: LLMInput,
        cancellationState: MLXGenerationCancellationState,
        continuation: AsyncThrowingStream<LLMStreamChunk, Error>.Continuation
    ) async {
        do {
            let span = MLXObservability.startSpan(.admissionWait)
            let lease: MLXGenerationAdmissionController.Lease
            do {
                lease = try await generationAdmission.acquire()
                span.end()
            } catch {
                span.end()
                throw error
            }
            cancellationState.activate()
            defer { finishGenerationAdmission(lease, cancellationState: cancellationState) }
            try await runAdmittedStream(input: input, continuation: continuation)
            continuation.finish()
        } catch is CancellationError {
            continuation.finish()
        } catch {
            logger.error("Stream generation failed: \(error.localizedDescription)")
            continuation.finish(throwing: error)
        }
    }

    private func runAdmittedStream(
        input: LLMInput,
        continuation: AsyncThrowingStream<LLMStreamChunk, Error>.Continuation
    ) async throws {
        stopFlag.reset()
        try Task.checkCancellation()
        beginGeneration()
        defer { finishGeneration() }

        if modelContainer == nil {
            logger.info("Model not preloaded - loading on demand")
            try await loadModel()
        }

        try Task.checkCancellation()
        logger.debug("Starting stream generation")
        if generationExecutionPlan?.selectedStrategy == .continuousBatching {
            try await generateContinuousBatchStream(input: input, continuation: continuation)
        } else {
            try await generateStream(input: input, continuation: continuation)
        }
    }

    private func finishGenerationAdmission(
        _ lease: MLXGenerationAdmissionController.Lease,
        cancellationState: MLXGenerationCancellationState
    ) {
        cancellationState.deactivate()
        Task {
            await self.generationAdmission.release(lease)
        }
    }
}

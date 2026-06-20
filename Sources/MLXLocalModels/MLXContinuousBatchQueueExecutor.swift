internal struct MLXContinuousBatchQueueExecutor<Stepper: MLXContinuousBatchStreamStepping> {
    internal let configuration: MLXContinuousBatchExecutorConfig
    internal let makeStepper: (MLXContinuousBatchGenerationBatch) throws -> Stepper
    internal let queue: MLXContinuousBatchRequestQueue

    internal init(
        queue: MLXContinuousBatchRequestQueue,
        configuration: MLXContinuousBatchExecutorConfig = .default,
        makeStepper: @escaping (MLXContinuousBatchGenerationBatch) throws -> Stepper
    ) {
        self.configuration = configuration
        self.makeStepper = makeStepper
        self.queue = queue
    }

    internal func runNextBatch() async throws -> MLXContinuousBatchExecutorResult? {
        let queuedRequests = try await queue.nextBatch(maxCount: configuration.maxBatchSize)
        guard !queuedRequests.isEmpty else {
            return nil
        }

        let batch: MLXContinuousBatchGenerationBatch
        let stepper: Stepper
        do {
            batch = try MLXContinuousBatchAssembler.assemble(
                requests: queuedRequests.map(\.request),
                pagedKVBlockCapacity: configuration.pagedKVBlockCapacity
            )
            stepper = try makeStepper(batch)
        } catch {
            fail(queuedRequests, error: error)
            throw error
        }

        let driver = try MLXContinuousBatchStreamDriver(
            stepper: stepper,
            rows: batch.streamRows
        )
        var runLoop = MLXContinuousBatchRunLoop(
            driver: driver,
            configuration: configuration.runLoopConfiguration
        )
        let result = try runLoop.run()

        return MLXContinuousBatchExecutorResult(
            requestIDs: queuedRequests.map(\.id),
            rowIDs: batch.orderedRowIDs,
            runLoopResult: result
        )
    }

    private func fail(
        _ queuedRequests: [MLXContinuousBatchQueuedRequest],
        error: any Error
    ) {
        for queuedRequest in queuedRequests {
            queuedRequest.request.sink.fail(error)
        }
    }
}

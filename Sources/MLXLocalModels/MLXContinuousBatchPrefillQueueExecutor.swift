internal struct MLXContinuousBatchPrefillQueueExecutor<
    PrefillRunner: MLXContinuousBatchPrefillRunning,
    Stepper: MLXContinuousBatchStreamStepping
> {
    internal let configuration: MLXContinuousBatchExecutorConfig
    internal let makeStepper: (MLXContinuousBatchPrefillResult) throws -> Stepper
    internal let prefillRunner: PrefillRunner
    internal let queue: MLXContinuousBatchPrefillRequestQueue

    internal init(
        queue: MLXContinuousBatchPrefillRequestQueue,
        prefillRunner: PrefillRunner,
        configuration: MLXContinuousBatchExecutorConfig = .default,
        makeStepper: @escaping (MLXContinuousBatchPrefillResult) throws -> Stepper
    ) {
        self.configuration = configuration
        self.makeStepper = makeStepper
        self.prefillRunner = prefillRunner
        self.queue = queue
    }

    internal func runNextBatch() async throws -> MLXContinuousBatchPrefillExecutorResult? {
        let queuedRequests = try await queue.nextBatch(maxCount: configuration.maxBatchSize)
        guard !queuedRequests.isEmpty else {
            return nil
        }

        let groups: [[MLXContinuousBatchQueuedPrefillRequest]]
        do {
            groups = try MLXContinuousBatchPrefillBatcher.groups(forQueued: queuedRequests)
        } catch {
            fail(queuedRequests, error: error)
            throw error
        }

        var groupResults: [MLXContinuousBatchExecutorResult] = []
        groupResults.reserveCapacity(groups.count)
        for group in groups {
            groupResults.append(try runGroup(group))
        }

        return MLXContinuousBatchPrefillExecutorResult(
            requestIDs: queuedRequests.map(\.id),
            groupResults: groupResults
        )
    }

    private func runGroup(
        _ queuedRequests: [MLXContinuousBatchQueuedPrefillRequest]
    ) throws -> MLXContinuousBatchExecutorResult {
        let prefillResult = try runPrefill(queuedRequests)
        guard !prefillResult.batch.isEmpty else {
            return emptyResult(for: queuedRequests)
        }
        let stepper = try makeStepper(for: prefillResult)
        return try runStreamLoop(
            prefillResult: prefillResult,
            queuedRequests: queuedRequests,
            stepper: stepper
        )
    }

    private func runPrefill(
        _ queuedRequests: [MLXContinuousBatchQueuedPrefillRequest]
    ) throws -> MLXContinuousBatchPrefillResult {
        do {
            return try prefillRunner.run(requests: queuedRequests.map(\.request))
        } catch {
            fail(queuedRequests, error: error)
            throw error
        }
    }

    private func emptyResult(
        for queuedRequests: [MLXContinuousBatchQueuedPrefillRequest]
    ) -> MLXContinuousBatchExecutorResult {
        MLXContinuousBatchExecutorResult(
            requestIDs: queuedRequests.map(\.id),
            rowIDs: [],
            runLoopResult: .init(finishedRows: [], stepCount: 0, streamedTokenCount: 0)
        )
    }

    private func makeStepper(
        for prefillResult: MLXContinuousBatchPrefillResult
    ) throws -> Stepper {
        do {
            return try makeStepper(prefillResult)
        } catch {
            fail(prefillResult.batch.streamRows, error: error)
            throw error
        }
    }

    private func runStreamLoop(
        prefillResult: MLXContinuousBatchPrefillResult,
        queuedRequests: [MLXContinuousBatchQueuedPrefillRequest],
        stepper: Stepper
    ) throws -> MLXContinuousBatchExecutorResult {
        let driver = try MLXContinuousBatchStreamDriver(
            stepper: stepper,
            rows: prefillResult.batch.streamRows
        )
        var runLoop = MLXContinuousBatchRunLoop(
            driver: driver,
            configuration: configuration.runLoopConfiguration
        )
        let result = try runLoop.run()

        return MLXContinuousBatchExecutorResult(
            requestIDs: queuedRequests.map(\.id),
            rowIDs: prefillResult.batch.orderedRowIDs,
            runLoopResult: result
        )
    }

    private func fail(
        _ queuedRequests: [MLXContinuousBatchQueuedPrefillRequest],
        error: any Error
    ) {
        for queuedRequest in queuedRequests {
            queuedRequest.request.sink.fail(error)
        }
    }

    private func fail(
        _ streamRows: [MLXContinuousBatchStreamRow],
        error: any Error
    ) {
        for row in streamRows {
            row.fail(error)
        }
    }
}

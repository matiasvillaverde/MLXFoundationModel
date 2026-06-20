import Foundation

internal actor MLXContinuousBatchSessionEngine {
    private let configuration: MLXContinuousBatchExecutorConfig
    private let container: ModelContainer
    private let queue = MLXContinuousBatchPrefillRequestQueue()
    private var drainTask: Task<Void, Never>?

    internal init(
        container: ModelContainer,
        scheduling: MLXGenerationSchedulingConfiguration
    ) {
        self.container = container
        self.configuration = MLXContinuousBatchExecutorConfig(
            maxBatchSize: scheduling.maxBatchSize
        )
    }

    internal func enqueue(
        _ request: MLXContinuousBatchPrefillRequest
    ) async throws -> MLXContinuousBatchRequestID {
        let id = try await queue.enqueue(request)
        startDrainIfNeeded()
        return id
    }

    internal func cancel(id: MLXContinuousBatchRequestID) async {
        guard let queuedRequest = await queue.cancel(id: id) else {
            return
        }
        queuedRequest.request.sink.fail(CancellationError())
    }

    internal func close() async {
        drainTask?.cancel()
        drainTask = nil
        await queue.close()
    }

    private func startDrainIfNeeded() {
        guard drainTask == nil else {
            return
        }
        drainTask = Task {
            await self.drainLoop()
        }
    }

    private func drainLoop() async {
        do {
            try await coalescePendingRequests()
            while try await drainNextAvailableBatch() {
                try Task.checkCancellation()
                await Task.yield()
            }
        } catch is CancellationError {
            await failPending(CancellationError())
        } catch {
            await failPending(error)
        }
        await finishDrainLoop()
    }

    private func coalescePendingRequests() async throws {
        try await Task.sleep(nanoseconds: 1_000_000)
    }

    private func drainNextAvailableBatch() async throws -> Bool {
        guard await queue.snapshot().pendingCount > 0 else {
            return false
        }
        let queuedRequests = try await queue.nextBatch(maxCount: configuration.maxBatchSize)
        guard !queuedRequests.isEmpty else {
            return false
        }
        try await run(queuedRequests)
        return true
    }

    private func run(
        _ queuedRequests: [MLXContinuousBatchQueuedPrefillRequest]
    ) async throws {
        let groups: [[MLXContinuousBatchQueuedPrefillRequest]]
        do {
            groups = try MLXContinuousBatchPrefillBatcher.groups(forQueued: queuedRequests)
        } catch {
            fail(queuedRequests, error: error)
            throw error
        }

        for group in groups {
            do {
                try await run(group: group)
            } catch {
                fail(group, error: error)
            }
        }
    }

    private func run(
        group: [MLXContinuousBatchQueuedPrefillRequest]
    ) async throws {
        let runLoopConfiguration = configuration.runLoopConfiguration
        try await container.performWithPromptCache { context, promptCacheEntries in
            let result = try MLXContinuousBatchPrefill.run(
                model: context.model,
                requests: group.map(\.request),
                grammarCompiler: context.grammarCompiler
            )
            try Self.storePromptCaches(
                for: group,
                prefillResult: result,
                promptCacheEntries: &promptCacheEntries
            )
            guard !result.batch.isEmpty else {
                return
            }
            try Self.runDecodeLoop(
                prefillResult: result,
                parameters: group[0].request.parameters,
                context: context,
                runLoopConfiguration: runLoopConfiguration
            )
        }
    }

    private static func storePromptCaches(
        for group: [MLXContinuousBatchQueuedPrefillRequest],
        prefillResult: MLXContinuousBatchPrefillResult,
        promptCacheEntries: inout [PromptCacheEntry]
    ) throws {
        for rowIndex in group.indices {
            try group[rowIndex].request.promptCacheStorage?.store(
                cache: prefillResult.cache,
                rowIndex: rowIndex,
                rowCount: group.count,
                entries: &promptCacheEntries
            )
        }
    }

    private static func runDecodeLoop(
        prefillResult: MLXContinuousBatchPrefillResult,
        parameters: GenerateParameters,
        context: ModelContext,
        runLoopConfiguration: MLXContinuousBatchRunLoopConfiguration
    ) throws {
        let scheduler = MLXContinuousBatchGenerationScheduler(
            coordinator: prefillResult.batch.coordinator,
            decoder: MLXContinuousBatchDecodeStep(
                model: context.model,
                cache: prefillResult.cache,
                logitRows: prefillResult.logitRows,
                kvBits: parameters.kvBits,
                kvGroupSize: parameters.kvGroupSize,
                quantizedKVStart: parameters.quantizedKVStart,
                quantizedKVSkipLastLayer: parameters.quantizedKVSkipLastLayer
            )
        )
        let driver = try MLXContinuousBatchStreamDriver(
            stepper: scheduler,
            rows: prefillResult.batch.streamRows
        )
        var runLoop = MLXContinuousBatchRunLoop(
            driver: driver,
            configuration: runLoopConfiguration
        )
        _ = try runLoop.run {
            Task.isCancelled
        }
    }

    private func failPending(_ error: any Error) async {
        await queue.close()
        while true {
            let queuedRequests = (try? await queue.nextBatch(maxCount: configuration.maxBatchSize)) ?? []
            guard !queuedRequests.isEmpty else {
                return
            }
            fail(queuedRequests, error: error)
        }
    }

    private func fail(
        _ queuedRequests: [MLXContinuousBatchQueuedPrefillRequest],
        error: any Error
    ) {
        for queuedRequest in queuedRequests {
            queuedRequest.request.sink.fail(error)
        }
    }

    private func finishDrainLoop() async {
        drainTask = nil
        if await queue.snapshot().pendingCount > 0 {
            startDrainIfNeeded()
        }
    }
}

internal actor MLXContinuousBatchPrefillRequestQueue {
    private struct Waiter {
        let id: Int
        let continuation: CheckedContinuation<[MLXContinuousBatchQueuedPrefillRequest], any Error>
        let maxCount: Int
    }

    private var isClosed = false
    private var nextID = MLXContinuousBatchRequestID(0)
    private var nextWaiterID = 0
    private var pending: [MLXContinuousBatchQueuedPrefillRequest] = []
    private var waiters: [Waiter] = []

    @discardableResult
    internal func enqueue(
        _ request: MLXContinuousBatchPrefillRequest
    ) throws -> MLXContinuousBatchRequestID {
        guard !isClosed else {
            throw MLXContinuousBatchRequestQueueError.closed
        }

        let id = allocateID()
        pending.append(.init(id: id, request: request))
        resumeWaitersIfPossible()
        return id
    }

    internal func cancel(
        id: MLXContinuousBatchRequestID
    ) -> MLXContinuousBatchQueuedPrefillRequest? {
        guard let index = pending.firstIndex(where: { $0.id == id }) else {
            return nil
        }
        return pending.remove(at: index)
    }

    internal func nextBatch(
        maxCount: Int
    ) async throws -> [MLXContinuousBatchQueuedPrefillRequest] {
        let count = normalizedBatchCount(maxCount)
        if !pending.isEmpty {
            return drainBatch(maxCount: count)
        }
        if isClosed {
            return []
        }

        let waiterID = allocateWaiterID()
        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                waiters.append(.init(id: waiterID, continuation: continuation, maxCount: count))
            }
        } onCancel: {
            Task {
                await self.cancelWaiter(id: waiterID)
            }
        }
    }

    internal func close() {
        isClosed = true
        let waiters = self.waiters
        self.waiters.removeAll(keepingCapacity: true)
        for waiter in waiters {
            waiter.continuation.resume(returning: [])
        }
    }

    internal func snapshot() -> MLXContinuousBatchRequestQueueSnapshot {
        .init(
            isClosed: isClosed,
            pendingCount: pending.count,
            waitingConsumerCount: waiters.count
        )
    }

    private func normalizedBatchCount(_ value: Int) -> Int {
        max(1, value)
    }

    private func allocateID() -> MLXContinuousBatchRequestID {
        defer {
            nextID = MLXContinuousBatchRequestID(nextID.rawValue + 1)
        }
        return nextID
    }

    private func allocateWaiterID() -> Int {
        defer {
            nextWaiterID += 1
        }
        return nextWaiterID
    }

    private func cancelWaiter(id: Int) {
        guard let index = waiters.firstIndex(where: { $0.id == id }) else {
            return
        }
        let waiter = waiters.remove(at: index)
        waiter.continuation.resume(throwing: CancellationError())
    }

    private func drainBatch(maxCount: Int) -> [MLXContinuousBatchQueuedPrefillRequest] {
        let count = min(maxCount, pending.count)
        let batch = Array(pending.prefix(count))
        pending.removeFirst(count)
        return batch
    }

    private func resumeWaitersIfPossible() {
        while !pending.isEmpty, !waiters.isEmpty {
            let waiter = waiters.removeFirst()
            waiter.continuation.resume(returning: drainBatch(maxCount: waiter.maxCount))
        }
    }
}

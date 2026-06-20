@testable import MLXLocalModels
import Testing

@Suite("MLX continuous batch request queue")
struct MLXContinuousBatchRequestQueueTests {
    @Test("drains FIFO batches up to requested capacity")
    func drainsFIFOBatchesUpToRequestedCapacity() async throws {
        let queue = MLXContinuousBatchRequestQueue()
        let first = try await queue.enqueue(Self.request(previousTokenID: 10))
        let second = try await queue.enqueue(Self.request(previousTokenID: 20))
        let third = try await queue.enqueue(Self.request(previousTokenID: 30))

        let batch = try await queue.nextBatch(maxCount: 2)
        let snapshot = await queue.snapshot()

        #expect(batch.map(\.id) == [first, second])
        #expect(batch.map(\.request.generationRow.previousTokenID) == [10, 20])
        #expect(snapshot.pendingCount == 1)
        #expect(snapshot.waitingConsumerCount == 0)

        let remainder = try await queue.nextBatch(maxCount: 2)
        #expect(remainder.map(\.id) == [third])
    }

    @Test("waiting consumer receives the next request")
    func waitingConsumerReceivesTheNextRequest() async throws {
        let queue = MLXContinuousBatchRequestQueue()
        let task = Task {
            try await queue.nextBatch(maxCount: 2)
        }

        try await Task.sleep(nanoseconds: 10_000_000)
        #expect(await queue.snapshot().waitingConsumerCount == 1)

        let id = try await queue.enqueue(Self.request(previousTokenID: 42))
        let batch = try await task.value

        #expect(batch.map(\.id) == [id])
        #expect(batch.map(\.request.generationRow.previousTokenID) == [42])
        #expect(await queue.snapshot().pendingCount == 0)
    }

    @Test("cancels queued requests before drain")
    func cancelsQueuedRequestsBeforeDrain() async throws {
        let queue = MLXContinuousBatchRequestQueue()
        let first = try await queue.enqueue(Self.request(previousTokenID: 10))
        let second = try await queue.enqueue(Self.request(previousTokenID: 20))

        let cancelled = await queue.cancel(id: first)
        let batch = try await queue.nextBatch(maxCount: 2)

        #expect(cancelled?.id == first)
        #expect(batch.map(\.id) == [second])
        #expect(await queue.snapshot().pendingCount == 0)
    }

    @Test("close wakes waiting consumers and rejects new requests")
    func closeWakesWaitingConsumersAndRejectsNewRequests() async throws {
        let queue = MLXContinuousBatchRequestQueue()
        let task = Task {
            try await queue.nextBatch(maxCount: 1)
        }

        try await Task.sleep(nanoseconds: 10_000_000)
        await queue.close()
        let batch = try await task.value

        #expect(batch.isEmpty)
        #expect(await queue.snapshot().isClosed)
        await Self.expectClosed(queue)
    }

    @Test("cancels waiting consumers without leaking waiters")
    func cancelsWaitingConsumersWithoutLeakingWaiters() async throws {
        let queue = MLXContinuousBatchRequestQueue()
        let task = Task {
            try await queue.nextBatch(maxCount: 1)
        }

        try await Task.sleep(nanoseconds: 10_000_000)
        #expect(await queue.snapshot().waitingConsumerCount == 1)

        task.cancel()
        do {
            _ = try await task.value
            Issue.record("Expected waiting consumer cancellation")
        } catch is CancellationError {
            #expect(true)
        } catch {
            Issue.record("Unexpected error: \(error)")
        }

        #expect(await queue.snapshot().waitingConsumerCount == 0)
        _ = try await queue.enqueue(Self.request(previousTokenID: 7))
        let batch = try await queue.nextBatch(maxCount: 1)
        #expect(batch.map(\.request.generationRow.previousTokenID) == [7])
    }

    private static func request(
        previousTokenID: Int
    ) -> MLXContinuousBatchGenerationRequest {
        MLXContinuousBatchGenerationRequest(
            generationRow: .init(
                previousTokenID: previousTokenID,
                maximumTokenCount: 2
            ),
            tokenText: { "token-\($0)" },
            sink: RecordingBatchSink().streamSink()
        )
    }

    private static func expectClosed(
        _ queue: MLXContinuousBatchRequestQueue
    ) async {
        do {
            _ = try await queue.enqueue(Self.request(previousTokenID: 99))
            Issue.record("Expected closed queue to reject enqueue")
        } catch MLXContinuousBatchRequestQueueError.closed {
            #expect(true)
        } catch {
            Issue.record("Unexpected error: \(error)")
        }
    }
}

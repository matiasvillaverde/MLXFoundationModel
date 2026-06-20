@testable import MLXLocalModels
import Testing

@Suite("MLX generation admission controller")
struct MLXGenerationAdmissionControllerTests {
    @Test("queues requests in FIFO order when capacity is full")
    func queuesRequestsInFIFOOrderWhenCapacityIsFull() async throws {
        let controller = MLXGenerationAdmissionController(maxConcurrentRequests: 1)
        let first = try await controller.acquire()
        let secondTask = Task {
            try await controller.acquire()
        }
        let thirdTask = Task {
            try await controller.acquire()
        }

        try await Task.sleep(nanoseconds: 10_000_000)
        #expect(await controller.snapshot().waitingCount == 2)

        await controller.release(first)
        let second = try await secondTask.value
        #expect(await controller.snapshot().activeCount == 1)
        #expect(await controller.snapshot().waitingCount == 1)

        await controller.release(second)
        let third = try await thirdTask.value
        #expect(await controller.snapshot().activeCount == 1)
        #expect(await controller.snapshot().waitingCount == 0)

        await controller.release(third)
        #expect(await controller.snapshot().activeCount == 0)
    }

    @Test("allows configured parallel capacity")
    func allowsConfiguredParallelCapacity() async throws {
        let controller = MLXGenerationAdmissionController(maxConcurrentRequests: 2)
        let first = try await controller.acquire()
        let second = try await controller.acquire()
        let thirdTask = Task {
            try await controller.acquire()
        }

        try await Task.sleep(nanoseconds: 10_000_000)
        let saturated = await controller.snapshot()
        #expect(saturated.activeCount == 2)
        #expect(saturated.waitingCount == 1)
        #expect(saturated.maxConcurrentRequests == 2)
        #expect(saturated.maxQueuedRequests == 32)
        #expect(saturated.maxBatchSize == 2)

        await controller.release(first)
        let third = try await thirdTask.value
        #expect(await controller.snapshot().activeCount == 2)

        await controller.release(second)
        await controller.release(third)
        #expect(await controller.snapshot().activeCount == 0)
    }

    @Test("cancels a queued request without consuming capacity")
    func cancelsQueuedRequestWithoutConsumingCapacity() async throws {
        let controller = MLXGenerationAdmissionController(maxConcurrentRequests: 1)
        let first = try await controller.acquire()
        let queuedTask = Task {
            try await controller.acquire()
        }

        try await Task.sleep(nanoseconds: 10_000_000)
        queuedTask.cancel()

        do {
            _ = try await queuedTask.value
            Issue.record("Queued admission should have thrown CancellationError")
        } catch is CancellationError {
            // Expected cancellation path.
        }

        let snapshot = await controller.snapshot()
        #expect(snapshot.activeCount == 1)
        #expect(snapshot.waitingCount == 0)

        await controller.release(first)
        #expect(await controller.snapshot().activeCount == 0)
    }

    @Test("rejects new requests when the waiting queue is full")
    func rejectsNewRequestsWhenWaitingQueueIsFull() async throws {
        let controller = MLXGenerationAdmissionController(
            configuration: .init(maxConcurrentRequests: 1, maxQueuedRequests: 1)
        )
        let first = try await controller.acquire()
        let queuedTask = Task {
            try await controller.acquire()
        }

        try await Task.sleep(nanoseconds: 10_000_000)
        try await Self.expectQueueFull(from: controller)

        let snapshot = await controller.snapshot()
        #expect(snapshot.activeCount == 1)
        #expect(snapshot.waitingCount == 1)
        #expect(snapshot.maxQueuedRequests == 1)

        await controller.release(first)
        let queuedLease = try await queuedTask.value
        await controller.release(queuedLease)
    }

    @Test("acquires immediate batches up to capacity and configured batch size")
    func acquiresImmediateBatchesUpToCapacityAndConfiguredBatchSize() async throws {
        let recorded = try await MLXGenerationDiagnostics.withRecording {
            let controller = MLXGenerationAdmissionController(
                configuration: .init(
                    maxConcurrentRequests: 4,
                    maxQueuedRequests: 2,
                    maxBatchSize: 3
                )
            )

            let batch = try await controller.acquireBatch(upTo: 8)
            let snapshot = await controller.snapshot()

            #expect(batch.count == 3)
            #expect(snapshot.activeCount == 3)
            #expect(snapshot.maxBatchSize == 3)

            for lease in batch {
                await controller.release(lease)
            }
        }

        let batchSnapshot = try #require(Self.admissionSnapshots(from: recorded.events).first { snapshot in
            snapshot.stage == .batchAdmitted
        })
        #expect(batchSnapshot.admittedCount == 3)
        #expect(batchSnapshot.maxBatchSize == 3)
    }

    @Test("queued batch acquisition preserves FIFO fairness")
    func queuedBatchAcquisitionPreservesFIFOFairness() async throws {
        let controller = MLXGenerationAdmissionController(
            configuration: .init(maxConcurrentRequests: 2, maxQueuedRequests: 4, maxBatchSize: 2)
        )
        let first = try await controller.acquire()
        let second = try await controller.acquire()
        let batchTask = Task {
            try await controller.acquireBatch(upTo: 2)
        }

        try await Task.sleep(nanoseconds: 10_000_000)
        #expect(await controller.snapshot().waitingCount == 1)

        await controller.release(first)
        let admitted = try await batchTask.value
        #expect(admitted.count == 1)
        #expect(await controller.snapshot().activeCount == 2)

        await controller.release(second)
        await controller.release(admitted[0])
    }

    @Test("queued batch acquisition admits grouped leases when admission resumes")
    func queuedBatchAcquisitionAdmitsGroupedLeasesWhenAdmissionResumes() async throws {
        let recorded = try await MLXGenerationDiagnostics.withRecording {
            let controller = MLXGenerationAdmissionController(
                configuration: .init(maxConcurrentRequests: 3, maxQueuedRequests: 4, maxBatchSize: 2)
            )
            let active = try await controller.acquire()
            await controller.setAdmissionPaused(true)
            let batchTask = Task {
                try await controller.acquireBatch(upTo: 3)
            }

            try await Task.sleep(nanoseconds: 10_000_000)
            #expect(await controller.snapshot().waitingCount == 1)

            await controller.release(active)
            #expect(await controller.snapshot().activeCount == 0)

            await controller.setAdmissionPaused(false)
            let admitted = try await batchTask.value
            let snapshot = await controller.snapshot()

            #expect(admitted.count == 2)
            #expect(snapshot.activeCount == 2)
            #expect(snapshot.waitingCount == 0)

            for lease in admitted {
                await controller.release(lease)
            }
        }

        let batchSnapshot = try #require(Self.admissionSnapshots(from: recorded.events).first { snapshot in
            snapshot.stage == .batchAdmitted && snapshot.admittedCount == 2
        })
        #expect(batchSnapshot.maxBatchSize == 2)
    }

    @Test("cancels a queued batch request without consuming capacity")
    func cancelsQueuedBatchRequestWithoutConsumingCapacity() async throws {
        let controller = MLXGenerationAdmissionController(
            configuration: .init(maxConcurrentRequests: 1, maxQueuedRequests: 2, maxBatchSize: 2)
        )
        let active = try await controller.acquire()
        let batchTask = Task {
            try await controller.acquireBatch(upTo: 2)
        }

        try await Task.sleep(nanoseconds: 10_000_000)
        batchTask.cancel()

        do {
            _ = try await batchTask.value
            Issue.record("Queued batch admission should have thrown CancellationError")
        } catch is CancellationError {
            // Expected cancellation path.
        }

        let snapshot = await controller.snapshot()
        #expect(snapshot.activeCount == 1)
        #expect(snapshot.waitingCount == 0)

        await controller.release(active)
        #expect(await controller.snapshot().activeCount == 0)
    }

    @Test("pause holds new admissions while active work drains")
    func pauseHoldsNewAdmissionsWhileActiveWorkDrains() async throws {
        let controller = MLXGenerationAdmissionController(maxConcurrentRequests: 2)
        let first = try await controller.acquire()
        await controller.setAdmissionPaused(true)

        let queuedTask = Task {
            try await controller.acquire()
        }
        try await Task.sleep(nanoseconds: 10_000_000)

        let paused = await controller.snapshot()
        #expect(paused.activeCount == 1)
        #expect(paused.waitingCount == 1)
        #expect(paused.admissionPaused)

        await controller.release(first)
        try await Task.sleep(nanoseconds: 10_000_000)

        let drained = await controller.snapshot()
        #expect(drained.activeCount == 0)
        #expect(drained.waitingCount == 1)
        #expect(drained.admissionPaused)

        await controller.setAdmissionPaused(false)
        let recoveryLease = try await queuedTask.value
        #expect(await controller.snapshot().activeCount == 1)

        await controller.release(recoveryLease)
    }

    @Test("records admission diagnostics")
    func recordsAdmissionDiagnostics() async throws {
        let recorded = try await MLXGenerationDiagnostics.withRecording {
            let controller = MLXGenerationAdmissionController(
                configuration: .init(maxConcurrentRequests: 1, maxQueuedRequests: 0)
            )
            let first = try await controller.acquire()
            try await Self.expectQueueFull(from: controller)
            await controller.release(first)
        }

        let stages = Self.admissionSnapshots(from: recorded.events).map(\.stage)
        #expect(stages.contains(.admitted))
        #expect(stages.contains(.queueFull))
        #expect(stages.contains(.released))
    }

    @Test("updates scheduling configuration")
    func updatesSchedulingConfiguration() async throws {
        let controller = MLXGenerationAdmissionController(
            configuration: .init(maxConcurrentRequests: 1, maxQueuedRequests: 1)
        )
        await controller.updateConfiguration(.init(maxConcurrentRequests: 3, maxQueuedRequests: 8))

        let snapshot = await controller.snapshot()
        #expect(snapshot.maxConcurrentRequests == 3)
        #expect(snapshot.maxQueuedRequests == 8)
    }

    private static func expectQueueFull(
        from controller: MLXGenerationAdmissionController
    ) async throws {
        do {
            _ = try await controller.acquire()
            Issue.record("Expected queue-full admission failure")
        } catch MLXGenerationAdmissionError.queueFull(let currentDepth, let maxDepth) {
            #expect(currentDepth == maxDepth)
        }
    }

    private static func admissionSnapshots(
        from events: [MLXGenerationDiagnosticEvent]
    ) -> [MLXGenerationAdmissionSnapshot] {
        events.compactMap { event in
            guard case .admission(let snapshot) = event else {
                return nil
            }
            return snapshot
        }
    }
}

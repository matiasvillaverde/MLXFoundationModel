@testable import MLXLocalModels
import Testing

@Suite("MLX continuous batch queue executor")
struct MLXContinuousBatchQueueExecutorTests {
    @Test("drains queued requests into one batch run")
    func drainsQueuedRequestsIntoOneBatchRun() async throws {
        let queue = MLXContinuousBatchRequestQueue()
        let firstSink = RecordingBatchSink()
        let secondSink = RecordingBatchSink()
        let firstID = try await queue.enqueue(Self.request(previousTokenID: 1, sink: firstSink))
        let secondID = try await queue.enqueue(Self.request(previousTokenID: 2, sink: secondSink))
        let executor = Self.executor(queue: queue, maxBatchSize: 2)

        let result = try await #require(executor.runNextBatch())

        #expect(result.requestIDs == [firstID, secondID])
        #expect(result.rowIDs == [0, 1])
        #expect(result.runLoopResult.stepCount == 1)
        #expect(result.runLoopResult.streamedTokenCount == 2)
        #expect(firstSink.texts() == ["token-10"])
        #expect(secondSink.texts() == ["token-20"])
        #expect(firstSink.finishReasons() == [.maximumTokenCount])
        #expect(secondSink.finishReasons() == [.maximumTokenCount])
    }

    @Test("returns nil after queue close")
    func returnsNilAfterQueueClose() async throws {
        let queue = MLXContinuousBatchRequestQueue()
        await queue.close()
        let executor = Self.executor(queue: queue, maxBatchSize: 2)

        let result = try await executor.runNextBatch()

        #expect(result == nil)
    }

    @Test("stepper factory failure fails queued sinks")
    func stepperFactoryFailureFailsQueuedSinks() async throws {
        let queue = MLXContinuousBatchRequestQueue()
        let sink = RecordingBatchSink()
        _ = try await queue.enqueue(Self.request(previousTokenID: 1, sink: sink))
        let executor: MLXContinuousBatchQueueExecutor<ScriptedStreamStepper> =
            MLXContinuousBatchQueueExecutor(
                queue: queue,
                configuration: .init(maxBatchSize: 1)
            ) { _ in
                throw ScriptedStreamStepError.failed
            }

        do {
            _ = try await executor.runNextBatch()
            Issue.record("Expected stepper factory failure")
        } catch ScriptedStreamStepError.failed {
            #expect(true)
        } catch {
            Issue.record("Unexpected error: \(error)")
        }
        #expect(sink.failureCount() == 1)
        #expect(await queue.snapshot().pendingCount == 0)
    }

    private static func executor(
        queue: MLXContinuousBatchRequestQueue,
        maxBatchSize: Int
    ) -> MLXContinuousBatchQueueExecutor<ScriptedStreamStepper> {
        MLXContinuousBatchQueueExecutor(
            queue: queue,
            configuration: .init(maxBatchSize: maxBatchSize)
        ) { batch in
            let rowIDs = batch.orderedRowIDs
            return ScriptedStreamStepper(
                orderedRowIDs: rowIDs,
                steps: [
                    .init(
                        sampledRowIDs: rowIDs,
                        sampledTokenIDs: rowIDs.indices.map { 10 + ($0 * 10) },
                        finishedRows: rowIDs.enumerated().map { index, rowID in
                            .init(
                                rowID: rowID,
                                tokenID: 10 + (index * 10),
                                generatedTokenCount: 1,
                                reason: .maximumTokenCount
                            )
                        },
                        activeRowIDs: []
                    )
                ]
            )
        }
    }

    private static func request(
        previousTokenID: Int,
        sink: RecordingBatchSink
    ) -> MLXContinuousBatchGenerationRequest {
        MLXContinuousBatchGenerationRequest(
            generationRow: .init(
                previousTokenID: previousTokenID,
                maximumTokenCount: 1
            ),
            tokenText: { "token-\($0)" },
            sink: sink.streamSink()
        )
    }
}

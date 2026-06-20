@testable import MLXLocalModels
import Testing

@Suite("MLX continuous batch prefill queue executor")
struct MLXContinuousBatchPrefillQTests {
    @Test("drains queued prompts through homogeneous prefill groups")
    func drainsQueuedPromptsThroughHomogeneousPrefillGroups() async throws {
        let queue = MLXContinuousBatchPrefillRequestQueue()
        let firstSink = RecordingBatchSink()
        let secondSink = RecordingBatchSink()
        let thirdSink = RecordingBatchSink()
        let firstID = try await queue.enqueue(Self.request(tokens: [1, 2], sink: firstSink))
        let secondID = try await queue.enqueue(Self.request(tokens: [3], sink: secondSink))
        let thirdID = try await queue.enqueue(Self.request(tokens: [4, 5], sink: thirdSink))
        let executor = Self.executor(queue: queue, maxBatchSize: 3)

        let result = try await #require(executor.runNextBatch())

        #expect(result.requestIDs == [firstID, secondID, thirdID])
        #expect(result.groupResults.map(\.requestIDs) == [
            [firstID, thirdID],
            [secondID]
        ])
        #expect(result.groupResults.map(\.rowIDs) == [[0, 1], [0]])
        #expect(firstSink.texts() == ["token-100"])
        #expect(secondSink.texts() == ["token-100"])
        #expect(thirdSink.texts() == ["token-101"])
        #expect(firstSink.finishReasons() == [.maximumTokenCount])
        #expect(secondSink.finishReasons() == [.maximumTokenCount])
        #expect(thirdSink.finishReasons() == [.maximumTokenCount])
    }

    @Test("prefill failure fails all requests in the affected group")
    func prefillFailureFailsAffectedGroup() async throws {
        let queue = MLXContinuousBatchPrefillRequestQueue()
        let sink = RecordingBatchSink()
        _ = try await queue.enqueue(Self.request(tokens: [1, 2], sink: sink))
        let executor = MLXContinuousBatchPrefillQueueExecutor(
            queue: queue,
            prefillRunner: FailingPrefillRunner(),
            configuration: .init(maxBatchSize: 1)
        ) { _ in
            ScriptedStreamStepper(orderedRowIDs: [], steps: [])
        }

        do {
            _ = try await executor.runNextBatch()
            Issue.record("Expected prefill failure")
        } catch ScriptedStreamStepError.failed {
            #expect(true)
        } catch {
            Issue.record("Unexpected error: \(error)")
        }
        #expect(sink.failureCount() == 1)
        #expect(await queue.snapshot().pendingCount == 0)
    }

    @Test("stepper factory failure fails active rows after successful prefill")
    func stepperFactoryFailureFailsActiveRows() async throws {
        let queue = MLXContinuousBatchPrefillRequestQueue()
        let sink = RecordingBatchSink()
        _ = try await queue.enqueue(Self.request(tokens: [1, 2], sink: sink))
        let executor: MLXContinuousBatchPrefillQueueExecutor<
            ScriptedPrefillRunner,
            ScriptedStreamStepper
        > = MLXContinuousBatchPrefillQueueExecutor(
            queue: queue,
            prefillRunner: ScriptedPrefillRunner(),
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
    }

    @Test("returns nil after queue close")
    func returnsNilAfterQueueClose() async throws {
        let queue = MLXContinuousBatchPrefillRequestQueue()
        await queue.close()
        let executor = Self.executor(queue: queue, maxBatchSize: 2)

        let result = try await executor.runNextBatch()

        #expect(result == nil)
    }

    private static func executor(
        queue: MLXContinuousBatchPrefillRequestQueue,
        maxBatchSize: Int
    ) -> MLXContinuousBatchPrefillQueueExecutor<ScriptedPrefillRunner, ScriptedStreamStepper> {
        MLXContinuousBatchPrefillQueueExecutor(
            queue: queue,
            prefillRunner: ScriptedPrefillRunner(),
            configuration: .init(maxBatchSize: maxBatchSize)
        ) { prefillResult in
            let rowIDs = prefillResult.batch.orderedRowIDs
            return ScriptedStreamStepper(
                orderedRowIDs: rowIDs,
                steps: [
                    .init(
                        sampledRowIDs: rowIDs,
                        sampledTokenIDs: rowIDs.indices.map { 100 + $0 },
                        finishedRows: rowIDs.enumerated().map { index, rowID in
                            .init(
                                rowID: rowID,
                                tokenID: 100 + index,
                                generatedTokenCount: 2,
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
        tokens: [Int],
        sink: RecordingBatchSink
    ) -> MLXContinuousBatchPrefillRequest {
        MLXContinuousBatchPrefillRequest(
            promptTokenIDs: tokens,
            parameters: GenerateParameters(maxTokens: 2),
            tokenText: { "token-\($0)" },
            sink: sink.streamSink()
        )
    }
}

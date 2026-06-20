@testable import MLXLocalModels
import Testing

@Suite("MLX continuous batch run loop")
struct MLXContinuousBatchRunLoopTests {
    @Test("runs until all rows finish")
    func runsUntilAllRowsFinish() throws {
        let rowID = MLXGenerationBatchRowID(0)
        let sink = RecordingBatchSink()
        var loop = try Self.makeLoop(
            rowID: rowID,
            sink: sink,
            steps: Self.twoTokenFinishSteps(rowID: rowID)
        )

        let result = try loop.run {
            false
        }

        #expect(result.stepCount == 2)
        #expect(result.streamedTokenCount == 2)
        #expect(result.finishedRows.map(\.rowID) == [rowID])
        #expect(sink.texts() == ["token-10", "token-11"])
        #expect(sink.finishReasons() == [.maximumTokenCount])
    }

    @Test("cancellation fails active rows before the next step")
    func cancellationFailsActiveRowsBeforeTheNextStep() throws {
        let rowID = MLXGenerationBatchRowID(0)
        let sink = RecordingBatchSink()
        var loop = try Self.makeLoop(
            rowID: rowID,
            sink: sink,
            steps: Self.twoTokenFinishSteps(rowID: rowID)
        )

        #expect(throws: CancellationError.self) {
            try loop.run {
                true
            }
        }
        #expect(sink.texts().isEmpty)
        #expect(sink.failureCount() == 1)
    }

    @Test("step limit fails active rows")
    func stepLimitFailsActiveRows() throws {
        let rowID = MLXGenerationBatchRowID(0)
        let sink = RecordingBatchSink()
        var loop = try Self.makeLoop(
            rowID: rowID,
            sink: sink,
            steps: Self.unfinishedSteps(rowID: rowID),
            maximumStepCount: 1
        )

        #expect(throws: MLXContinuousBatchRunLoopError.self) {
            try loop.run {
                false
            }
        }
        #expect(sink.texts() == ["token-10"])
        #expect(sink.failureCount() == 1)
    }

    private static func makeLoop(
        rowID: MLXGenerationBatchRowID,
        sink: RecordingBatchSink,
        steps: [MLXContinuousBatchStreamStep],
        maximumStepCount: Int = 4_096
    ) throws -> MLXContinuousBatchRunLoop<ScriptedStreamStepper> {
        let driver = try MLXContinuousBatchStreamDriver(
            stepper: ScriptedStreamStepper(orderedRowIDs: [rowID], steps: steps),
            rows: [Self.row(id: rowID, sink: sink)]
        )
        return MLXContinuousBatchRunLoop(
            driver: driver,
            configuration: .init(maximumStepCount: maximumStepCount)
        )
    }

    private static func twoTokenFinishSteps(
        rowID: MLXGenerationBatchRowID
    ) -> [MLXContinuousBatchStreamStep] {
        [
            .init(sampledRowIDs: [rowID], sampledTokenIDs: [10], finishedRows: [], activeRowIDs: [rowID]),
            .init(
                sampledRowIDs: [rowID],
                sampledTokenIDs: [11],
                finishedRows: [
                    .init(
                        rowID: rowID,
                        tokenID: 11,
                        generatedTokenCount: 2,
                        reason: .maximumTokenCount
                    )
                ],
                activeRowIDs: []
            )
        ]
    }

    private static func unfinishedSteps(
        rowID: MLXGenerationBatchRowID
    ) -> [MLXContinuousBatchStreamStep] {
        [
            .init(sampledRowIDs: [rowID], sampledTokenIDs: [10], finishedRows: [], activeRowIDs: [rowID])
        ]
    }

    private static func row(
        id: MLXGenerationBatchRowID,
        sink: RecordingBatchSink
    ) -> MLXContinuousBatchStreamRow {
        MLXContinuousBatchStreamRow(
            id: id,
            tokenText: { "token-\($0)" },
            sink: sink.streamSink()
        )
    }
}

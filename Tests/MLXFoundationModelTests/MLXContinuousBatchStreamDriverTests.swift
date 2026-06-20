import Foundation
@testable import MLXLocalModels
import Testing

@Suite("MLX continuous batch stream driver")
struct MLXContinuousBatchStreamDriverTests {
    @Test("streams sampled tokens to matching rows and finishes independently")
    func streamsSampledTokensToMatchingRowsAndFinishesIndependently() throws {
        let first = MLXGenerationBatchRowID(0)
        let second = MLXGenerationBatchRowID(1)
        let firstSink = RecordingBatchSink()
        let secondSink = RecordingBatchSink()
        var driver = try Self.makeDriver(
            initialRowIDs: [first, second],
            steps: Self.independentFinishSteps(first: first, second: second),
            rows: [
                Self.row(id: first, sink: firstSink),
                Self.row(id: second, sink: secondSink)
            ]
        )

        let firstSummary = try driver.step()
        let secondSummary = try driver.step()

        #expect(firstSink.texts() == ["token-10"])
        #expect(secondSink.texts() == ["token-20"])
        #expect(firstSink.tokenIDs() == [[10]])
        #expect(secondSink.tokenIDs() == [[20]])
        #expect(firstSink.finishReasons() == [.stopToken(11)])
        #expect(secondSink.finishReasons() == [.maximumTokenCount])
        #expect(firstSummary.streamedTokenIDsByRowID == [first: 10, second: 20])
        #expect(secondSummary.streamedTokenIDsByRowID.isEmpty)
        #expect(secondSummary.finishedRows.map(\.rowID) == [first])
        #expect(driver.isEmpty)
    }

    @Test("stream-requested row finish realigns active rows")
    func streamRequestedRowFinishRealignsActiveRows() throws {
        let first = MLXGenerationBatchRowID(0)
        let second = MLXGenerationBatchRowID(1)
        let firstSink = RecordingBatchSink()
        let secondSink = RecordingBatchSink()
        var driver = try Self.makeDriver(
            initialRowIDs: [first, second],
            steps: Self.streamRequestedFinishSteps(first: first, second: second),
            rows: Self.streamRequestedFinishRows(
                first: first,
                second: second,
                firstSink: firstSink,
                secondSink: secondSink
            )
        )

        let firstSummary = try driver.step()
        let secondSummary = try driver.step()

        #expect(firstSink.texts() == ["token-10", "token-11"])
        #expect(secondSink.texts() == ["token-20"])
        #expect(firstSink.tokenIDs() == [[10], [11]])
        #expect(secondSink.tokenIDs() == [[20]])
        #expect(secondSink.finishReasons() == [.streamRequestedStop(20)])
        #expect(firstSummary.activeRowIDs == [first])
        #expect(firstSummary.finishedRows.map(\.rowID) == [second])
        #expect(firstSummary.streamedTokenIDsByRowID == [first: 10, second: 20])
        #expect(secondSummary.finishedRows.map(\.rowID) == [first])
        #expect(driver.isEmpty)
    }

    @Test("rejects duplicate row sinks")
    func rejectsDuplicateRowSinks() throws {
        let rowID = MLXGenerationBatchRowID(0)
        let sink = RecordingBatchSink()

        #expect(throws: MLXContinuousBatchStreamDriverError.self) {
            _ = try Self.makeDriver(
                initialRowIDs: [rowID],
                steps: [],
                rows: [
                    Self.row(id: rowID, sink: sink),
                    Self.row(id: rowID, sink: sink)
                ]
            )
        }
    }

    @Test("fails every active row when the stepper throws")
    func failsEveryActiveRowWhenTheStepperThrows() throws {
        let first = MLXGenerationBatchRowID(0)
        let second = MLXGenerationBatchRowID(1)
        let firstSink = RecordingBatchSink()
        let secondSink = RecordingBatchSink()
        var driver = try Self.makeDriver(
            initialRowIDs: [first, second],
            steps: [],
            rows: [
                Self.row(id: first, sink: firstSink),
                Self.row(id: second, sink: secondSink)
            ],
            error: ScriptedStreamStepError.failed
        )

        #expect(throws: ScriptedStreamStepError.self) {
            try driver.step()
        }
        #expect(firstSink.failureCount() == 1)
        #expect(secondSink.failureCount() == 1)
        #expect(driver.isEmpty)
    }

    private static func independentFinishSteps(
        first: MLXGenerationBatchRowID,
        second: MLXGenerationBatchRowID
    ) -> [MLXContinuousBatchStreamStep] {
        [
            .init(
                sampledRowIDs: [first, second],
                sampledTokenIDs: [10, 20],
                finishedRows: [
                    .init(
                        rowID: second,
                        tokenID: 20,
                        generatedTokenCount: 1,
                        reason: .maximumTokenCount
                    )
                ],
                activeRowIDs: [first]
            ),
            .init(
                sampledRowIDs: [first],
                sampledTokenIDs: [11],
                finishedRows: [
                    .init(
                        rowID: first,
                        tokenID: 11,
                        generatedTokenCount: 2,
                        reason: .stopToken(11)
                    )
                ],
                activeRowIDs: []
            )
        ]
    }

    private static func streamRequestedFinishSteps(
        first: MLXGenerationBatchRowID,
        second: MLXGenerationBatchRowID
    ) -> [MLXContinuousBatchStreamStep] {
        [
            .init(
                sampledRowIDs: [first, second],
                sampledTokenIDs: [10, 20],
                finishedRows: [],
                activeRowIDs: [first, second]
            ),
            .init(
                sampledRowIDs: [first],
                sampledTokenIDs: [11],
                finishedRows: [
                    .init(
                        rowID: first,
                        tokenID: 11,
                        generatedTokenCount: 2,
                        reason: .maximumTokenCount
                    )
                ],
                activeRowIDs: []
            )
        ]
    }

    private static func streamRequestedFinishRows(
        first: MLXGenerationBatchRowID,
        second: MLXGenerationBatchRowID,
        firstSink: RecordingBatchSink,
        secondSink: RecordingBatchSink
    ) -> [MLXContinuousBatchStreamRow] {
        [
            Self.row(id: first, sink: firstSink),
            Self.stoppingRow(id: second, sink: secondSink)
        ]
    }

    private static func makeDriver(
        initialRowIDs: [MLXGenerationBatchRowID],
        steps: [MLXContinuousBatchStreamStep],
        rows: [MLXContinuousBatchStreamRow],
        error: (any Error)? = nil
    ) throws -> MLXContinuousBatchStreamDriver<ScriptedStreamStepper> {
        try MLXContinuousBatchStreamDriver(
            stepper: ScriptedStreamStepper(
                orderedRowIDs: initialRowIDs,
                steps: steps,
                error: error
            ),
            rows: rows
        )
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

    private static func stoppingRow(
        id: MLXGenerationBatchRowID,
        sink: RecordingBatchSink
    ) -> MLXContinuousBatchStreamRow {
        MLXContinuousBatchStreamRow(
            id: id,
            sink: sink.streamSink()
        ) { tokenID, sink in
            sink.yield(LLMStreamChunk(
                text: "token-\(tokenID)",
                event: .text,
                tokenCount: 1,
                tokenIDs: [tokenID]
            ))
            return .finish(.streamRequestedStop(tokenID))
        }
    }
}

@testable import MLXLocalModels

struct ScriptedStreamStepper: MLXContinuousBatchStreamStepping {
    private(set) var orderedRowIDs: [MLXGenerationBatchRowID]
    private var error: (any Error)?
    private var steps: [MLXContinuousBatchStreamStep]

    init(
        orderedRowIDs: [MLXGenerationBatchRowID],
        steps: [MLXContinuousBatchStreamStep],
        error: (any Error)? = nil
    ) {
        self.error = error
        self.orderedRowIDs = orderedRowIDs
        self.steps = steps
    }

    mutating func stepForStreaming() throws -> MLXContinuousBatchStreamStep {
        if let error {
            throw error
        }
        let step = steps.removeFirst()
        orderedRowIDs = step.activeRowIDs
        return step
    }

    mutating func finishRows(
        _ rowIDs: Set<MLXGenerationBatchRowID>,
        reason: MLXContinuousBatchFinishReason
    ) throws -> [MLXContinuousBatchFinishedRow] {
        let finishedRows = orderedRowIDs
            .filter(rowIDs.contains)
            .map { rowID in
                MLXContinuousBatchFinishedRow(
                    rowID: rowID,
                    tokenID: -1,
                    generatedTokenCount: 0,
                    reason: reason
                )
            }
        orderedRowIDs.removeAll { rowIDs.contains($0) }
        return finishedRows
    }
}

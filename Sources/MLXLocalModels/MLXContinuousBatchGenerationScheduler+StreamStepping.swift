extension MLXContinuousBatchGenerationScheduler: MLXContinuousBatchStreamStepping {
    internal mutating func stepForStreaming() throws -> MLXContinuousBatchStreamStep {
        let result = try step()
        return MLXContinuousBatchStreamStep(
            sampledRowIDs: result.sampledTokens.rowIDs,
            sampledTokenIDs: result.sampledTokens.tokenIDs,
            finishedRows: result.finishedRows,
            activeRowIDs: result.activeRowIDs
        )
    }

    internal mutating func finishRows(
        _ rowIDs: Set<MLXGenerationBatchRowID>,
        reason: MLXContinuousBatchFinishReason
    ) throws -> [MLXContinuousBatchFinishedRow] {
        guard !rowIDs.isEmpty else {
            return []
        }
        let removedRows = coordinator.finish(ids: rowIDs)
        try decoder.realign(to: coordinator.orderedRowIDs)
        return removedRows.map { row in
            MLXContinuousBatchFinishedRow(
                rowID: row.id,
                tokenID: row.payload.previousTokenID,
                generatedTokenCount: row.payload.generatedTokenCount,
                reason: reason
            )
        }
    }
}

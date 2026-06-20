import MLX

internal struct MLXContinuousBatchGenerationScheduler<
    Decoder: MLXContinuousBatchDecodingStrategy
> {
    internal var coordinator: MLXContinuousBatchCoordinator<MLXContinuousBatchGenerationRow>
    internal var decoder: Decoder

    internal init(
        coordinator: MLXContinuousBatchCoordinator<MLXContinuousBatchGenerationRow>,
        decoder: Decoder
    ) {
        self.coordinator = coordinator
        self.decoder = decoder
    }

    internal var isEmpty: Bool {
        coordinator.isEmpty
    }

    internal var orderedRowIDs: [MLXGenerationBatchRowID] {
        coordinator.orderedRowIDs
    }

    internal mutating func step() throws -> MLXContinuousBatchGenerationStepResult {
        let expectedRowIDs = coordinator.orderedRowIDs
        guard !expectedRowIDs.isEmpty else {
            throw MLXContinuousBatchSchedulerError.emptyBatch
        }
        guard decoder.orderedRowIDs == expectedRowIDs else {
            throw MLXContinuousBatchSchedulerError.decoderRowMismatch(
                expected: expectedRowIDs,
                actual: decoder.orderedRowIDs
            )
        }

        let previousTokens = MLXArray(coordinator.orderedStates.map(\.previousTokenID))
        let sampledTokens = try decoder.step(previousTokens: previousTokens)
        try validate(sampledTokens, expectedRowIDs: expectedRowIDs)

        let finishedRows = try accept(sampledTokens)
        if !finishedRows.isEmpty {
            coordinator.finish(ids: Set(finishedRows.map(\.rowID)))
            try decoder.realign(to: coordinator.orderedRowIDs)
        }

        return MLXContinuousBatchGenerationStepResult(
            sampledTokens: sampledTokens,
            finishedRows: finishedRows,
            activeRowIDs: coordinator.orderedRowIDs
        )
    }

    private func validate(
        _ sampledTokens: MLXContinuousBatchSampledTokens,
        expectedRowIDs: [MLXGenerationBatchRowID]
    ) throws {
        guard sampledTokens.rowIDs == expectedRowIDs else {
            throw MLXContinuousBatchSchedulerError.sampledRowMismatch(
                expected: expectedRowIDs,
                actual: sampledTokens.rowIDs
            )
        }
        guard sampledTokens.tokenIDs.count == expectedRowIDs.count else {
            throw MLXContinuousBatchSchedulerError.sampledTokenCountMismatch(
                expected: expectedRowIDs.count,
                actual: sampledTokens.tokenIDs.count
            )
        }
    }

    private mutating func accept(
        _ sampledTokens: MLXContinuousBatchSampledTokens
    ) throws -> [MLXContinuousBatchFinishedRow] {
        var finishedRows: [MLXContinuousBatchFinishedRow] = []
        finishedRows.reserveCapacity(sampledTokens.rowIDs.count)

        for index in sampledTokens.rowIDs.indices {
            let rowID = sampledTokens.rowIDs[index]
            let tokenID = sampledTokens.tokenIDs[index]
            var finishedRow: MLXContinuousBatchFinishedRow?

            try coordinator.updateState(for: rowID) { state in
                if let reason = state.accept(tokenID: tokenID) {
                    finishedRow = MLXContinuousBatchFinishedRow(
                        rowID: rowID,
                        tokenID: tokenID,
                        generatedTokenCount: state.generatedTokenCount,
                        reason: reason
                    )
                }
            }

            if let finishedRow {
                finishedRows.append(finishedRow)
            }
        }

        return finishedRows
    }
}

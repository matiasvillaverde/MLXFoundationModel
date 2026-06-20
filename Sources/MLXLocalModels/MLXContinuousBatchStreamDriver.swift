internal struct MLXContinuousBatchStreamDriver<Stepper: MLXContinuousBatchStreamStepping> {
    internal private(set) var stepper: Stepper
    private var rowsByID: [MLXGenerationBatchRowID: MLXContinuousBatchStreamRow]

    internal init(
        stepper: Stepper,
        rows: [MLXContinuousBatchStreamRow]
    ) throws {
        self.stepper = stepper
        var rowsByID: [MLXGenerationBatchRowID: MLXContinuousBatchStreamRow] = [:]
        rowsByID.reserveCapacity(rows.count)
        for row in rows {
            guard rowsByID[row.id] == nil else {
                throw MLXContinuousBatchStreamDriverError.duplicateSink(row.id)
            }
            rowsByID[row.id] = row
        }
        self.rowsByID = rowsByID
    }

    internal var isEmpty: Bool {
        rowsByID.isEmpty
    }

    internal var activeRowIDs: [MLXGenerationBatchRowID] {
        stepper.orderedRowIDs
    }

    internal mutating func step() throws -> MLXContinuousBatchStreamStepSummary {
        do {
            let step = try stepper.stepForStreaming()
            return try stream(step)
        } catch {
            failActiveRows(error)
            throw error
        }
    }

    internal mutating func failActiveRows(_ error: any Error) {
        for row in rowsByID.values {
            row.fail(error)
        }
        rowsByID.removeAll(keepingCapacity: true)
    }

    private mutating func stream(
        _ step: MLXContinuousBatchStreamStep
    ) throws -> MLXContinuousBatchStreamStepSummary {
        try validate(step)

        let finishedRowsByID = Dictionary(uniqueKeysWithValues: step.finishedRows.map { ($0.rowID, $0) })
        let streamResult = try streamSampledTokens(step, finishedRowsByID: finishedRowsByID)
        let localFinishedRows = try finishStreamRequestedStops(streamResult.streamRequestedStops)
        try finishRows(localFinishedRows)
        try finishRows(step.finishedRows)
        let activeRowIDs = activeRows(
            step.activeRowIDs,
            removing: Set(localFinishedRows.map(\.rowID))
        )
        try validateActiveRows(activeRowIDs)

        return MLXContinuousBatchStreamStepSummary(
            activeRowIDs: activeRowIDs,
            finishedRows: step.finishedRows + localFinishedRows,
            streamedTokenIDsByRowID: streamResult.streamedTokenIDsByRowID
        )
    }

    private mutating func streamSampledTokens(
        _ step: MLXContinuousBatchStreamStep,
        finishedRowsByID: [MLXGenerationBatchRowID: MLXContinuousBatchFinishedRow]
    ) throws -> (
        streamRequestedStops: [MLXGenerationBatchRowID: MLXContinuousBatchFinishReason],
        streamedTokenIDsByRowID: [MLXGenerationBatchRowID: Int]
    ) {
        var stops: [MLXGenerationBatchRowID: MLXContinuousBatchFinishReason] = [:]
        var streamed: [MLXGenerationBatchRowID: Int] = [:]

        for index in step.sampledRowIDs.indices {
            let rowID = step.sampledRowIDs[index]
            let tokenID = step.sampledTokenIDs[index]
            let finishedRow = finishedRowsByID[rowID]
            guard let row = rowsByID[rowID] else {
                throw MLXContinuousBatchStreamDriverError.missingSink(rowID)
            }
            if let result = streamToken(row, tokenID: tokenID, finishedRow: finishedRow) {
                streamed[rowID] = result.streamedTokenID
                if let reason = result.stopReason {
                    stops[rowID] = reason
                }
            }
        }
        return (stops, streamed)
    }

    private func streamToken(
        _ row: MLXContinuousBatchStreamRow,
        tokenID: Int,
        finishedRow: MLXContinuousBatchFinishedRow?
    ) -> (streamedTokenID: Int, stopReason: MLXContinuousBatchFinishReason?)? {
        guard shouldStreamSampledToken(finishedRow) else {
            return nil
        }
        switch row.stream(tokenID: tokenID) {
        case .finish(let reason):
            return (tokenID, finishedRow == nil ? reason : nil)

        case .streamed:
            return (tokenID, nil)

        case .suppressed:
            return nil
        }
    }

    private func validate(_ step: MLXContinuousBatchStreamStep) throws {
        guard step.sampledRowIDs.count == step.sampledTokenIDs.count else {
            throw MLXContinuousBatchStreamDriverError.sampledTokenCountMismatch(
                expected: step.sampledRowIDs.count,
                actual: step.sampledTokenIDs.count
            )
        }
    }

    private func shouldStreamSampledToken(
        _ finishedRow: MLXContinuousBatchFinishedRow?
    ) -> Bool {
        switch finishedRow?.reason {
        case .stopToken:
            false

        case .maximumTokenCount, .streamRequestedStop, nil:
            true
        }
    }

    private mutating func finishStreamRequestedStops(
        _ stops: [MLXGenerationBatchRowID: MLXContinuousBatchFinishReason]
    ) throws -> [MLXContinuousBatchFinishedRow] {
        guard !stops.isEmpty else {
            return []
        }
        var finishedRows: [MLXContinuousBatchFinishedRow] = []
        finishedRows.reserveCapacity(stops.count)

        for (reason, rowIDs) in groupedRowIDsByReason(stops) {
            finishedRows.append(contentsOf: try stepper.finishRows(rowIDs, reason: reason))
        }
        return finishedRows
    }

    private func groupedRowIDsByReason(
        _ stops: [MLXGenerationBatchRowID: MLXContinuousBatchFinishReason]
    ) -> [(MLXContinuousBatchFinishReason, Set<MLXGenerationBatchRowID>)] {
        var groups: [(MLXContinuousBatchFinishReason, Set<MLXGenerationBatchRowID>)] = []
        for rowID in stops.keys.sorted() {
            guard let reason = stops[rowID] else {
                continue
            }
            if let index = groups.firstIndex(where: { $0.0 == reason }) {
                groups[index].1.insert(rowID)
            } else {
                groups.append((reason, [rowID]))
            }
        }
        return groups
    }

    private mutating func finishRows(
        _ finishedRows: [MLXContinuousBatchFinishedRow]
    ) throws {
        for finishedRow in finishedRows {
            guard let row = rowsByID.removeValue(forKey: finishedRow.rowID) else {
                throw MLXContinuousBatchStreamDriverError.missingSink(finishedRow.rowID)
            }
            row.finish(reason: finishedRow.reason)
        }
    }

    private func activeRows(
        _ activeRowIDs: [MLXGenerationBatchRowID],
        removing removedRowIDs: Set<MLXGenerationBatchRowID>
    ) -> [MLXGenerationBatchRowID] {
        guard !removedRowIDs.isEmpty else {
            return activeRowIDs
        }
        return activeRowIDs.filter { !removedRowIDs.contains($0) }
    }

    private func validateActiveRows(_ activeRowIDs: [MLXGenerationBatchRowID]) throws {
        let actual = rowsByID.keys.sorted()
        let expected = activeRowIDs.sorted()
        guard actual == expected else {
            throw MLXContinuousBatchStreamDriverError.activeRowMismatch(
                expected: expected,
                actual: actual
            )
        }
    }
}

extension MLXContinuousBatchCoordinator {
    @discardableResult
    internal mutating func admit(_ state: RowState) throws -> MLXGenerationBatchRowID {
        let id = allocateRowID()
        try rows.append(id: id, payload: state)
        return id
    }

    @discardableResult
    internal mutating func admitBatch(_ states: [RowState]) throws -> [MLXGenerationBatchRowID] {
        guard !states.isEmpty else {
            throw MLXContinuousBatchCoordinatorError.emptyAdmission
        }
        var ids: [MLXGenerationBatchRowID] = []
        ids.reserveCapacity(states.count)
        for state in states {
            ids.append(try admit(state))
        }
        return ids
    }

    internal mutating func updateState(
        for id: MLXGenerationBatchRowID,
        _ update: (inout RowState) throws -> Void
    ) throws {
        try rows.updatePayload(for: id, update)
    }

    internal mutating func replaceOrderedStates(_ states: [RowState]) throws {
        try rows.replaceOrderedPayloads(states)
    }

    @discardableResult
    internal mutating func finish(id: MLXGenerationBatchRowID) -> RowState? {
        _ = try? pagedKVBlocks?.detach(rowID: id)
        return rows.remove(id: id)?.payload
    }

    @discardableResult
    internal mutating func finish(ids: Set<MLXGenerationBatchRowID>) -> [MLXGenerationBatchRow<RowState>] {
        for id in ids {
            _ = try? pagedKVBlocks?.detach(rowID: id)
        }
        return rows.remove(ids: ids)
    }

    internal mutating func realign(to orderedIDs: [MLXGenerationBatchRowID]) throws {
        try rows.keep(ids: orderedIDs)
    }

    internal mutating func allocateRowID() -> MLXGenerationBatchRowID {
        defer {
            nextRowID = MLXGenerationBatchRowID(nextRowID.rawValue + 1)
        }
        return nextRowID
    }
}

extension MLXContinuousBatchLogitRows {
    internal mutating func append(
        id: MLXGenerationBatchRowID,
        row: MLXContinuousBatchLogitRow
    ) throws {
        try rows.append(id: id, payload: row)
    }

    @discardableResult
    internal mutating func remove(id: MLXGenerationBatchRowID) -> MLXContinuousBatchLogitRow? {
        rows.remove(id: id)?.payload
    }

    internal mutating func keep(ids: [MLXGenerationBatchRowID]) throws {
        try rows.keep(ids: ids)
    }
}

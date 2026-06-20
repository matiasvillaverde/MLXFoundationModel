import MLX

internal struct MLXContinuousBatchLogitRows: Sendable {
    internal var rows = MLXGenerationBatchRowTable<MLXContinuousBatchLogitRow>()

    internal var count: Int {
        rows.count
    }

    internal var orderedRowIDs: [MLXGenerationBatchRowID] {
        rows.orderedIDs
    }

    internal subscript(id: MLXGenerationBatchRowID) -> MLXContinuousBatchLogitRow? {
        rows[id]
    }
}

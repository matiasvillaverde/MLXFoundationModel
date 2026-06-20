internal struct MLXContinuousBatchPrefillResult: @unchecked Sendable {
    internal let batch: MLXContinuousBatchGenerationBatch
    internal let cache: [KVCache]
    internal let firstTokenIDs: [Int]
    internal let logitRows: MLXContinuousBatchLogitRows
}

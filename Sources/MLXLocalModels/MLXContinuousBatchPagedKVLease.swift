internal struct MLXContinuousBatchPagedKVLease: Sendable, Equatable {
    internal let rowID: MLXGenerationBatchRowID
    internal let blockIDs: [MLXPagedKVBlockID]
}

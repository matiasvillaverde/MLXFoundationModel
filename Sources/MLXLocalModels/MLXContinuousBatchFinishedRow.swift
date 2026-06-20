internal struct MLXContinuousBatchFinishedRow: Equatable, Sendable {
    internal let rowID: MLXGenerationBatchRowID
    internal let tokenID: Int
    internal let generatedTokenCount: Int
    internal let reason: MLXContinuousBatchFinishReason
}

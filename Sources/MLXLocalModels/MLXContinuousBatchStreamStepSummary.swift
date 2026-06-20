internal struct MLXContinuousBatchStreamStepSummary: Equatable, Sendable {
    internal let activeRowIDs: [MLXGenerationBatchRowID]
    internal let finishedRows: [MLXContinuousBatchFinishedRow]
    internal let streamedTokenIDsByRowID: [MLXGenerationBatchRowID: Int]
}

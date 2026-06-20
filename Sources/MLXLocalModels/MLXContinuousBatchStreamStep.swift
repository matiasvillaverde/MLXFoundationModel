internal struct MLXContinuousBatchStreamStep: Equatable, Sendable {
    internal let activeRowIDs: [MLXGenerationBatchRowID]
    internal let finishedRows: [MLXContinuousBatchFinishedRow]
    internal let sampledRowIDs: [MLXGenerationBatchRowID]
    internal let sampledTokenIDs: [Int]

    internal init(
        sampledRowIDs: [MLXGenerationBatchRowID],
        sampledTokenIDs: [Int],
        finishedRows: [MLXContinuousBatchFinishedRow],
        activeRowIDs: [MLXGenerationBatchRowID]
    ) {
        self.activeRowIDs = activeRowIDs
        self.finishedRows = finishedRows
        self.sampledRowIDs = sampledRowIDs
        self.sampledTokenIDs = sampledTokenIDs
    }

    internal var sampledTokenIDsByRowID: [MLXGenerationBatchRowID: Int] {
        Dictionary(uniqueKeysWithValues: zip(sampledRowIDs, sampledTokenIDs))
    }
}

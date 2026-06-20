internal struct MLXContinuousBatchGenerationStepResult: @unchecked Sendable {
    internal let sampledTokens: MLXContinuousBatchSampledTokens
    internal let finishedRows: [MLXContinuousBatchFinishedRow]
    internal let activeRowIDs: [MLXGenerationBatchRowID]

    internal var emittedTokenIDsByRowID: [MLXGenerationBatchRowID: Int] {
        Dictionary(
            uniqueKeysWithValues: zip(sampledTokens.rowIDs, sampledTokens.tokenIDs)
        )
    }
}

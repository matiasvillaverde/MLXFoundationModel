import MLX

internal struct MLXContinuousBatchSampledTokens: @unchecked Sendable {
    internal let rowIDs: [MLXGenerationBatchRowID]
    internal let tokenIDs: [Int]
    internal let tokenArray: MLXArray
}

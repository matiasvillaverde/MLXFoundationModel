import MLX

internal protocol MLXContinuousBatchDecodingStrategy {
    var orderedRowIDs: [MLXGenerationBatchRowID] { get }

    mutating func step(previousTokens: MLXArray) throws -> MLXContinuousBatchSampledTokens
    mutating func realign(to orderedRowIDs: [MLXGenerationBatchRowID]) throws
}

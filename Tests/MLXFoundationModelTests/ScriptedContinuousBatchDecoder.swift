import MLX
@testable import MLXLocalModels

struct ScriptedContinuousBatchDecoder: MLXContinuousBatchDecodingStrategy {
    private(set) var orderedRowIDs: [MLXGenerationBatchRowID]
    private var tokenBatches: [[Int]]

    init(
        rowIDs: [MLXGenerationBatchRowID],
        tokenBatches: [[Int]]
    ) {
        self.orderedRowIDs = rowIDs
        self.tokenBatches = tokenBatches
    }

    mutating func step(previousTokens: MLXArray) throws -> MLXContinuousBatchSampledTokens {
        let tokens = tokenBatches.removeFirst()
        return MLXContinuousBatchSampledTokens(
            rowIDs: orderedRowIDs,
            tokenIDs: tokens,
            tokenArray: MLXArray(tokens)
        )
    }

    mutating func realign(to orderedRowIDs: [MLXGenerationBatchRowID]) {
        self.orderedRowIDs = orderedRowIDs
    }
}

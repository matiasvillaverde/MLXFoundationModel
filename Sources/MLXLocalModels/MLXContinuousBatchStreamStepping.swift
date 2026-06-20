internal protocol MLXContinuousBatchStreamStepping {
    var orderedRowIDs: [MLXGenerationBatchRowID] { get }

    mutating func stepForStreaming() throws -> MLXContinuousBatchStreamStep
    mutating func finishRows(
        _ rowIDs: Set<MLXGenerationBatchRowID>,
        reason: MLXContinuousBatchFinishReason
    ) throws -> [MLXContinuousBatchFinishedRow]
}

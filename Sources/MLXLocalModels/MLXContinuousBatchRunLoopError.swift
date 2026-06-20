internal enum MLXContinuousBatchRunLoopError: Error, Equatable {
    case stepLimitExceeded(
        limit: Int,
        activeRowIDs: [MLXGenerationBatchRowID]
    )
}

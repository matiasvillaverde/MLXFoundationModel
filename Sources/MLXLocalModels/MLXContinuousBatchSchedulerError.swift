internal enum MLXContinuousBatchSchedulerError: Error, Equatable, Sendable {
    case decoderRowMismatch(
        expected: [MLXGenerationBatchRowID],
        actual: [MLXGenerationBatchRowID]
    )
    case emptyBatch
    case sampledRowMismatch(
        expected: [MLXGenerationBatchRowID],
        actual: [MLXGenerationBatchRowID]
    )
    case sampledTokenCountMismatch(expected: Int, actual: Int)
}

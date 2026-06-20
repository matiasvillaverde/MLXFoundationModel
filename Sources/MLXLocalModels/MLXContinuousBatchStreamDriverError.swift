internal enum MLXContinuousBatchStreamDriverError: Error, Equatable {
    case activeRowMismatch(expected: [MLXGenerationBatchRowID], actual: [MLXGenerationBatchRowID])
    case duplicateSink(MLXGenerationBatchRowID)
    case missingSink(MLXGenerationBatchRowID)
    case sampledTokenCountMismatch(expected: Int, actual: Int)
}

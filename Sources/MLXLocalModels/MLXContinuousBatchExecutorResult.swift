internal struct MLXContinuousBatchExecutorResult: Equatable, Sendable {
    internal let requestIDs: [MLXContinuousBatchRequestID]
    internal let rowIDs: [MLXGenerationBatchRowID]
    internal let runLoopResult: MLXContinuousBatchRunLoopResult
}

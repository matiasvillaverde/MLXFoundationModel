internal struct MLXContinuousBatchPrefillExecutorResult: Equatable, Sendable {
    internal let requestIDs: [MLXContinuousBatchRequestID]
    internal let groupResults: [MLXContinuousBatchExecutorResult]
}

internal struct MLXContinuousBatchRunLoopResult: Equatable, Sendable {
    internal let finishedRows: [MLXContinuousBatchFinishedRow]
    internal let stepCount: Int
    internal let streamedTokenCount: Int
}

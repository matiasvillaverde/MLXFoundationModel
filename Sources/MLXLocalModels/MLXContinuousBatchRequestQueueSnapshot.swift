internal struct MLXContinuousBatchRequestQueueSnapshot: Equatable, Sendable {
    internal let isClosed: Bool
    internal let pendingCount: Int
    internal let waitingConsumerCount: Int
}

internal struct MLXContinuousBatchExecutorConfig: Equatable, Hashable, Sendable {
    internal let maxBatchSize: Int
    internal let pagedKVBlockCapacity: Int
    internal let runLoopConfiguration: MLXContinuousBatchRunLoopConfiguration

    internal static let `default` = Self()

    internal init(
        maxBatchSize: Int = 1,
        pagedKVBlockCapacity: Int = 0,
        runLoopConfiguration: MLXContinuousBatchRunLoopConfiguration = .default
    ) {
        self.maxBatchSize = max(1, maxBatchSize)
        self.pagedKVBlockCapacity = max(0, pagedKVBlockCapacity)
        self.runLoopConfiguration = runLoopConfiguration
    }
}

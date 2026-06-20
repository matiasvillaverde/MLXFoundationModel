internal struct MLXContinuousBatchRunLoopConfiguration: Equatable, Hashable, Sendable {
    internal let maximumStepCount: Int

    internal static let `default` = Self()

    internal init(maximumStepCount: Int = 4_096) {
        self.maximumStepCount = max(1, maximumStepCount)
    }
}

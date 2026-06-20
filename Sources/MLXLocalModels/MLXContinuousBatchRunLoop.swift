internal struct MLXContinuousBatchRunLoop<Stepper: MLXContinuousBatchStreamStepping> {
    internal private(set) var driver: MLXContinuousBatchStreamDriver<Stepper>
    internal let configuration: MLXContinuousBatchRunLoopConfiguration

    internal init(
        driver: MLXContinuousBatchStreamDriver<Stepper>,
        configuration: MLXContinuousBatchRunLoopConfiguration = .default
    ) {
        self.configuration = configuration
        self.driver = driver
    }

    internal mutating func run() throws -> MLXContinuousBatchRunLoopResult {
        try run {
            Task.isCancelled
        }
    }

    internal mutating func run(
        shouldCancel: @Sendable () -> Bool
    ) throws -> MLXContinuousBatchRunLoopResult {
        var finishedRows: [MLXContinuousBatchFinishedRow] = []
        var stepCount = 0
        var streamedTokenCount = 0

        while !driver.isEmpty {
            try checkCancellation(shouldCancel)
            try checkStepLimit(stepCount)

            let summary = try driver.step()
            stepCount += 1
            streamedTokenCount += summary.streamedTokenIDsByRowID.count
            finishedRows.append(contentsOf: summary.finishedRows)
        }

        return MLXContinuousBatchRunLoopResult(
            finishedRows: finishedRows,
            stepCount: stepCount,
            streamedTokenCount: streamedTokenCount
        )
    }

    private mutating func checkCancellation(
        _ shouldCancel: @Sendable () -> Bool
    ) throws {
        guard shouldCancel() else {
            return
        }
        let error = CancellationError()
        driver.failActiveRows(error)
        throw error
    }

    private mutating func checkStepLimit(_ stepCount: Int) throws {
        guard stepCount < configuration.maximumStepCount else {
            let error = MLXContinuousBatchRunLoopError.stepLimitExceeded(
                limit: configuration.maximumStepCount,
                activeRowIDs: driver.activeRowIDs
            )
            driver.failActiveRows(error)
            throw error
        }
    }
}

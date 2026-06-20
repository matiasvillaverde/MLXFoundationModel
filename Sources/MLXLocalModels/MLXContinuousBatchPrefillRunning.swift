internal protocol MLXContinuousBatchPrefillRunning {
    func run(
        requests: [MLXContinuousBatchPrefillRequest]
    ) throws -> MLXContinuousBatchPrefillResult
}

@testable import MLXLocalModels

struct FailingPrefillRunner: MLXContinuousBatchPrefillRunning {
    func run(
        requests _: [MLXContinuousBatchPrefillRequest]
    ) throws -> MLXContinuousBatchPrefillResult {
        throw ScriptedStreamStepError.failed
    }
}

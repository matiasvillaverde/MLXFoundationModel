@testable import MLXLocalModels

struct ScriptedPrefillRunner: MLXContinuousBatchPrefillRunning {
    func run(
        requests: [MLXContinuousBatchPrefillRequest]
    ) throws -> MLXContinuousBatchPrefillResult {
        let activeRequests = requests.enumerated().map { index, request in
            request.continuousBatchGenerationRequest(
                previousTokenID: 10 + index,
                generatedTokenCount: 1
            )
        }
        return MLXContinuousBatchPrefillResult(
            batch: try MLXContinuousBatchAssembler.assemble(requests: activeRequests),
            cache: [],
            firstTokenIDs: activeRequests.map(\.generationRow.previousTokenID),
            logitRows: MLXContinuousBatchLogitRows()
        )
    }
}

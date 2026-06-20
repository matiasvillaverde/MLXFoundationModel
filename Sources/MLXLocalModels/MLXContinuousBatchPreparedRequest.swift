import Foundation

internal struct MLXContinuousBatchPreparedRequest: Sendable {
    internal let completion: MLXContinuousBatchStreamCompletion
    internal let parameters: GenerateParameters
    internal let prefillRequest: MLXContinuousBatchPrefillRequest
    internal let promptCacheReusedTokenCount: Int
    internal let promptStartTime: ContinuousClock.Instant
    internal let promptTokenCount: Int
    internal let state: GenerationState
}

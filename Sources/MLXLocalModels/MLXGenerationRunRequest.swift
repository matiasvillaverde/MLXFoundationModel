import Foundation

internal struct MLXGenerationRunRequest {
    let input: LLMInput
    let parameters: GenerateParameters
    let runtimePreferences: ModelRuntimePreferences
    let generationStartTime: ContinuousClock.Instant
    let continuation: AsyncThrowingStream<LLMStreamChunk, Error>.Continuation
}

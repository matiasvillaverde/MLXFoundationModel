@testable import MLXLocalModels

struct TextEmitterHarness {
    let context: MLXStreamTextEmitter.Context
    let continuation: AsyncThrowingStream<LLMStreamChunk, Error>.Continuation
    let state: GenerationState
    let stream: AsyncThrowingStream<LLMStreamChunk, Error>
}

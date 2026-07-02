import Foundation
@testable import MLXLocalModels

extension MLXRealModelHarness {
    static func collectGeneration(
        from stream: AsyncThrowingStream<LLMStreamChunk, Error>
    ) async throws -> GenerationResult {
        var text = ""
        var textChunkCount = 0
        var metrics: ChunkMetrics?
        var lifecycleEvents: [StreamLifecycleEvent] = []
        for try await chunk in stream {
            if case .text = chunk.event {
                text += chunk.text
                if !chunk.text.isEmpty {
                    textChunkCount += 1
                }
            }
            if case .lifecycle(let event) = chunk.event {
                lifecycleEvents.append(event)
            }
            metrics = chunk.metrics ?? metrics
        }
        return GenerationResult(
            text: text,
            textChunkCount: textChunkCount,
            metrics: metrics,
            lifecycleEvents: lifecycleEvents
        )
    }
}

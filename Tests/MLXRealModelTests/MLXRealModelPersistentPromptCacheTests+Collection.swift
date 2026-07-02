import Foundation
@testable import MLXLocalModels

extension MLXRealModelPersistentPromptCacheTests {
    static func preload(
        session: any MLXGeneratingSession,
        configuration: ProviderConfiguration
    ) async throws {
        let progress = await session.preload(configuration: configuration)
        for try await _ in progress {
            // Consume preload progress before generation.
        }
    }

    static func collectGeneration(
        from stream: AsyncThrowingStream<LLMStreamChunk, Error>
    ) async throws -> MLXRealModelHarness.GenerationResult {
        var text = ""
        var textChunkCount = 0
        var metrics: ChunkMetrics?
        var lifecycleEvents: [StreamLifecycleEvent] = []
        for try await chunk in stream {
            if case .text = chunk.event {
                text += chunk.text
                textChunkCount += chunk.text.isEmpty ? 0 : 1
            }
            if case .lifecycle(let event) = chunk.event {
                lifecycleEvents.append(event)
            }
            metrics = chunk.metrics ?? metrics
        }
        return .init(
            text: text,
            textChunkCount: textChunkCount,
            metrics: metrics,
            lifecycleEvents: lifecycleEvents
        )
    }
}

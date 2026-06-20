import Foundation
import MLXLocalModels

struct MLXStreamEventTranslator: Sendable {
    func translate(
        _ stream: AsyncThrowingStream<LLMStreamChunk, Error>,
        into sink: any MLXStreamEventSink,
        tools: [MLXBridgeToolDefinition],
        promptStyle: MLXPromptStyle? = nil,
        reasoningStartsOpen: Bool = false
    ) async throws {
        var reducer = MLXToolAwareStreamReducer(tools: tools, promptStyle: promptStyle)
        var eventBuilder = MLXTranslatedStreamEventBuilder(reasoningStartsOpen: reasoningStartsOpen)
        for try await chunk in stream {
            try Task.checkCancellation()
            for action in reducer.consume(chunk) {
                for event in eventBuilder.events(from: action) {
                    await sink.send(event)
                }
            }
        }
        for action in reducer.finish() {
            for event in eventBuilder.events(from: action) {
                await sink.send(event)
            }
        }
        for event in eventBuilder.finish() {
            await sink.send(event)
        }
        await sink.finish()
    }
}

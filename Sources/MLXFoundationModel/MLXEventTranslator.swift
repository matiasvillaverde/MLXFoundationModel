#if FOUNDATION_MODELS_PROVIDER_API && canImport(FoundationModels)
import Foundation
import FoundationModels
import MLXLocalModels

@available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
struct MLXEventTranslator: Sendable {
    private let responseEntryID = UUID().uuidString
    private let toolCallsEntryID = UUID().uuidString

    func translate(
        _ stream: AsyncThrowingStream<LLMStreamChunk, Error>,
        into channel: LanguageModelExecutorGenerationChannel,
        tools: [MLXBridgeToolDefinition],
        promptStyle: MLXPromptStyle? = nil,
        reasoningStartsOpen: Bool = false
    ) async throws {
        try await MLXStreamEventTranslator().translate(
            stream,
            into: FoundationModelsStreamEventSink(
                channel: channel,
                responseEntryID: responseEntryID,
                toolCallsEntryID: toolCallsEntryID
            ),
            tools: tools,
            promptStyle: promptStyle,
            reasoningStartsOpen: reasoningStartsOpen
        )
    }
}
#endif

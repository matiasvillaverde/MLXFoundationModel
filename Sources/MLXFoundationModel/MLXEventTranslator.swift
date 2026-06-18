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
        toolDefinitionsEnabled: Bool
    ) async throws {
        var buffered = ""
        for try await chunk in stream {
            try Task.checkCancellation()
            guard !chunk.text.isEmpty else {
                continue
            }
            buffered += chunk.text
            await channel.send(
                .response(
                    entryID: responseEntryID,
                    action: .appendText(chunk.text, tokenCount: 1)
                )
            )
        }

        guard
            toolDefinitionsEnabled,
            let call = MLXToolCallExtractor.extract(from: buffered)
        else {
            return
        }
        await channel.send(
            .toolCalls(
                entryID: toolCallsEntryID,
                action: .toolCall(
                    id: UUID().uuidString,
                    name: call.name,
                    action: .appendArguments(call.argumentsJSON, tokenCount: 1)
                )
            )
        )
    }
}
#endif

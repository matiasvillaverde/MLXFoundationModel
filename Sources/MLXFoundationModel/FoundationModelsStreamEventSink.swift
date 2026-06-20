#if FOUNDATION_MODELS_PROVIDER_API && canImport(FoundationModels)
import Foundation
import FoundationModels
import os

@available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
struct FoundationModelsStreamEventSink: MLXStreamEventSink {
    let channel: LanguageModelExecutorGenerationChannel
    let responseEntryID: String
    let toolCallsEntryID: String
    private let reasoningEntryID = UUID().uuidString
    private let state = OSAllocatedUnfairLock(
        initialState: FoundationModelsStreamSinkState()
    )

    init(
        channel: LanguageModelExecutorGenerationChannel,
        responseEntryID: String,
        toolCallsEntryID: String
    ) {
        self.channel = channel
        self.responseEntryID = responseEntryID
        self.toolCallsEntryID = toolCallsEntryID
    }

    func send(_ event: MLXTranslatedStreamEvent) async {
        let events = state.withLock { state in
            state.events(for: event, entryIDs: entryIDs)
        }
        for event in events {
            await event.send(into: channel)
        }
    }

    func finish() async {
        let events = state.withLock { state in
            state.finish(entryIDs: entryIDs)
        }
        for event in events {
            await event.send(into: channel)
        }
    }

    private var entryIDs: FoundationModelsStreamEntryIDs {
        FoundationModelsStreamEntryIDs(
            response: responseEntryID,
            reasoning: reasoningEntryID,
            toolCalls: toolCallsEntryID
        )
    }
}
#endif

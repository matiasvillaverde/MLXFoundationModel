#if FOUNDATION_MODELS_PROVIDER_API && canImport(FoundationModels)
import Foundation
import FoundationModels
import MLXLocalModels

@available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
struct FoundationModelsStreamSinkState: Sendable {
    private var thinkSplitter = MLXThinkTagStreamSplitter()
    private var reasoningTokenCount = 0

    mutating func events(
        for event: MLXTranslatedStreamEvent,
        entryIDs: FoundationModelsStreamEntryIDs
    ) -> [FoundationModelsPendingChannelEvent] {
        switch event {
        case let .responseText(text, tokenCount):
            return splitTextEvents(text, tokenCount: tokenCount, entryIDs: entryIDs)

        case let .reasoningText(text, tokenCount):
            return finish(entryIDs: entryIDs) + [
                directReasoningTextEvent(text, tokenCount: tokenCount, entryIDs: entryIDs)
            ]

        case .responseUsage(let usage):
            return finish(entryIDs: entryIDs) + [
                responseUsageEvent(usage, entryIDs: entryIDs)
            ]

        case let .toolCall(call, tokenCount):
            return finish(entryIDs: entryIDs) + [
                toolCallEvent(call, tokenCount: tokenCount, entryIDs: entryIDs)
            ]

        case .toolUsage(let usage):
            return finish(entryIDs: entryIDs) + [
                toolUsageEvent(usage, entryIDs: entryIDs)
            ]
        }
    }

    mutating func finish(
        entryIDs: FoundationModelsStreamEntryIDs
    ) -> [FoundationModelsPendingChannelEvent] {
        sendSegments(thinkSplitter.finish(), sourceTokenCount: 0, entryIDs: entryIDs)
    }

    private mutating func splitTextEvents(
        _ text: String,
        tokenCount: Int,
        entryIDs: FoundationModelsStreamEntryIDs
    ) -> [FoundationModelsPendingChannelEvent] {
        sendSegments(
            thinkSplitter.consume(text),
            sourceTokenCount: tokenCount,
            entryIDs: entryIDs
        )
    }

    private mutating func sendSegments(
        _ segments: [MLXThinkTagStreamSplitter.Segment],
        sourceTokenCount: Int,
        entryIDs: FoundationModelsStreamEntryIDs
    ) -> [FoundationModelsPendingChannelEvent] {
        allocateTokens(to: segments, sourceTokenCount: sourceTokenCount).map { item in
            switch item.segment.kind {
            case .response:
                return responseTextEvent(
                    item.segment.text,
                    tokenCount: item.tokenCount,
                    entryIDs: entryIDs
                )

            case .reasoning:
                reasoningTokenCount += item.tokenCount
                return reasoningTextEvent(
                    item.segment.text,
                    tokenCount: item.tokenCount,
                    entryIDs: entryIDs
                )
            }
        }
    }

    private func allocateTokens(
        to segments: [MLXThinkTagStreamSplitter.Segment],
        sourceTokenCount: Int
    ) -> [(segment: MLXThinkTagStreamSplitter.Segment, tokenCount: Int)] {
        let tokenBudget = max(sourceTokenCount, segments.count)
        let totalLength = segments.reduce(0) { $0 + max($1.text.utf8.count, 1) }
        guard totalLength > 0 else {
            return []
        }
        return allocatedSegments(segments, tokenBudget: tokenBudget, totalLength: totalLength)
    }

    private func allocatedSegments(
        _ segments: [MLXThinkTagStreamSplitter.Segment],
        tokenBudget: Int,
        totalLength: Int
    ) -> [(segment: MLXThinkTagStreamSplitter.Segment, tokenCount: Int)] {
        var remainingTokens = tokenBudget
        var remainingLength = totalLength
        return segments.enumerated().map { index, segment in
            let length = max(segment.text.utf8.count, 1)
            let tokenCount = allocatedTokenCount(
                length: length,
                remainingLength: remainingLength,
                remainingTokens: remainingTokens,
                remainingSegments: segments.count - index
            )
            remainingTokens -= tokenCount
            remainingLength -= length
            return (segment, tokenCount)
        }
    }

    private func allocatedTokenCount(
        length: Int,
        remainingLength: Int,
        remainingTokens: Int,
        remainingSegments: Int
    ) -> Int {
        guard remainingSegments > 1 else {
            return max(remainingTokens, 1)
        }
        let proportional = Int((Double(length) / Double(remainingLength)) * Double(remainingTokens))
        let maximum = max(remainingTokens - (remainingSegments - 1), 1)
        return min(max(proportional, 1), maximum)
    }

    private func responseTextEvent(
        _ text: String,
        tokenCount: Int,
        entryIDs: FoundationModelsStreamEntryIDs
    ) -> FoundationModelsPendingChannelEvent {
        .response(.response(
            entryID: entryIDs.response,
            action: .appendText(text, tokenCount: max(tokenCount, 1))
        ))
    }

    private func reasoningTextEvent(
        _ text: String,
        tokenCount: Int,
        entryIDs: FoundationModelsStreamEntryIDs
    ) -> FoundationModelsPendingChannelEvent {
        .reasoning(.reasoning(
            entryID: entryIDs.reasoning,
            action: .appendText(text, tokenCount: max(tokenCount, 1))
        ))
    }

    private mutating func directReasoningTextEvent(
        _ text: String,
        tokenCount: Int,
        entryIDs: FoundationModelsStreamEntryIDs
    ) -> FoundationModelsPendingChannelEvent {
        reasoningTokenCount += max(tokenCount, 1)
        return reasoningTextEvent(text, tokenCount: tokenCount, entryIDs: entryIDs)
    }

    private func toolCallEvent(
        _ call: MLXExtractedToolCall,
        tokenCount: Int,
        entryIDs: FoundationModelsStreamEntryIDs
    ) -> FoundationModelsPendingChannelEvent {
        .toolCalls(.toolCalls(
            entryID: entryIDs.toolCalls,
            action: .toolCall(
                id: UUID().uuidString,
                name: call.name,
                action: .appendArguments(call.argumentsJSON, tokenCount: max(tokenCount, 1))
            )
        ))
    }

    private func responseUsageEvent(
        _ usage: UsageMetrics,
        entryIDs: FoundationModelsStreamEntryIDs
    ) -> FoundationModelsPendingChannelEvent {
        .response(.response(
            entryID: entryIDs.response,
            action: .updateUsage(
                input: inputUsage(from: usage),
                output: outputUsage(from: usage)
            )
        ))
    }

    private func toolUsageEvent(
        _ usage: UsageMetrics,
        entryIDs: FoundationModelsStreamEntryIDs
    ) -> FoundationModelsPendingChannelEvent {
        .toolCalls(.toolCalls(
            entryID: entryIDs.toolCalls,
            action: .updateUsage(
                input: inputUsage(from: usage),
                output: outputUsage(from: usage)
            )
        ))
    }

    private func inputUsage(
        from usage: UsageMetrics
    ) -> LanguageModelExecutorGenerationChannel.Usage.Input {
        LanguageModelExecutorGenerationChannel.Usage.Input(
            totalTokenCount: usage.promptTokens ?? max(usage.totalTokens - usage.generatedTokens, 0),
            cachedTokenCount: usage.promptCacheReusedTokenCount ?? 0
        )
    }

    private func outputUsage(
        from usage: UsageMetrics
    ) -> LanguageModelExecutorGenerationChannel.Usage.Output {
        LanguageModelExecutorGenerationChannel.Usage.Output(
            totalTokenCount: usage.generatedTokens,
            reasoningTokenCount: min(reasoningTokenCount, usage.generatedTokens)
        )
    }
}
#endif
